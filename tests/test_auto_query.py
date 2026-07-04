"""Auto-derivation of ``Config.query`` from the last user message (issue #124).

``compress()`` defaults ``query="auto"``: it mines the relevance query from the
most recent ``role="user"`` turn so query-aware compressors (SmartCrusher
JSON-match, SearchCompressor BM25, LogCompressor, TextCompressor) keep the rows
the user is actually asking about. ``query=None`` is the documented opt-out and
reproduces the pre-#124 behavior exactly.
"""

from __future__ import annotations

import json

from tokenslim import compress
from tokenslim.compress import _QUERY_MAX_CHARS, _derive_query, _resolve_query
from tokenslim.config import AUTO_QUERY, Config

_MARKER = "zzxqmarker"


def _rows_with_marker(n: int = 500, marker_index: int = 250) -> str:
    """JSON array where only one middle row (not head/tail) mentions the marker."""
    rows = [{"id": i, "status": "ok", "note": f"routine entry number {i}"} for i in range(n)]
    rows[marker_index] = {"id": marker_index, "status": "ok", "note": f"{_MARKER} checkpoint"}
    return json.dumps(rows)


def _convo(question: str) -> list[dict]:
    return [
        {"role": "user", "content": question},
        {"role": "tool", "tool_call_id": "t1", "content": _rows_with_marker()},
    ]


# --- _derive_query ----------------------------------------------------------


def test_derive_picks_last_user_message() -> None:
    messages = [
        {"role": "user", "content": "first question about apples"},
        {"role": "assistant", "content": "some answer"},
        {"role": "user", "content": "final question about bananas"},
    ]
    assert _derive_query(messages) == "final question about bananas"


def test_derive_reads_text_blocks_not_tool_results() -> None:
    # Anthropic tool results ride in a role="user" message as tool_result
    # blocks; that payload is data, not the question, so it must be skipped and
    # the earlier human turn used instead.
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "where is the widget?"}]},
        {"role": "user", "content": [{"type": "tool_result", "content": "huge blob of data"}]},
    ]
    assert _derive_query(messages) == "where is the widget?"


def test_derive_truncates_to_budget() -> None:
    long = "x" * (_QUERY_MAX_CHARS + 100)
    q = _derive_query([{"role": "user", "content": long}])
    assert q is not None
    assert len(q) == _QUERY_MAX_CHARS


def test_derive_none_without_user_message() -> None:
    assert _derive_query([{"role": "tool", "content": "x"}]) is None
    assert _derive_query([]) is None
    assert _derive_query("not a list") is None  # type: ignore[arg-type]


# --- _resolve_query semantics: auto / None / forced -------------------------


def test_default_query_is_auto_sentinel() -> None:
    assert Config().query == AUTO_QUERY


def test_resolve_auto_derives_from_user() -> None:
    cfg, source = _resolve_query(Config(), [{"role": "user", "content": "hello there"}])
    assert cfg.query == "hello there"
    assert source == 0, "the derived-from user turn is reported for self-filter exclusion"


def test_resolve_auto_without_user_becomes_none() -> None:
    # The sentinel must never leak to compressors as a literal "auto".
    cfg, source = _resolve_query(Config(), [{"role": "tool", "content": "x"}])
    assert cfg.query is None
    assert source is None


def test_resolve_none_is_untouched() -> None:
    cfg, source = _resolve_query(Config(query=None), [{"role": "user", "content": "hello"}])
    assert cfg.query is None
    assert source is None


def test_resolve_explicit_query_is_forced() -> None:
    # A forced query applies to every message (no self-filter exclusion).
    cfg, source = _resolve_query(Config(query="forced query"), [{"role": "user", "content": "hi"}])
    assert cfg.query == "forced query"
    assert source is None


# --- end-to-end: the referenced row survives compression --------------------


def test_referenced_row_survives_auto_query() -> None:
    out, stats = compress(_convo(f"what happened with {_MARKER}?"), options=Config(min_bytes=0))
    tool_content = out[1]["content"]
    assert _MARKER in tool_content, "row referenced by the user must survive compression"
    assert stats.new_tokens < stats.orig_tokens, "the 500-row array is still crushed"


def test_query_none_reproduces_pre_124_behavior() -> None:
    # Opt-out: with derivation off the middle row is crushed away, exactly as
    # before #124 (it is neither head/tail, error, rare, nor an outlier).
    out, _ = compress(
        _convo(f"what happened with {_MARKER}?"), options=Config(min_bytes=0, query=None)
    )
    assert _MARKER not in out[1]["content"]


def test_no_user_message_costs_nothing_and_crushes() -> None:
    # No user turn => sentinel resolves to None => no query, no error, and the
    # array is crushed just like the opt-out path.
    messages = [{"role": "tool", "tool_call_id": "t1", "content": _rows_with_marker()}]
    out, stats = compress(messages, options=Config(min_bytes=0))
    assert stats.error is None
    assert _MARKER not in out[0]["content"]


def test_auto_query_does_not_mutate_input() -> None:
    messages = _convo(f"tell me about {_MARKER}")
    snapshot = json.loads(messages[1]["content"])
    compress(messages, options=Config(min_bytes=0))
    assert json.loads(messages[1]["content"]) == snapshot
