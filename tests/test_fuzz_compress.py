"""Fuzz / hostile-input tests for the compress() never-raise contract (#116).

The audit's fuzzing broke ``compress()`` with three families — deeply nested
JSON (RecursionError), lone UTF-16 surrogates (UnicodeEncodeError) and non-dict
messages (TypeError). These tests pin the contract: for any hostile input,
``compress()`` returns the input intact (never mutated), never raises, and the
output is serializable; genuinely unrecoverable blocks are annotated on
``stats.error``.
"""

from __future__ import annotations

import copy
import json

import pytest

from tokenslim import compress
from tokenslim import router as router_module

# --- The three audit reproducers -------------------------------------------

_DEEP_JSON_WALKER = "[" * 2000 + "]" * 2000  # parses; walker depth-limit path
_DEEP_JSON_LOADS = "[" * 20000 + "]" * 20000  # json.loads itself RecursionErrors
_LONE_SURROGATE = "before \ud800 after " * 40  # unpaired UTF-16 surrogate
_SURROGATE_IN_JSON = json.dumps({"k": "v"})[:-2] + "\ud800" + '"}'


def _deep_dict(depth: int) -> dict:
    node: dict = {"leaf": "value here padded out to be routable " * 3}
    for _ in range(depth):
        node = {"child": node}
    return node


_DEEP_DICT_JSON = json.dumps(_deep_dict(500))

_HOSTILE_MESSAGES: list[list[object]] = [
    # deeply nested JSON — both the walker-limited and json.loads-breaking depths
    [{"role": "user", "content": _DEEP_JSON_WALKER}],
    [{"role": "user", "content": _DEEP_JSON_LOADS}],
    # lone surrogates in a plain string and inside a JSON-looking string
    [{"role": "user", "content": _LONE_SURROGATE}],
    [{"role": "user", "content": _SURROGATE_IN_JSON}],
    # non-dict messages in the array
    ["a bare string message that is long enough to route " * 5],
    [42, {"role": "user", "content": "y" * 400}],
    [None],
    [3.14, True, {"role": "user", "content": "z" * 400}],
    [["nested", "list", "as", "message"]],
    # odd content shapes
    [{"role": "user", "content": None}],
    [{"role": "user", "content": 123}],
    [{"role": "user"}],  # no content key at all
    [{}],
    [{"role": "user", "content": [{"type": "text"}, "loose", 7, None]}],
    [{"role": "user", "content": [{"type": "text", "text": _LONE_SURROGATE}]}],
    # a deeply nested dict (not just arrays)
    [{"role": "user", "content": _DEEP_DICT_JSON}],
]


@pytest.mark.parametrize("messages", _HOSTILE_MESSAGES)
def test_compress_never_raises_on_hostile_input(messages):
    snapshot = copy.deepcopy(messages)
    out, stats = compress(messages, min_bytes=0)
    # 1. input is never mutated
    assert messages == snapshot
    # 2. output is serializable (default ensure_ascii escapes lone surrogates)
    json.dumps(out)
    # 3. token accounting stays sane (never inflates, non-negative)
    assert stats.new_tokens <= stats.orig_tokens
    assert stats.orig_tokens >= 0


def test_deep_json_that_breaks_json_loads_sets_error():
    """The 20k-deep array RecursionErrors in json.loads → barrier annotates it."""
    out, stats = compress([{"role": "user", "content": _DEEP_JSON_LOADS}], min_bytes=0)
    assert stats.error is not None
    assert "RecursionError" in stats.error
    # passthrough: nothing removed
    assert stats.new_tokens == stats.orig_tokens


def test_non_dict_messages_pass_through_untouched():
    messages = [42, None, "loose", {"role": "user", "content": "compress me " * 40}]
    out, stats = compress(messages, min_bytes=0)
    assert out[0] == 42
    assert out[1] is None
    assert out[2] == "loose"
    # the real message is still processed
    assert isinstance(out[3], dict)


def test_surrogates_do_not_raise_and_survive():
    messages = [{"role": "user", "content": _LONE_SURROGATE}]
    out, stats = compress(messages, min_bytes=0)
    assert isinstance(out[0]["content"], str)
    json.dumps(out)  # serializable


def test_perimeter_barrier_annotates_and_passes_through(monkeypatch):
    """Any unforeseen error inside routing degrades to passthrough + stats.error."""

    def boom(self, text):
        raise ValueError("synthetic compressor explosion")

    monkeypatch.setattr(router_module.ContentRouter, "route", boom)
    messages = [{"role": "user", "content": "x" * 500}]
    snapshot = copy.deepcopy(messages)
    out, stats = compress(messages, min_bytes=0)

    assert out == snapshot  # input returned intact
    assert stats.error is not None
    assert "synthetic compressor explosion" in stats.error
    assert stats.new_tokens == stats.orig_tokens


def test_clean_run_has_no_error():
    out, stats = compress([{"role": "user", "content": "hello " * 100}], min_bytes=0)
    assert stats.error is None
