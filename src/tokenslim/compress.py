"""Public ``compress()`` entry point and result schema.

Walks an OpenAI/Anthropic-style message array, routes each large text block
through the :class:`ContentRouter`, rewrites the messages in place (on a copy),
and returns the new array alongside :class:`CompressionStats`.

Two contract guarantees are enforced here (issues #116 and #117):

* **Never raise** — any unforeseen error degrades to a passthrough of the input
  with :attr:`CompressionStats.error` annotated; the exception is logged, never
  propagated (drop-in integration wrappers must never crash the host app).
* **Never inflate** — a compressed block is kept only when it saves tokens net
  of its CCR marker cost; otherwise it is reverted, so per-block (and therefore
  aggregate) ``new_tokens <= orig_tokens`` always holds.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .config import AUTO_QUERY, Config, load_config
from .detector import ContentType
from .router import ContentRouter
from .tokenizer import count_tokens

if TYPE_CHECKING:
    from .store import CCRStore

__all__ = ["compress", "CompressionStats", "BlockStat"]

logger = logging.getLogger("tokenslim")

Message = dict[str, Any]

# Cap on the auto-derived relevance query (issue #124). A few hundred chars is
# plenty for BM25/anchor matching while bounding the cost on huge user turns.
_QUERY_MAX_CHARS = 512


def _safe_count(text: str, model: str | None) -> int:
    """Count tokens without ever raising (real tokenizers choke on surrogates).

    Falls back to a coarse chars/4 estimate so the never-raise contract holds
    even when the active tokenizer rejects the input.
    """
    try:
        return count_tokens(text, model)
    except Exception:
        return max(1, len(text) // 4)


def _copy_messages(messages: list[Message]) -> list[Any]:
    """Copy the array for rewriting without mutating the caller's input.

    Only dict messages are deep-copied (their ``content`` may be rewritten);
    non-dict entries are passed through *by reference* so framework message
    objects keep their identity and are truly left intact (issue #116).
    """
    if not isinstance(messages, list):
        return copy.deepcopy(messages)
    return [copy.deepcopy(msg) if isinstance(msg, dict) else msg for msg in messages]


@dataclass(frozen=True)
class BlockStat:
    """Per-block compression detail."""

    message_index: int
    block_path: str
    content_type: ContentType
    confidence: float
    compressor: str
    orig_tokens: int
    new_tokens: int
    changed: bool
    skipped: bool


@dataclass
class CompressionStats:
    """Aggregate result of a compression run."""

    orig_tokens: int = 0
    new_tokens: int = 0
    blocks: list[BlockStat] = field(default_factory=list)
    # The CCR store holding dropped originals (when CCR is enabled). Pass its
    # hashes to tokenslim.retrieve.retrieve() / use it with CCRContext.
    store: CCRStore | None = None
    # Set when the never-raise barrier caught an unexpected error and degraded
    # to passthrough; None on a clean run. Informative, not fatal.
    error: str | None = None

    @property
    def ratio(self) -> float:
        """Fraction of tokens removed (0.0 = no change, 1.0 = everything)."""
        if self.orig_tokens == 0:
            return 0.0
        return 1.0 - (self.new_tokens / self.orig_tokens)

    @property
    def saved_tokens(self) -> int:
        return self.orig_tokens - self.new_tokens

    def note_error(self, exc: BaseException) -> None:
        """Record an exception the barrier swallowed (joined if several)."""
        message = f"{type(exc).__name__}: {exc}"
        self.error = message if self.error is None else f"{self.error}; {message}"


def _rewrite_content(
    content: Any,
    router: ContentRouter,
    config: Config,
    msg_index: int,
    stats: CompressionStats,
) -> Any:
    """Compress the text in a message's ``content`` (string or block list)."""
    if isinstance(content, str):
        return _rewrite_text(content, router, config, msg_index, "content", stats)

    if isinstance(content, list):
        new_blocks = []
        for i, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                block = dict(block)
                block["text"] = _rewrite_text(
                    block["text"], router, config, msg_index, f"content[{i}].text", stats
                )
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                block = dict(block)
                inner = block.get("content", "")
                if isinstance(inner, str):
                    block["content"] = _rewrite_text(
                        inner, router, config, msg_index, f"content[{i}].tool_result", stats
                    )
            new_blocks.append(block)
        return new_blocks

    return content


def _rewrite_text(
    text: str,
    router: ContentRouter,
    config: Config,
    msg_index: int,
    path: str,
    stats: CompressionStats,
) -> str:
    """Route a single text string, apply the token guard, record its stats.

    Two safety rails run here:

    * **Never inflate** (#117) — the routed text is kept only if it saves more
      than ``config.min_token_savings`` tokens net; otherwise the block reverts
      to its original and is recorded as an unchanged passthrough.
    * **Never raise** (#116) — if a compressor throws, the block degrades to a
      passthrough and the error is noted on ``stats``.
    """
    if not isinstance(text, str) or not text:
        return text

    model = config.model
    orig_tokens = _safe_count(text, model)

    content_type = ContentType.TEXT
    confidence = 0.0
    compressor = "passthrough"
    out_text = text
    new_tokens = orig_tokens
    changed = False
    skipped = False

    try:
        result = router.route(text)
        content_type = result.content_type
        confidence = result.confidence
        compressor = result.compressor
        skipped = result.skipped
        if result.changed:
            candidate_tokens = _safe_count(result.text, model)
            # Token guard: keep the compression only when it is a net token win
            # (CCR markers can shrink characters while growing tokens, #117).
            if orig_tokens - candidate_tokens > config.min_token_savings:
                out_text = result.text
                new_tokens = candidate_tokens
                changed = True
            else:
                # Not worth it — revert to the original, record as passthrough.
                out_text = text
                new_tokens = orig_tokens
                changed = False
                skipped = True
    except Exception as exc:
        # Never-raise: degrade this block to a passthrough, keep going.
        logger.warning(
            "tokenslim: block %s in message %d degraded to passthrough (%s)",
            path,
            msg_index,
            type(exc).__name__,
        )
        stats.note_error(exc)
        out_text = text
        new_tokens = orig_tokens
        changed = False
        skipped = True

    stats.orig_tokens += orig_tokens
    stats.new_tokens += new_tokens
    stats.blocks.append(
        BlockStat(
            message_index=msg_index,
            block_path=path,
            content_type=content_type,
            confidence=confidence,
            compressor=compressor,
            orig_tokens=orig_tokens,
            new_tokens=new_tokens,
            changed=changed,
            skipped=skipped,
        )
    )
    return out_text


def _passthrough_totals(messages: Any, config: Config, stats: CompressionStats) -> list[Any]:
    """Best-effort passthrough copy + token totals for the never-raise path.

    Never raises: used from the outer barrier where correctness matters more
    than fidelity. Resets any partial block stats accumulated before the fault.
    """
    try:
        out = copy.deepcopy(messages)
    except Exception:
        out = messages
    stats.blocks = []
    total = 0
    try:
        for msg in out if isinstance(out, list) else []:
            if isinstance(msg, dict):
                for text in _iter_text(msg.get("content")):
                    total += _safe_count(text, config.model)
    except Exception:
        total = 0
    stats.orig_tokens = total
    stats.new_tokens = total
    return out if isinstance(out, list) else list(messages) if isinstance(messages, list) else out


def _user_query_text(content: Any) -> str:
    """Flatten a user message's plain text (string or ``type=="text"`` blocks).

    Deliberately ignores ``tool_result`` blocks: in the Anthropic shape a tool
    result rides in a ``role="user"`` message, and its payload is data, not the
    human's question — so it must never become the derived query.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        return " ".join(parts).strip()
    return ""


def _last_user_message(messages: Any) -> tuple[int, str] | None:
    """Return ``(index, text)`` of the most recent user turn carrying text.

    Scans from the end for a ``role == "user"`` message with plain text and
    returns its array index alongside that text. ``None`` when no such message
    exists, so an array without a user turn costs a single reverse pass.
    """
    if not isinstance(messages, list):
        return None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _user_query_text(msg.get("content"))
            if text:
                return i, text
    return None


def _derive_query(messages: Any, max_chars: int = _QUERY_MAX_CHARS) -> str | None:
    """Relevance query from the most recent user message (issue #124), or None.

    The text is truncated to ``max_chars`` — a few hundred chars is plenty for
    BM25/anchor matching while bounding the cost on huge user turns.
    """
    found = _last_user_message(messages)
    return found[1][:max_chars] if found is not None else None


def _resolve_query(config: Config, messages: Any) -> tuple[Config, int | None]:
    """Resolve the ``Config.query`` sentinel; also report the source index (#124).

    Returns ``(config, source_index)`` where ``source_index`` is the array index
    of the user message the query was derived from (so the caller can avoid
    letting that turn filter itself by its own words); ``None`` when nothing was
    auto-derived. Semantics of ``Config.query``:

    * ``"auto"`` (the default) → derive from the last user message; the sentinel
      is replaced by the derived string, or by ``None`` when no user message
      carries text — so downstream compressors only ever see a real query or
      nothing, never the literal ``"auto"``.
    * ``None`` → left untouched: derivation is off (exact pre-#124 behavior).
    * any other string → left untouched: the caller forced this query.
    """
    if config.query != AUTO_QUERY:
        return config, None
    found = _last_user_message(messages)
    if found is None:
        return replace(config, query=None), None
    index, text = found
    return replace(config, query=text[:_QUERY_MAX_CHARS]), index


def compress(
    messages: list[Message],
    options: Config | None = None,
    **overrides: Any,
) -> tuple[list[Message], CompressionStats]:
    """Compress large text blocks in a message array.

    Args:
        messages: An OpenAI- or Anthropic-style message array.
        options: A resolved :class:`Config`. If omitted, config is loaded from
            environment variables and ``overrides``.
        **overrides: Per-call config overrides (e.g. ``min_bytes=0``).

    Returns:
        ``(rewritten_messages, stats)``. The input is never mutated. Honours a
        never-raise contract: on any unforeseen error the input is returned
        intact and ``stats.error`` is annotated.
    """
    config = options.merged(**overrides) if options is not None else load_config(**overrides)
    stats = CompressionStats()

    try:
        return _compress_impl(messages, config, stats)
    except Exception as exc:
        # Perimeter barrier (#116): never let compression crash the caller.
        logger.warning(
            "tokenslim.compress fell back to passthrough after %s: %s",
            type(exc).__name__,
            exc,
        )
        stats.note_error(exc)
        return _passthrough_totals(messages, config, stats), stats


def _compress_impl(
    messages: list[Message],
    config: Config,
    stats: CompressionStats,
) -> tuple[list[Message], CompressionStats]:
    """Core compression walk (wrapped by the never-raise barrier in compress)."""
    if not config.enabled:
        # No-op passthrough, but still report token totals for observability.
        out = _copy_messages(messages)
        for msg in out:
            if not isinstance(msg, dict):
                continue
            for text in _iter_text(msg.get("content")):
                stats.orig_tokens += _safe_count(text, config.model)
        stats.new_tokens = stats.orig_tokens
        return out, stats

    # Resolve the query-aware selection query before routing so JSON-match /
    # BM25 / log compressors see the real user question (issue #124).
    config, source_index = _resolve_query(config, messages)

    router = ContentRouter(config=config)
    stats.store = router.store
    # The user turn the query was auto-derived from must not filter itself by
    # its own words, so it is compressed by a query-less router (sharing the
    # same CCR store). Forced/absent queries never set source_index.
    plain_router = router
    if source_index is not None and config.query is not None:
        plain_router = ContentRouter(config=replace(config, query=None), store=router.store)

    out = _copy_messages(messages)
    for i, msg in enumerate(out):
        # Lax shape validation (#116): non-dict entries pass through untouched.
        if isinstance(msg, dict) and "content" in msg:
            active = plain_router if i == source_index else router
            msg["content"] = _rewrite_content(msg["content"], active, config, i, stats)

    # Dispatch anonymous usage telemetry
    from .telemetry import send_telemetry_event

    content_types = [str(block.content_type) for block in stats.blocks if block.changed]
    send_telemetry_event(
        orig_tokens=stats.orig_tokens,
        new_tokens=stats.new_tokens,
        model=config.model,
        content_types=content_types,
        enabled=config.telemetry,
    )

    # Opt-in local session capture (see tokenslim.capture, issue #41).
    if config.capture:
        from .capture import get_capture

        capture = get_capture(config)
        if capture is not None:
            payload: dict[str, Any] = {
                "orig_tokens": stats.orig_tokens,
                "new_tokens": stats.new_tokens,
                "ratio": stats.ratio,
                "content_types": [block.content_type.value for block in stats.blocks],
            }
            if config.capture_content:
                # Raw content is privacy-sensitive: only with the explicit knob.
                payload["messages"] = messages
            capture.record("compress", payload)

    return out, stats


def _iter_text(content: Any):
    """Yield every text string inside a content value."""
    if isinstance(content, str):
        if content:
            yield content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                yield block.get("text", "")
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, str) and inner:
                    yield inner
