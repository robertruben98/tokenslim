"""Public ``compress()`` entry point and result schema.

Walks an OpenAI/Anthropic-style message array, routes each large text block
through the :class:`ContentRouter`, rewrites the messages in place (on a copy),
and returns the new array alongside :class:`CompressionStats`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .config import Config, load_config
from .detector import ContentType
from .router import ContentRouter
from .tokenizer import count_tokens

if TYPE_CHECKING:
    from .store import CCRStore

__all__ = ["compress", "CompressionStats", "BlockStat"]

Message = dict[str, Any]


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

    @property
    def ratio(self) -> float:
        """Fraction of tokens removed (0.0 = no change, 1.0 = everything)."""
        if self.orig_tokens == 0:
            return 0.0
        return 1.0 - (self.new_tokens / self.orig_tokens)

    @property
    def saved_tokens(self) -> int:
        return self.orig_tokens - self.new_tokens


def _rewrite_content(
    content: Any,
    router: ContentRouter,
    model: str | None,
    msg_index: int,
    stats: CompressionStats,
) -> Any:
    """Compress the text in a message's ``content`` (string or block list)."""
    if isinstance(content, str):
        return _rewrite_text(content, router, model, msg_index, "content", stats)

    if isinstance(content, list):
        new_blocks = []
        for i, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                block = dict(block)
                block["text"] = _rewrite_text(
                    block["text"], router, model, msg_index, f"content[{i}].text", stats
                )
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                block = dict(block)
                inner = block.get("content", "")
                if isinstance(inner, str):
                    block["content"] = _rewrite_text(
                        inner, router, model, msg_index, f"content[{i}].tool_result", stats
                    )
            new_blocks.append(block)
        return new_blocks

    return content


def _rewrite_text(
    text: str,
    router: ContentRouter,
    model: str | None,
    msg_index: int,
    path: str,
    stats: CompressionStats,
) -> str:
    """Route a single text string and record its stats."""
    if not isinstance(text, str) or not text:
        return text

    orig_tokens = count_tokens(text, model)
    result = router.route(text)
    new_tokens = count_tokens(result.text, model) if result.changed else orig_tokens

    stats.orig_tokens += orig_tokens
    stats.new_tokens += new_tokens
    stats.blocks.append(
        BlockStat(
            message_index=msg_index,
            block_path=path,
            content_type=result.content_type,
            confidence=result.confidence,
            compressor=result.compressor,
            orig_tokens=orig_tokens,
            new_tokens=new_tokens,
            changed=result.changed,
            skipped=result.skipped,
        )
    )
    return result.text


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
        ``(rewritten_messages, stats)``. The input is never mutated.
    """
    config = options.merged(**overrides) if options is not None else load_config(**overrides)
    stats = CompressionStats()

    if not config.enabled:
        # No-op passthrough, but still report token totals for observability.
        out = copy.deepcopy(messages)
        for msg in out:
            content = msg.get("content")
            for text in _iter_text(content):
                stats.orig_tokens += count_tokens(text, config.model)
        stats.new_tokens = stats.orig_tokens
        return out, stats

    router = ContentRouter(config=config)
    stats.store = router.store
    out = copy.deepcopy(messages)
    for i, msg in enumerate(out):
        if "content" in msg:
            msg["content"] = _rewrite_content(msg["content"], router, config.model, i, stats)

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
