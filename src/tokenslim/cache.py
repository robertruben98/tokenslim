"""Cache optimization — dynamic content normalization and cache boundary injection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .tokenizer import count_tokens

__all__ = [
    "normalize_dynamic_content",
    "insert_anthropic_cache_control",
    "stabilize_message_order",
    "optimize_for_prefix_cache",
    "find_volatile_spans",
    "PrefixCacheReport",
    "VolatileSpan",
    "UUID_RE",
    "ISO_DATE_RE",
    "EPOCH_TIME_RE",
    "HEX_HASH_RE",
    "API_KEY_RE",
]

UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
ISO_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
)
EPOCH_TIME_RE = re.compile(r"\b1\d{9}(?:\.\d+)?\b")
HEX_HASH_RE = re.compile(r"\b[0-9a-fA-F]{32,64}\b")
API_KEY_RE = re.compile(r"\b(?:gh[osr]_[0-9a-zA-Z]{36,255}|sk-[0-9a-zA-Z]{48})\b")


def normalize_dynamic_content(text: str) -> str:
    """Normalize dynamic content like UUIDs, timestamps, and keys to keep prompt prefixes stable.

    Replaces them with generic placeholders to prevent prompt cache invalidation.
    """
    text = UUID_RE.sub("<UUID>", text)
    text = API_KEY_RE.sub("<TOKEN>", text)
    text = ISO_DATE_RE.sub("<TIMESTAMP>", text)
    text = EPOCH_TIME_RE.sub("<TIMESTAMP>", text)
    text = HEX_HASH_RE.sub("<HASH>", text)
    return text


def insert_anthropic_cache_control(
    messages: list[dict[str, Any]],
    system: list[dict[str, Any]] | str | None = None,
    min_bytes: int = 2048,
    max_breakpoints: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | str | None]:
    """Injects cache_control ephemeral breakpoints into messages and/or system prompt.

    Only targets stable blocks that exceed `min_bytes` and keeps within `max_breakpoints`.
    """
    # Create copies of messages to avoid mutating input
    new_messages = [dict(m) for m in messages]
    new_system = system

    breakpoints_inserted = 0

    # 1. Check system prompt (high priority for caching)
    if new_system:
        if isinstance(new_system, str):
            if len(new_system) >= min_bytes and breakpoints_inserted < max_breakpoints:
                # Convert system string to a block list to support cache_control
                new_system = [
                    {
                        "type": "text",
                        "text": new_system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                breakpoints_inserted += 1
        elif isinstance(new_system, list) and new_system:
            # Add cache_control to the last block of the system prompt
            last_block = dict(new_system[-1])
            if (
                last_block.get("type") == "text"
                and len(last_block.get("text", "")) >= min_bytes
                and breakpoints_inserted < max_breakpoints
            ):
                last_block["cache_control"] = {"type": "ephemeral"}
                new_system = list(new_system)
                new_system[-1] = last_block
                breakpoints_inserted += 1

    # 2. Iterate backward through messages to cache recent large contexts
    for i in range(len(new_messages) - 1, -1, -1):
        if breakpoints_inserted >= max_breakpoints:
            break

        msg = new_messages[i]
        content = msg.get("content")
        if not content:
            continue

        if isinstance(content, str):
            if len(content) >= min_bytes:
                # Convert content string to block list
                msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                breakpoints_inserted += 1
        elif isinstance(content, list) and content:
            last_idx = len(content) - 1
            last_block = dict(content[last_idx])
            if last_block.get("type") == "text" and len(last_block.get("text", "")) >= min_bytes:
                last_block["cache_control"] = {"type": "ephemeral"}
                new_content = list(content)
                new_content[last_idx] = last_block
                msg["content"] = new_content
                breakpoints_inserted += 1

    return new_messages, new_system


# --- Prefix-cache awareness (OpenAI / Google implicit caches, Anthropic breakpoints) ---

# Roles that are provably order-independent context rather than conversation turns.
_STABLE_ROLES = ("system", "developer")

# Minimum stable-prefix size (tokens) for the provider's prompt cache to engage.
# See optimize_for_prefix_cache docstring for the provider documentation references.
_MIN_PREFIX_TOKENS: dict[str, int] = {
    "openai": 1024,
    "google": 1024,
    "anthropic": 1024,
}

# Same patterns and precedence order as normalize_dynamic_content.
_VOLATILE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("uuid", UUID_RE),
    ("token", API_KEY_RE),
    ("timestamp", ISO_DATE_RE),
    ("timestamp", EPOCH_TIME_RE),
    ("hash", HEX_HASH_RE),
)


@dataclass(frozen=True)
class VolatileSpan:
    """A cache-busting substring found by :func:`find_volatile_spans`.

    ``kind`` mirrors the placeholder normalize_dynamic_content would substitute:
    ``uuid``, ``token``, ``timestamp`` or ``hash``.
    """

    kind: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class PrefixCacheReport:
    """Result of :func:`optimize_for_prefix_cache`.

    ``hints`` are actionable suggestions (e.g. "move volatile timestamp out of
    system prompt") for making more of the prompt prefix cacheable.
    """

    provider: str
    stable_prefix_tokens: int
    cacheable: bool
    hints: tuple[str, ...] = ()


def find_volatile_spans(text: str) -> tuple[VolatileSpan, ...]:
    """Report volatile (cache-busting) substrings in ``text`` without rewriting it.

    Reuses the exact regexes behind :func:`normalize_dynamic_content` (UUIDs, API
    keys, ISO/epoch timestamps, long hex hashes). Overlapping matches are resolved
    with the same precedence normalize_dynamic_content applies its substitutions
    in; results are ordered by position in the text.
    """
    spans: list[VolatileSpan] = []
    claimed: list[tuple[int, int]] = []
    for kind, pattern in _VOLATILE_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < c_end and c_start < end for c_start, c_end in claimed):
                continue
            claimed.append((start, end))
            spans.append(VolatileSpan(kind=kind, start=start, end=end, text=match.group(0)))
    spans.sort(key=lambda span: span.start)
    return tuple(spans)


def stabilize_message_order(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hoist system-level messages to the front so the prompt prefix stays stable.

    Only the provably order-independent move is made: messages whose role is
    ``system`` or ``developer`` (instructions, tool definitions and other stable
    context) are hoisted, keeping their relative order, ahead of the conversation.
    The relative order of every other message — user/assistant turns and
    ``tool`` results paired with their preceding ``tool_calls`` — is preserved
    exactly, so the conversation flow is never rewritten.

    Non-mutating: returns a new list referencing the original message dicts.
    """
    stable = [m for m in messages if m.get("role") in _STABLE_ROLES]
    volatile = [m for m in messages if m.get("role") not in _STABLE_ROLES]
    return stable + volatile


def _message_text(message: dict[str, Any]) -> str:
    """Concatenate the text content of a message (string or text-block list)."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``message`` with normalized text content (non-mutating)."""
    content = message.get("content")
    if isinstance(content, str):
        normalized = normalize_dynamic_content(content)
        if normalized == content:
            return message
        new_message = dict(message)
        new_message["content"] = normalized
        return new_message
    if isinstance(content, list):
        changed = False
        new_content: list[Any] = []
        for block in content:
            if isinstance(block, str):
                normalized = normalize_dynamic_content(block)
                if normalized != block:
                    block = normalized
                    changed = True
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                normalized = normalize_dynamic_content(block["text"])
                if normalized != block["text"]:
                    block = {**block, "text": normalized}
                    changed = True
            new_content.append(block)
        if not changed:
            return message
        new_message = dict(message)
        new_message["content"] = new_content
        return new_message
    return message


def optimize_for_prefix_cache(
    messages: list[dict[str, Any]],
    provider: str = "openai",
    model: str | None = None,
) -> tuple[list[dict[str, Any]], PrefixCacheReport]:
    """Shape ``messages`` for the provider's prompt/prefix cache and report cacheability.

    OpenAI and Google Gemini run *implicit* prefix caches: requests whose leading
    tokens are byte-identical to a recent request get the cached prefix for free,
    so the win comes from message shaping — stable content first, volatile content
    last, no per-request churn inside the prefix. Anthropic uses explicit
    breakpoints, so that branch additionally delegates to
    :func:`insert_anthropic_cache_control` after the same shaping.

    Steps applied (all non-mutating):

    1. :func:`stabilize_message_order` — hoist system/developer messages to the front.
    2. :func:`normalize_dynamic_content` on the designated-stable segment (the
       system prompt) ONLY — conversation turns are never rewritten.
    3. Count the stable prefix with :func:`tokenslim.tokenizer.count_tokens`
       (using ``model`` when given) and compare against the provider minimum.

    Provider thresholds (``cacheable`` is True when the stable prefix meets them):

    - OpenAI: prompt caching activates automatically for prompts of 1024+ tokens
      (https://platform.openai.com/docs/guides/prompt-caching).
    - Google: Gemini implicit caching needs a minimum prompt size of 1024 tokens
      on 2.5 Flash (2048 on 2.5 Pro); 1024 is used here as the conservative signal
      (https://ai.google.dev/gemini-api/docs/caching).
    - Anthropic: minimum cacheable prefix is 1024 tokens on most models (2048 on
      Haiku); breakpoints are inserted via :func:`insert_anthropic_cache_control`
      (https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).

    Raises ``ValueError`` for an unknown ``provider``.
    """
    provider = provider.lower()
    if provider not in _MIN_PREFIX_TOKENS:
        raise ValueError(
            f"unknown provider {provider!r}; expected one of {sorted(_MIN_PREFIX_TOKENS)}"
        )

    ordered = stabilize_message_order(messages)
    hoisted = [id(m) for m in ordered] != [id(m) for m in messages]

    stable_count = 0
    for message in ordered:
        if message.get("role") in _STABLE_ROLES:
            stable_count += 1
        else:
            break

    hints: list[str] = []
    if hoisted:
        hints.append(
            "hoisted system message(s) ahead of conversation turns to keep the prefix stable"
        )

    # Report volatility on the original stable segment, then normalize it.
    volatile_counts: dict[str, int] = {}
    for message in ordered[:stable_count]:
        for span in find_volatile_spans(_message_text(message)):
            volatile_counts[span.kind] = volatile_counts.get(span.kind, 0) + 1
    for kind in sorted(volatile_counts):
        hints.append(
            f"move volatile {kind} out of system prompt "
            f"({volatile_counts[kind]} span(s) normalized to keep the prefix byte-stable)"
        )

    optimized = list(ordered)
    for i in range(stable_count):
        optimized[i] = _normalize_message(optimized[i])

    prefix_text = "\n".join(_message_text(m) for m in optimized[:stable_count])
    stable_prefix_tokens = count_tokens(prefix_text, model) if prefix_text else 0
    threshold = _MIN_PREFIX_TOKENS[provider]
    cacheable = stable_prefix_tokens >= threshold

    if stable_count == 0:
        hints.append(
            "no system message found; put stable instructions and tool definitions "
            "in a system message to enable prefix caching"
        )
    if not cacheable:
        hints.append(
            f"stable prefix is {stable_prefix_tokens} tokens, below the ~{threshold}-token "
            f"minimum for {provider} prompt caching; move more stable context "
            "(tool definitions, few-shot examples) into the system prompt"
        )

    if provider == "anthropic":
        optimized, _ = insert_anthropic_cache_control(optimized)

    report = PrefixCacheReport(
        provider=provider,
        stable_prefix_tokens=stable_prefix_tokens,
        cacheable=cacheable,
        hints=tuple(hints),
    )
    return optimized, report
