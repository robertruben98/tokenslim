"""Cache optimization — dynamic content normalization and cache boundary injection."""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "normalize_dynamic_content",
    "insert_anthropic_cache_control",
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
