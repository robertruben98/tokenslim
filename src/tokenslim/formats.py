"""Message-format detection and conversion.

The pipeline is format-agnostic: it understands both OpenAI chat-completion
messages and Anthropic Messages-API messages. These helpers detect which shape
an array uses and convert between them, plus walk message content to extract
and rewrite the text blocks the router operates on.

OpenAI shape: ``{"role": "...", "content": "str" | [parts]}`` plus a top-level
``tool`` role for tool results.
Anthropic shape: ``{"role": "user"|"assistant", "content": "str" | [blocks]}``
where tool results are ``{"type": "tool_result", ...}`` blocks under a ``user``
message.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

__all__ = [
    "MessageFormat",
    "detect_format",
    "openai_to_anthropic",
    "anthropic_to_openai",
]

Message = dict[str, Any]


class MessageFormat(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    UNKNOWN = "unknown"


def detect_format(messages: list[Message]) -> MessageFormat:
    """Best-effort detection of the message-array dialect."""
    if not messages:
        return MessageFormat.UNKNOWN

    for msg in messages:
        role = msg.get("role")
        # The `tool` and `system` roles only exist in OpenAI's chat format.
        if role in ("tool", "system", "function"):
            return MessageFormat.OPENAI
        if "tool_call_id" in msg or "tool_calls" in msg:
            return MessageFormat.OPENAI
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype in ("tool_use", "tool_result"):
                    return MessageFormat.ANTHROPIC
                if btype in ("image_url", "input_audio"):
                    return MessageFormat.OPENAI

    # Plain user/assistant string-content messages are valid in both; default
    # to OpenAI as the more common interchange shape.
    return MessageFormat.OPENAI


def _content_to_text_blocks(content: Any) -> list[str]:
    """Extract text strings from OpenAI/Anthropic content (string or parts)."""
    if isinstance(content, str):
        return [content]
    texts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                texts.append(part["text"])
    return texts


def openai_to_anthropic(messages: list[Message]) -> list[Message]:
    """Convert an OpenAI chat array to Anthropic Messages format.

    ``system`` messages are dropped from the array (Anthropic carries the
    system prompt out of band); ``tool`` results become ``tool_result`` blocks
    attached to a ``user`` message.
    """
    out: list[Message] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        if role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": msg.get("content", ""),
                        }
                    ],
                }
            )
            continue
        out.append({"role": role, "content": msg.get("content", "")})
    return out


def anthropic_to_openai(messages: list[Message]) -> list[Message]:
    """Convert an Anthropic Messages array to OpenAI chat format.

    ``tool_result`` blocks are lifted into standalone ``tool`` messages.
    """
    out: list[Message] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):
            tool_results = [
                b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            text_blocks = _content_to_text_blocks(content)
            if text_blocks:
                out.append({"role": role, "content": "\n".join(text_blocks)})
            for block in tool_results:
                inner = block.get("content", "")
                if isinstance(inner, list):
                    inner = "\n".join(_content_to_text_blocks(inner))
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": inner,
                    }
                )
        else:
            out.append({"role": role, "content": content})
    return out
