"""JsonMinifier — lossless JSON whitespace strip.

Parse → re-serialise compact. Always safe: byte-lossless on the *value* (the
re-serialised JSON parses back to an equal object), and if minifying doesn't
actually shrink the text it returns the original untouched. A cheap 2-5% win,
usable as a pre-pass or for non-array JSON that SmartCrusher wouldn't crush.
"""

from __future__ import annotations

import json

from ..config import Config
from ..detector import ContentType

__all__ = ["JsonMinifier", "minify"]


def minify(text: str) -> str:
    """Compact-serialise ``text`` if it is JSON and the result is shorter.

    Returns the original string on parse failure or when minifying wouldn't
    help, so the operation never corrupts or inflates a payload.
    """
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return text
    compact = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return compact if len(compact) < len(text) else text


class JsonMinifier:
    """Configurable lossless JSON minifier compressor."""

    name = "json-minify"

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def __call__(self, text: str, content_type: ContentType = ContentType.JSON) -> str:
        return minify(text)
