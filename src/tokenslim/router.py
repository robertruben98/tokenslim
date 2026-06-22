"""Content router + compressor registry.

A *compressor* is any callable ``(text, content_type) -> str``. They register
against one or more :class:`ContentType` values. The :class:`ContentRouter`
detects a block's type, picks the matching compressor, skips tiny payloads, and
returns a :class:`RouteResult` describing what happened.

For M0 the only real compressor is JSON whitespace minification; everything
else falls through to an identity passthrough. Real algorithms land in M1
behind this same registry.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from .config import Config
from .detector import ContentType, detect_content_type

__all__ = [
    "Compressor",
    "RouteResult",
    "ContentRouter",
    "minify_json",
    "passthrough",
    "default_registry",
    "build_registry",
]

Compressor = Callable[[str, ContentType], str]


@dataclass(frozen=True)
class RouteResult:
    """Outcome of routing a single block of text."""

    text: str
    content_type: ContentType
    confidence: float
    compressor: str
    changed: bool
    skipped: bool


def minify_json(text: str, content_type: ContentType) -> str:
    """Strip insignificant whitespace from a JSON document.

    Falls back to the original text if parsing fails so the operation is always
    safe (never corrupts a payload that merely *looked* like JSON).
    """
    try:
        return json.dumps(json.loads(text), separators=(",", ":"), ensure_ascii=False)
    except (ValueError, TypeError):
        return text


def passthrough(text: str, content_type: ContentType) -> str:
    """Identity compressor — returns text unchanged."""
    return text


def default_registry() -> dict[ContentType, tuple[str, Compressor]]:
    """Minimal registry: JSON whitespace-minify, everything else passthrough.

    This is the M0 baseline. :func:`build_registry` layers the M1 algorithmic
    compressors on top and is what :class:`ContentRouter` uses by default.
    """
    return {
        ContentType.JSON: ("json-minify", minify_json),
        ContentType.CODE: ("passthrough", passthrough),
        ContentType.LOG: ("passthrough", passthrough),
        ContentType.DIFF: ("passthrough", passthrough),
        ContentType.SEARCH: ("passthrough", passthrough),
        ContentType.MARKDOWN: ("passthrough", passthrough),
        ContentType.TEXT: ("passthrough", passthrough),
    }


def build_registry(config: Config) -> dict[ContentType, tuple[str, Compressor]]:
    """Build the default M1 registry, wiring in the real compressors.

    Imported lazily to keep :mod:`tokenslim.router` free of a hard dependency
    on the compressors package (avoids an import cycle and keeps the M0 surface
    importable on its own).
    """
    from .compressors import LogCompressor, SearchCompressor, SmartCrusher

    registry = default_registry()
    crusher = SmartCrusher(config)
    registry[ContentType.JSON] = (SmartCrusher.name, crusher)
    registry[ContentType.LOG] = (LogCompressor.name, LogCompressor(config))
    registry[ContentType.SEARCH] = (SearchCompressor.name, SearchCompressor(config))
    return registry


class ContentRouter:
    """Routes text blocks to registered compressors."""

    def __init__(
        self,
        config: Config | None = None,
        registry: dict[ContentType, tuple[str, Compressor]] | None = None,
    ) -> None:
        self.config = config or Config()
        self.registry = registry if registry is not None else build_registry(self.config)

    def register(self, content_type: ContentType, name: str, compressor: Compressor) -> None:
        """Register (or replace) the compressor for ``content_type``."""
        self.registry[content_type] = (name, compressor)

    def route(self, text: str) -> RouteResult:
        """Detect, then compress ``text`` according to config + registry."""
        detection = detect_content_type(text)
        ctype = detection.content_type

        # Skip payloads below the byte threshold — not worth the overhead.
        if len(text.encode("utf-8")) < self.config.min_bytes:
            return RouteResult(text, ctype, detection.confidence, "skip", False, True)

        entry = self.registry.get(ctype)
        if entry is None:
            return RouteResult(text, ctype, detection.confidence, "none", False, False)

        name, compressor = entry
        if (
            self.config.enabled_compressors is not None
            and name not in self.config.enabled_compressors
        ):
            return RouteResult(text, ctype, detection.confidence, name, False, True)

        new_text = compressor(text, ctype)
        return RouteResult(
            new_text,
            ctype,
            detection.confidence,
            name,
            new_text != text,
            False,
        )
