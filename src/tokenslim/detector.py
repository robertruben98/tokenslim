"""Rule-based content-type detection.

Given a chunk of text, classify it as one of :class:`ContentType` with a
confidence score. The implementation is deliberately heuristic (regex + cheap
structural checks) so it has no dependencies; the public surface is shaped so
an ML detector (e.g. Magika) can drop in later behind the same function.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

__all__ = ["ContentType", "DetectionResult", "detect_content_type"]


class ContentType(str, Enum):
    JSON = "json"
    CODE = "code"
    LOG = "log"
    DIFF = "diff"
    SEARCH = "search"
    MARKDOWN = "markdown"
    TEXT = "text"


@dataclass(frozen=True)
class DetectionResult:
    content_type: ContentType
    confidence: float


# --- signal regexes -------------------------------------------------------

# Lines like "2024-01-02 13:45:01 INFO ..." or "[ERROR] ...".
_LOG_LINE_RE = re.compile(
    r"^\s*(?:\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}|"
    r"\[?(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b)",
    re.IGNORECASE | re.MULTILINE,
)
_DIFF_RE = re.compile(r"^(?:diff --git |@@ -\d|index [0-9a-f]+\.\.|\+\+\+ |--- )", re.MULTILINE)
_MARKDOWN_RE = re.compile(r"^(?:#{1,6}\s|\s*[-*+]\s|\s*\d+\.\s|```|>\s)", re.MULTILINE)
_CODE_KEYWORD_RE = re.compile(
    r"\b(?:def|class|import|from|function|const|let|var|return|public|private|"
    r"package|func|fn|impl|module|namespace)\b"
)
_CODE_SYMBOL_RE = re.compile(r"[{};]|=>|->|::|==|!=|\+=")
_SEARCH_HIT_RE = re.compile(r"^\s*\d+[:\-]", re.MULTILINE)  # grep/ripgrep "line:..."
_URL_RE = re.compile(r"https?://\S+")


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        json.loads(stripped)
    except (ValueError, TypeError):
        return False
    return True


def detect_content_type(text: str) -> DetectionResult:
    """Classify ``text`` and return the type with a confidence in [0, 1]."""
    if not text or not text.strip():
        return DetectionResult(ContentType.TEXT, 1.0)

    # JSON is structurally verifiable, so trust it most.
    if _looks_like_json(text):
        return DetectionResult(ContentType.JSON, 0.99)

    if _DIFF_RE.search(text):
        return DetectionResult(ContentType.DIFF, 0.95)

    lines = text.splitlines() or [text]
    n_lines = len(lines)

    log_hits = len(_LOG_LINE_RE.findall(text))
    if n_lines >= 2 and log_hits / n_lines >= 0.4:
        return DetectionResult(ContentType.LOG, min(0.95, 0.6 + log_hits / n_lines))

    search_hits = len(_SEARCH_HIT_RE.findall(text))
    if n_lines >= 3 and search_hits / n_lines >= 0.5:
        return DetectionResult(ContentType.SEARCH, min(0.9, 0.55 + search_hits / n_lines))

    code_score = len(_CODE_KEYWORD_RE.findall(text)) + 0.5 * len(_CODE_SYMBOL_RE.findall(text))
    md_hits = len(_MARKDOWN_RE.findall(text))

    # Prose with URLs / sentences shouldn't be misread as code; require a
    # reasonable density of code signals relative to size.
    code_density = code_score / max(1, n_lines)
    if code_score >= 3 and code_density >= 0.8 and code_score >= md_hits:
        return DetectionResult(ContentType.CODE, min(0.9, 0.5 + code_density / 4))

    if md_hits >= 2:
        return DetectionResult(ContentType.MARKDOWN, min(0.85, 0.5 + md_hits / n_lines))

    return DetectionResult(ContentType.TEXT, 0.6)
