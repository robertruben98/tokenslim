"""Rule-based content-type detection.

Given a chunk of text, classify it as one of :class:`ContentType` with a
confidence score. The implementation is deliberately heuristic (regex + cheap
structural checks) so it has no dependencies; the public surface is shaped so
an ML detector (e.g. Magika) can drop in later behind the same function.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from enum import Enum

__all__ = ["ContentType", "DetectionResult", "detect_content_type"]


class ContentType(str, Enum):
    JSON = "json"
    HTML = "html"
    CODE = "code"
    LOG = "log"
    DIFF = "diff"
    SEARCH = "search"
    CSV = "csv"
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
# HTML: a doctype/<html> prologue, or several distinct well-known tags up front.
_HTML_DOC_RE = re.compile(r"<!doctype\s+html\b|<html[\s>]", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)\b")
_HTML_TAGS = frozenset(
    {
        "html",
        "head",
        "body",
        "title",
        "meta",
        "link",
        "script",
        "style",
        "div",
        "span",
        "p",
        "a",
        "ul",
        "ol",
        "li",
        "table",
        "thead",
        "tbody",
        "tr",
        "td",
        "th",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "br",
        "hr",
        "img",
        "nav",
        "header",
        "footer",
        "section",
        "article",
        "aside",
        "main",
        "form",
        "input",
        "button",
        "strong",
        "em",
        "b",
        "i",
        "pre",
        "code",
        "blockquote",
    }
)


# Delimiters recognised for tabular (CSV-like) payloads.
_CSV_DELIMITERS = (",", ";", "\t", "|")


def _csv_field_count(lines: list[str], delimiter: str) -> int:
    """Constant field count of ``lines`` under ``delimiter``; 0 if inconsistent.

    Conservative on purpose: every sampled line must parse to exactly the same
    number of fields (>= 2), with no multi-line records, so code/prose/log
    payloads that merely contain the delimiter are not misread as tables.
    """
    try:
        rows = list(csv.reader(lines, delimiter=delimiter))
    except csv.Error:
        return 0
    if len(rows) != len(lines):
        return 0  # blank lines or quoted multi-line records — not a plain table
    counts = {len(row) for row in rows}
    if len(counts) != 1:
        return 0
    n_fields = counts.pop()
    if n_fields < 2:
        return 0
    # Reject "delimiter-terminated" lines (e.g. code ending in ';', markdown
    # pipe-table edges): a constant empty first/last field is not real data.
    if all(row[0].strip() == "" for row in rows) or all(row[-1].strip() == "" for row in rows):
        return 0
    return n_fields


def _looks_like_csv(lines: list[str]) -> bool:
    """True when >= 3 lines share one delimiter with a constant field count."""
    if len(lines) < 3:
        return False
    # JSONL guard: uniform-schema JSON lines also show a constant comma count,
    # but they are JSON payloads, not tables — never steal them.
    if lines[0].lstrip().startswith(("{", "[")):
        return False
    sample = lines if len(lines) <= 40 else lines[:20] + lines[-20:]
    return any(_csv_field_count(sample, d) >= 2 for d in _CSV_DELIMITERS)


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

    # HTML is likewise structurally checkable: a doctype/<html> prologue, or a
    # markup-shaped start with several distinct well-known tags near the top.
    stripped = text.lstrip()
    if stripped.startswith("<"):
        if _HTML_DOC_RE.match(stripped):
            return DetectionResult(ContentType.HTML, 0.97)
        head_tags = {m.group(1).lower() for m in _HTML_TAG_RE.finditer(stripped[:2048])}
        if len(head_tags & _HTML_TAGS) >= 3:
            return DetectionResult(ContentType.HTML, 0.85)

    if _DIFF_RE.search(text):
        return DetectionResult(ContentType.DIFF, 0.95)

    lines = text.splitlines() or [text]
    n_lines = len(lines)

    log_hits = len(_LOG_LINE_RE.findall(text))
    if n_lines >= 2 and log_hits / n_lines >= 0.4:
        return DetectionResult(ContentType.LOG, min(0.95, 0.6 + log_hits / n_lines))

    # Tabular data must be checked BEFORE the search branch: CSV rows with
    # date-like leading fields ("12-05", "10:30") also match _SEARCH_HIT_RE.
    if _looks_like_csv(lines):
        return DetectionResult(ContentType.CSV, min(0.9, 0.6 + n_lines / 50))

    search_hits = len(_SEARCH_HIT_RE.findall(text))
    if n_lines >= 3 and search_hits / n_lines >= 0.5:
        return DetectionResult(ContentType.SEARCH, min(0.9, 0.55 + search_hits / n_lines))

    keyword_hits = len(_CODE_KEYWORD_RE.findall(text))
    symbol_hits = len(_CODE_SYMBOL_RE.findall(text))
    code_score = keyword_hits + 0.5 * symbol_hits
    md_hits = len(_MARKDOWN_RE.findall(text))

    # Prose shouldn't be misread as code. A single physical line can carry a few
    # English words that happen to be code keywords ("please import ... from ...
    # and return it"), so keyword density alone is not enough: real code either
    # spans multiple lines or carries structural punctuation ({} ; => -> :: == …).
    # Require a reasonable signal density AND one of those structural cues.
    code_density = code_score / max(1, n_lines)
    has_structure = n_lines >= 2 or symbol_hits >= 2
    if code_score >= 3 and code_density >= 0.8 and code_score >= md_hits and has_structure:
        return DetectionResult(ContentType.CODE, min(0.9, 0.5 + code_density / 4))

    if md_hits >= 2:
        return DetectionResult(ContentType.MARKDOWN, min(0.85, 0.5 + md_hits / n_lines))

    return DetectionResult(ContentType.TEXT, 0.6)
