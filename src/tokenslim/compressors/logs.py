"""LogCompressor — build/test output compaction.

Build and test logs are mostly noise (per-test PASS lines, progress chatter)
punctuated by a few high-value lines: failures, errors, warnings, and the final
summary. This compressor detects the log flavour (pytest / npm / cargo / jest /
make / generic), classifies each line by severity, scores it, and keeps the
important lines plus a small context window around each — dropping the rest
behind a single CCR marker.

Dedup is *conservative*: consecutive identical lines collapse with a counter,
but lines that differ only by an embedded id/address/number are NOT merged, so
distinguishing detail survives.

Stack-trace capture across blank-line-separated chained exceptions (#27) is
deferred to a follow-up; this handles the P0 build/test-output case (#26).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ..ccr import text_marker
from ..config import Config
from ..detector import ContentType

if TYPE_CHECKING:
    from ..store import CCRStore

__all__ = ["LogCompressor", "LogFormat", "Level", "classify_line"]


class LogFormat(str, Enum):
    PYTEST = "pytest"
    NPM = "npm"
    CARGO = "cargo"
    JEST = "jest"
    MAKE = "make"
    GENERIC = "generic"


class Level(int, Enum):
    DEBUG = 0
    INFO = 1
    SUMMARY = 2
    WARN = 3
    ERROR = 4


# --- format detection -----------------------------------------------------

_FORMAT_SIGNALS: list[tuple[LogFormat, re.Pattern[str]]] = [
    (LogFormat.PYTEST, re.compile(r"={3,}.*test session starts|^PASSED|^FAILED|::", re.M)),
    (LogFormat.CARGO, re.compile(r"^\s*Compiling \w|^error\[E\d+\]|^\s*Running .*target", re.M)),
    (LogFormat.JEST, re.compile(r"^\s*(?:PASS|FAIL)\s+\S+\.(?:test|spec)\.[jt]sx?", re.M)),
    (LogFormat.NPM, re.compile(r"^npm (?:ERR!|WARN)|node_modules", re.M)),
    (LogFormat.MAKE, re.compile(r"^make(?:\[\d+\])?:|^g?cc \S", re.M)),
]


def detect_log_format(text: str) -> LogFormat:
    for fmt, pattern in _FORMAT_SIGNALS:
        if pattern.search(text):
            return fmt
    return LogFormat.GENERIC


# --- line classification --------------------------------------------------

_ERROR_RE = re.compile(
    r"\b(?:error|fatal|critical|panic|exception|traceback|fail(?:ed|ure)?|"
    r"assert(?:ion)?error|err!)\b|^E\s|^\s*error\[E\d+\]",
    re.IGNORECASE,
)
_WARN_RE = re.compile(r"\b(?:warn(?:ing)?|deprecat)\b|^npm WARN", re.IGNORECASE)
_DEBUG_RE = re.compile(r"\b(?:debug|trace|verbose)\b", re.IGNORECASE)
# Summary lines: "5 passed, 2 failed in 1.2s", "Tests: 1 failed, 4 passed",
# "test result: FAILED", "BUILD SUCCESSFUL".
_SUMMARY_RE = re.compile(
    r"\b\d+\s+(?:passed|failed|error|skipped|warning)|"
    r"^Tests:|^test result:|BUILD (?:SUCCESS|SUCCESSFUL|FAILED)|"
    r"^={3,}.*(?:passed|failed|summary)|^Summary",
    re.IGNORECASE,
)


def classify_line(line: str) -> Level:
    if _SUMMARY_RE.search(line):
        return Level.SUMMARY
    if _ERROR_RE.search(line):
        return Level.ERROR
    if _WARN_RE.search(line):
        return Level.WARN
    if _DEBUG_RE.search(line):
        return Level.DEBUG
    return Level.INFO


# Tokens that distinguish otherwise-identical lines; if two adjacent lines
# differ only outside these, we still keep them separate (conservative dedup).
_DISTINGUISHING_RE = re.compile(
    r"0x[0-9a-fA-F]+|"  # hex addresses
    r"\b[0-9a-fA-F]{8,}\b|"  # long hex ids
    r"\b\d{3,}\b|"  # multi-digit numbers / ids
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"  # uuids
)


@dataclass
class _Line:
    index: int
    text: str
    level: Level


def detect_tracebacks(raw_lines: list[str]) -> set[int]:
    keep_indices: set[int] = set()
    n = len(raw_lines)

    # Python traceback patterns
    py_tb_header = re.compile(r"^\s*Traceback \(most recent call last\):", re.I)
    py_file_line = re.compile(r"^\s*File\s+\"[^\"]+\",\s+line\s+\d+", re.I)
    py_chain_msg = re.compile(
        r"^\s*(?:During handling of the above exception|"
        r"The above exception was the direct cause of the following exception):",
        re.I,
    )

    # JS at-frame pattern: starts with whitespace and "at "
    js_at_line = re.compile(r"^\s+at\s+\S+")

    i = 0
    while i < n:
        line = raw_lines[i]

        # 1. Detect Python Traceback
        if py_tb_header.match(line):
            tb_block = [i]
            i += 1
            while i < n:
                curr_line = raw_lines[i]
                if not curr_line.strip():
                    j = i + 1
                    while j < n and not raw_lines[j].strip():
                        j += 1
                    is_chain = (py_tb_header.match(raw_lines[j]) if j < n else False) or (
                        py_chain_msg.match(raw_lines[j]) if j < n else False
                    )
                    if j < n and is_chain:
                        tb_block.extend(range(i, j))
                        i = j
                        tb_block.append(i)
                    else:
                        break
                elif (
                    curr_line.startswith(" ")
                    or py_file_line.match(curr_line)
                    or py_chain_msg.match(curr_line)
                    or py_tb_header.match(curr_line)
                ):
                    tb_block.append(i)
                else:
                    tb_block.append(i)
                    # Look ahead to see if there is a chained exception coming up!
                    j = i + 1
                    while j < n and not raw_lines[j].strip():
                        j += 1
                    is_chain = (py_tb_header.match(raw_lines[j]) if j < n else False) or (
                        py_chain_msg.match(raw_lines[j]) if j < n else False
                    )
                    if j < n and is_chain:
                        tb_block.extend(range(i + 1, j))
                        i = j
                        tb_block.append(i)
                    else:
                        i += 1
                        break
                i += 1

            keep_indices.update(tb_block)
            continue

        # 2. Detect JS stack trace
        if js_at_line.match(line):
            if i > 0 and (i - 1) not in keep_indices:
                prev_line = raw_lines[i - 1].strip()
                if prev_line and not prev_line.startswith("at "):
                    keep_indices.add(i - 1)

            keep_indices.add(i)
            i += 1
            while i < n and (js_at_line.match(raw_lines[i]) or not raw_lines[i].strip()):
                if not raw_lines[i].strip():
                    j = i + 1
                    while j < n and not raw_lines[j].strip():
                        j += 1
                    if j < n and js_at_line.match(raw_lines[j]):
                        keep_indices.update(range(i, j))
                        i = j
                        keep_indices.add(i)
                    else:
                        break
                else:
                    keep_indices.add(i)
                i += 1
            continue

        i += 1

    return keep_indices


class LogCompressor:
    """Keeps high-signal log lines with context; drops the rest."""

    name = "log-compressor"

    def __init__(self, config: Config | None = None, store: CCRStore | None = None) -> None:
        self.config = config or Config()
        self.store = store

    def __call__(self, text: str, content_type: ContentType = ContentType.LOG) -> str:
        raw_lines = text.split("\n")
        if len(raw_lines) < 8:
            return text

        lines = [_Line(i, t, classify_line(t)) for i, t in enumerate(raw_lines)]
        keep = self._select(lines)

        if len(keep) >= len(lines):
            return text

        return self._render(lines, keep)

    def _select(self, lines: list[_Line]) -> set[int]:
        """Indices to keep: important lines + a context window around each."""
        ctx = self.config.log_context
        important = {ln.index for ln in lines if ln.level >= Level.SUMMARY}

        raw_texts = [ln.text for ln in lines]
        traceback_indices = detect_tracebacks(raw_texts)

        keep: set[int] = set(traceback_indices)
        all_important = important | traceback_indices
        for idx in all_important:
            lo = max(0, idx - ctx)
            hi = min(len(lines) - 1, idx + ctx)
            keep.update(range(lo, hi + 1))

        # Always keep the first and last line for orientation.
        keep.add(0)
        keep.add(len(lines) - 1)
        return keep

    def _render(self, lines: list[_Line], keep: set[int]) -> str:
        # Pass 1: emit kept lines, replacing each maximal run of dropped lines
        # with a single CCR marker.
        pieces: list[str] = []
        dropped: list[str] = []
        for ln in lines:
            if ln.index in keep:
                if dropped:
                    pieces.append(text_marker(dropped, store=self.store))
                    dropped = []
                pieces.append(ln.text)
            else:
                dropped.append(ln.text)
        if dropped:
            pieces.append(text_marker(dropped, store=self.store))

        # Pass 2: conservative run-length dedup of consecutive identical lines
        # that carry no distinguishing id/address (so "ok" repeats collapse but
        # "request 1041" / "request 1042" stay distinct).
        out: list[str] = []
        last_raw: str | None = None
        run = 1
        for piece in pieces:
            is_tb = (
                piece.startswith("Traceback (most recent call last):")
                or piece.strip().startswith("at ")
                or piece.strip().startswith("File ")
            )
            if piece == last_raw and not is_tb and not _DISTINGUISHING_RE.search(piece):
                run += 1
                out[-1] = f"{piece}  (x{run})"
            else:
                out.append(piece)
                last_raw = piece
                run = 1
        return "\n".join(out)
