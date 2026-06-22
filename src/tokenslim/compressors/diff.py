"""DiffCompressor — unified-diff compaction.

Large diffs (think a 4,000-line PR dump) are mostly context. This compressor:

1. parses the unified diff into per-file blocks of hunks;
2. caps the number of files, keeping the most-changed first (by added+removed
   line count);
3. per kept file, keeps the first/last and highest-churn hunks (capped);
4. trims each kept hunk's leading/trailing context lines;
5. records everything it drops (whole files + extra hunks) to the CCR store
   behind a single marker — but only commits the compaction if it actually
   shrinks the diff below ~0.8 of the original (otherwise returns the original).

Budgets come from the adaptive sizer / config, so the same knobs drive log,
search, and diff compaction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..ccr import TEXT_PREFIX, content_hash, make_marker
from ..config import Config
from ..detector import ContentType
from ..sizer import compute_optimal_k

if TYPE_CHECKING:
    from ..store import CCRStore

__all__ = ["DiffCompressor", "Hunk", "FileDiff", "parse_diff"]

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
# A new file block starts at "diff --git" or a bare "--- " header (when there
# is no git preamble). We treat "diff --git" as the canonical boundary.
_FILE_RE = re.compile(r"^diff --git ")


@dataclass
class Hunk:
    header: str
    lines: list[str] = field(default_factory=list)

    @property
    def churn(self) -> int:
        """Added + removed lines in this hunk (context excluded)."""
        return sum(1 for ln in self.lines if ln[:1] in ("+", "-"))


@dataclass
class FileDiff:
    header_lines: list[str] = field(default_factory=list)  # diff --git, ---, +++, index
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def churn(self) -> int:
        return sum(h.churn for h in self.hunks)


def parse_diff(text: str) -> list[FileDiff]:
    """Parse a unified diff into per-file blocks. Best-effort, never raises."""
    files: list[FileDiff] = []
    current: FileDiff | None = None
    current_hunk: Hunk | None = None

    for line in text.split("\n"):
        if _FILE_RE.match(line):
            current = FileDiff(header_lines=[line])
            current_hunk = None
            files.append(current)
            continue
        if current is None:
            # Tolerate diffs that start straight at "--- a/x" with no git line.
            current = FileDiff()
            files.append(current)
        if _HUNK_RE.match(line):
            current_hunk = Hunk(header=line)
            current.hunks.append(current_hunk)
            continue
        if current_hunk is not None:
            current_hunk.lines.append(line)
        else:
            current.header_lines.append(line)
    return files


class DiffCompressor:
    """Caps files & hunks in a unified diff, trims context, CCRs the rest."""

    name = "diff-compressor"

    def __init__(self, config: Config | None = None, store: CCRStore | None = None) -> None:
        self.config = config or Config()
        self.store = store

    def __call__(self, text: str, content_type: ContentType = ContentType.DIFF) -> str:
        files = parse_diff(text)
        if not files or all(not f.hunks for f in files):
            return text

        kept_files, dropped_files = self._select_files(files)
        out_lines: list[str] = []
        dropped_chunks: list[str] = []

        for fd in kept_files:
            out_lines.extend(fd.header_lines)
            kept_hunks, dropped_hunks = self._select_hunks(fd)
            for hunk in kept_hunks:
                out_lines.append(hunk.header)
                out_lines.extend(self._trim_context(hunk.lines))
            for hunk in dropped_hunks:
                dropped_chunks.append(self._render_hunk(hunk))

        for fd in dropped_files:
            dropped_chunks.append(self._render_file(fd))

        if dropped_chunks:
            out_lines.append(self._marker(dropped_chunks))

        result = "\n".join(out_lines)
        # Only commit the compaction if it's a real win (< ~0.8 of original).
        if len(result) >= 0.8 * len(text):
            return text
        return result

    # -- selection -------------------------------------------------------

    def _select_files(self, files: list[FileDiff]) -> tuple[list[FileDiff], list[FileDiff]]:
        cap = self.config.diff_max_files
        if len(files) <= cap:
            return files, []
        # Rank by churn (most-changed first), keep cap, preserve original order.
        ranked = sorted(range(len(files)), key=lambda i: files[i].churn, reverse=True)
        keep_idx = set(ranked[:cap])
        kept = [f for i, f in enumerate(files) if i in keep_idx]
        dropped = [f for i, f in enumerate(files) if i not in keep_idx]
        return kept, dropped

    def _select_hunks(self, fd: FileDiff) -> tuple[list[Hunk], list[Hunk]]:
        n = len(fd.hunks)
        # The configured per-file cap is the floor; for files with very many
        # hunks the adaptive sizer is allowed to raise it so we don't crush a
        # huge file down to the same few hunks as a small one.
        cap = max(
            self.config.diff_max_hunks_per_file, compute_optimal_k(n, self.config.target_ratio)
        )
        if n <= cap:
            return fd.hunks, []

        # Keep first + last for orientation, fill the rest with highest churn.
        keep_idx = {0, n - 1}
        by_churn = sorted(range(n), key=lambda i: fd.hunks[i].churn, reverse=True)
        for i in by_churn:
            if len(keep_idx) >= cap:
                break
            keep_idx.add(i)
        kept = [h for i, h in enumerate(fd.hunks) if i in keep_idx]
        dropped = [h for i, h in enumerate(fd.hunks) if i not in keep_idx]
        return kept, dropped

    # -- rendering -------------------------------------------------------

    def _trim_context(self, lines: list[str]) -> list[str]:
        """Trim leading/trailing context (space-prefixed) lines to diff_context."""
        ctx = self.config.diff_context
        start = 0
        while start < len(lines) and lines[start][:1] == " ":
            start += 1
        end = len(lines)
        while end > start and lines[end - 1][:1] == " ":
            end -= 1
        lead = lines[max(0, start - ctx) : start]
        trail = lines[end : end + ctx]
        return [*lead, *lines[start:end], *trail]

    def _render_hunk(self, hunk: Hunk) -> str:
        return "\n".join([hunk.header, *hunk.lines])

    def _render_file(self, fd: FileDiff) -> str:
        parts = list(fd.header_lines)
        for h in fd.hunks:
            parts.append(self._render_hunk(h))
        return "\n".join(parts)

    def _marker(self, dropped_chunks: list[str]) -> str:
        payload = "\n".join(dropped_chunks)
        key = self.store.put(payload) if self.store is not None else content_hash(payload)
        return (
            f"{TEXT_PREFIX} {len(dropped_chunks)} diff chunks elided "
            f"{make_marker(key, len(dropped_chunks), 'diff-elided')}"
        )
