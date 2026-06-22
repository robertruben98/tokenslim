"""SearchCompressor — grep / ripgrep output compaction.

grep/ripgrep dumps repeat the file path on every single hit. Grouping hits by
file kills that repetition, and capping the number of files keeps a 10k-line
search from blowing the budget. Hits are scored by a light relevance heuristic
(definition/assignment lines rank above bare references) so the most useful
matches survive the cap.

Handles ``file:line:content`` and ``file-line-content`` (ripgrep ``-C`` context
uses ``-`` as the separator), Windows drive paths (``C:\\src\\x.py:12:...``),
and hyphenated filenames.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..ccr import TEXT_PREFIX, content_hash, make_marker
from ..config import Config
from ..detector import ContentType

if TYPE_CHECKING:
    from ..store import CCRStore

__all__ = ["SearchCompressor", "SearchHit", "parse_search_line"]


@dataclass
class SearchHit:
    path: str
    lineno: int | None
    content: str
    is_match: bool  # True for ":" separator (a match), False for "-" (context)


@dataclass
class _FileGroup:
    path: str
    hits: list[SearchHit] = field(default_factory=list)
    score: float = 0.0


# grep/ripgrep hit lines look like ``path<sep>lineno<sep>content`` where sep is
# ``:`` for matches and ``-`` for ``-C`` context lines. The hard part is that
# the path itself may contain hyphens ("my-file.py") and a Windows drive colon
# ("C:\src\x.py"). We anchor on the ``<sep>\d+<sep>`` lineno bracket and let the
# path be greedy up to it, so hyphenated filenames stay intact.
_LINE_RE = re.compile(
    r"^(?P<path>(?:[A-Za-z]:[\\/])?.*?)"  # path (optionally drive-prefixed)
    r"(?P<sep>[:-])"
    r"(?P<lineno>\d+)"
    r"(?P=sep)"
    r"(?P<content>.*)$"
)

# Relevance signals — lines that define/assign rank higher than references.
_DEFINITION_RE = re.compile(
    r"\b(?:def|class|function|func|fn|interface|struct|impl|type|const|let|var)\b|"
    r"=\s|=>|:=|public|private|export"
)


def parse_search_line(line: str) -> SearchHit | None:
    """Parse one grep/ripgrep line; return ``None`` if it isn't a hit line."""
    m = _LINE_RE.match(line)
    if not m:
        return None
    path = m.group("path")
    # A bare "12:foo" with an empty path isn't a file hit.
    if not path:
        return None
    return SearchHit(
        path=path,
        lineno=int(m.group("lineno")),
        content=m.group("content"),
        is_match=m.group("sep") == ":",
    )


def _score_hit(hit: SearchHit) -> float:
    score = 1.0 if hit.is_match else 0.3  # context lines matter less
    if _DEFINITION_RE.search(hit.content):
        score += 1.0
    # Very long lines are usually minified noise; mild penalty.
    if len(hit.content) > 200:
        score -= 0.3
    return score


class SearchCompressor:
    """Groups search hits by file, scores them, caps the number of files."""

    name = "search-compressor"

    def __init__(self, config: Config | None = None, store: CCRStore | None = None) -> None:
        self.config = config or Config()
        self.store = store

    def __call__(self, text: str, content_type: ContentType = ContentType.SEARCH) -> str:
        lines = text.split("\n")
        groups: dict[str, _FileGroup] = {}
        unparsed: list[str] = []
        order: list[str] = []

        for line in lines:
            if not line.strip():
                continue
            hit = parse_search_line(line)
            if hit is None:
                unparsed.append(line)
                continue
            if hit.path not in groups:
                groups[hit.path] = _FileGroup(hit.path)
                order.append(hit.path)
            grp = groups[hit.path]
            grp.hits.append(hit)
            grp.score += _score_hit(hit)

        if not groups:
            return text  # nothing parseable — leave it alone

        max_files = self.config.search_max_files
        ranked = sorted(groups.values(), key=lambda g: g.score, reverse=True)
        kept = ranked[:max_files]
        dropped = ranked[max_files:]

        # Preserve original file order among the kept groups for readability.
        kept_paths = {g.path for g in kept}
        kept_in_order = [groups[p] for p in order if p in kept_paths]

        out: list[str] = []
        for grp in kept_in_order:
            out.append(f"{grp.path}:")
            for hit in grp.hits:
                marker = ":" if hit.is_match else "-"
                out.append(f"  {hit.lineno}{marker} {hit.content}")

        if dropped:
            n_hits = sum(len(g.hits) for g in dropped)
            # Persist the dropped hits verbatim so they can be retrieved.
            dropped_text = "\n".join(
                f"{g.path}:{h.lineno}{':' if h.is_match else '-'}{h.content}"
                for g in dropped
                for h in g.hits
            )
            key = (
                self.store.put(dropped_text)
                if self.store is not None
                else content_hash(dropped_text)
            )
            out.append(
                f"{TEXT_PREFIX} {len(dropped)} files / {n_hits} hits elided "
                f"(low relevance) {make_marker(key, n_hits, 'files-elided')}"
            )

        result = "\n".join(out)
        # Don't expand: if grouping somehow didn't help, keep the original.
        return result if len(result) < len(text) else text
