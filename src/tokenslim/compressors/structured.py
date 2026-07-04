"""Structured-format compressors: JSONL and Markdown tables.

Both delegate to an existing, tuned algorithm rather than re-implementing
crushing:

* :class:`JsonlCompressor` wraps its records into a JSON array and reuses
  :class:`~tokenslim.compressors.smartcrusher.SmartCrusher`, then re-emits the
  kept records (and the CCR sentinel) one per line.
* :class:`MarkdownTableCompressor` keeps the header + separator + head/tail data
  rows and drops the redundant middle behind a single CCR marker, reusing the
  ``csv_keep_head`` / ``csv_keep_tail`` knobs.

Neither ever raises: on any parse trouble they return the input unchanged, and
the per-block token guard in :mod:`tokenslim.compress` reverts them if the
output would not be smaller.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from ..ccr import find_markers, text_marker
from ..config import Config
from ..detector import ContentType
from .smartcrusher import SmartCrusher

if TYPE_CHECKING:
    from ..store import CCRStore

__all__ = ["JsonlCompressor", "MarkdownTableCompressor"]

_MD_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$")


class JsonlCompressor:
    """Compress newline-delimited JSON by reusing SmartCrusher on the array."""

    name = "jsonl"

    def __init__(self, config: Config | None = None, store: CCRStore | None = None) -> None:
        self.config = config or Config()
        self.store = store
        # Share the CCR store so dropped records stay retrievable.
        self._crusher = SmartCrusher(self.config, store)

    def __call__(self, text: str, content_type: ContentType = ContentType.JSONL) -> str:
        records: list[object] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except (ValueError, TypeError):
                return text  # not clean JSONL — never corrupt it
        if len(records) < 2:
            return text

        crushed_text = self._crusher(json.dumps(records, ensure_ascii=False), ContentType.JSON)
        try:
            crushed = json.loads(crushed_text)
        except (ValueError, TypeError):
            return text
        if not isinstance(crushed, list):
            return text

        out: list[str] = []
        for element in crushed:
            # SmartCrusher replaces the dropped middle with a single string CCR
            # sentinel; emit that verbatim and re-encode real records compactly
            # (same whitespace-free form SmartCrusher uses for the array).
            if isinstance(element, str) and find_markers(element):
                out.append(element)
            else:
                out.append(json.dumps(element, ensure_ascii=False, separators=(",", ":")))
        return "\n".join(out)


class MarkdownTableCompressor:
    """Crush a Markdown pipe table: keep header + head/tail rows, CCR the rest."""

    name = "markdown-table"

    def __init__(self, config: Config | None = None, store: CCRStore | None = None) -> None:
        self.config = config or Config()
        self.store = store

    def __call__(self, text: str, content_type: ContentType = ContentType.MD_TABLE) -> str:
        lines = text.splitlines()
        sep_idx = next(
            (i for i in range(1, len(lines)) if _MD_SEP_RE.match(lines[i]) and "|" in lines[i - 1]),
            None,
        )
        if sep_idx is None:
            return text

        # Contiguous pipe rows after the separator are the data body; anything
        # after (blank line, prose) is preserved untouched as a trailer.
        data: list[str] = []
        trailer_start = len(lines)
        for j in range(sep_idx + 1, len(lines)):
            if "|" in lines[j] and lines[j].strip():
                data.append(lines[j])
            else:
                trailer_start = j
                break
        trailer = lines[trailer_start:]

        head = max(0, self.config.csv_keep_head)
        tail = max(0, self.config.csv_keep_tail)
        if len(data) <= head + tail + 1:
            return text

        dropped = data[head : len(data) - tail] if tail else data[head:]
        kept = data[:head] + (data[len(data) - tail :] if tail else [])
        if not dropped:
            return text

        result = lines[: sep_idx + 1] + kept
        if self.config.ccr:
            # A blank line ends the table so the marker renders as a plain note.
            result.append("")
            result.append(text_marker(dropped, reason="rows-elided", store=self.store))
        result.extend(trailer)
        return "\n".join(result)
