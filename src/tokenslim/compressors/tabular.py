"""TabularCompressor — CSV/TSV table compaction.

Wide tables are mostly repetitive rows; the signal is the header, a few
representative rows, the numeric outliers, and per-column statistics. This
compressor sniffs the delimiter (``,`` ``;`` tab ``|``), keeps the header plus
the first/last rows and any outlier rows (numeric columns: |z| > 2.5 or the
min/max holders), appends a ``# stats`` summary line per numeric column
(count/min/max/mean), and drops the remaining rows behind a single CCR marker.
Kept rows are emitted verbatim from the original lines — never re-serialised —
so the model sees exactly the bytes it would have seen uncompressed.

The marker stores the ORIGINAL full table (not just the dropped rows) because
the compressed view reorders rows and synthesises stats lines — only the full
original allows a faithful reconstruction on retrieval.
"""

from __future__ import annotations

import csv
import math
import statistics
from typing import TYPE_CHECKING

from ..ccr import TEXT_PREFIX, content_hash, make_marker
from ..config import Config
from ..detector import ContentType

if TYPE_CHECKING:
    from ..store import CCRStore

__all__ = ["TabularCompressor"]

_DELIMITERS = (",", ";", "\t", "|")
_REASON = "rows-elided"
_Z_THRESHOLD = 2.5


def _sniff_delimiter(lines: list[str]) -> str | None:
    """Pick the delimiter yielding a constant field count >= 2 across lines.

    When several candidates qualify, the one producing the most fields wins
    (e.g. ``a;b;c`` lines also "parse" under ``,`` as single-field rows).
    """
    best: tuple[int, str] | None = None
    for delim in _DELIMITERS:
        try:
            rows = list(csv.reader(lines, delimiter=delim))
        except csv.Error:
            continue
        if len(rows) != len(lines):
            continue
        counts = {len(row) for row in rows}
        if len(counts) != 1:
            continue
        n_fields = counts.pop()
        if n_fields < 2:
            continue
        if best is None or n_fields > best[0]:
            best = (n_fields, delim)
    return best[1] if best is not None else None


def _to_float(value: str) -> float | None:
    """Parse ``value`` as a finite float, or ``None``."""
    stripped = value.strip()
    if not stripped:
        return None
    try:
        number = float(stripped)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _fmt_num(value: float) -> str:
    """Compact number rendering: integral values as ints, else 4 sig. digits."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return format(value, ".4g")


class TabularCompressor:
    """Keeps header, head/tail and outlier rows of a table; drops the rest."""

    name = "tabular"

    def __init__(self, config: Config | None = None, store: CCRStore | None = None) -> None:
        self.config = config or Config()
        self.store = store

    def __call__(self, text: str, content_type: ContentType = ContentType.CSV) -> str:
        try:
            return self._compress(text)
        except Exception:  # compressors must never raise — fall back to original
            return text

    # --- internals ---------------------------------------------------------

    def _compress(self, text: str) -> str:
        lines = text.splitlines()
        if len(lines) < 3:
            return text
        delimiter = _sniff_delimiter(lines)
        if delimiter is None:
            return text

        # _sniff_delimiter parsed this same ``lines`` list and required one
        # line == one record, so ``rows`` maps 1:1 onto ``lines`` and kept rows
        # can be emitted verbatim below (no re-serialisation, no quote mangling).
        rows = list(csv.reader(lines, delimiter=delimiter))
        if len(rows) != len(lines):
            return text
        width = len(rows[0]) if rows else 0
        # Ragged tables are not compacted — column stats would be meaningless.
        if width < 2 or any(len(row) != width for row in rows):
            return text

        header, data = rows[0], rows[1:]
        keep_head = max(0, self.config.csv_keep_head)
        keep_tail = max(0, self.config.csv_keep_tail)
        if len(data) <= keep_head + keep_tail:
            return text

        head_idx = set(range(keep_head))
        tail_idx = set(range(len(data) - keep_tail, len(data)))
        numeric_cols = self._numeric_columns(data, width)
        outlier_idx = self._outliers(data, numeric_cols, exclude=head_idx | tail_idx)

        kept = head_idx | tail_idx | outlier_idx
        elided = len(data) - len(kept)
        if elided <= 0:
            return text

        pieces = [lines[0]]
        pieces += [lines[1 + i] for i in sorted(head_idx)]
        pieces.append(self._marker_line(text, elided))
        pieces += [lines[1 + i] for i in sorted(outlier_idx)]
        pieces += [lines[1 + i] for i in sorted(tail_idx)]
        for col, values in sorted(numeric_cols.items()):
            name = header[col].strip() or f"col{col + 1}"
            numbers = [v for v in values if v is not None]
            pieces.append(
                f"# stats {name}: count={len(numbers)} min={_fmt_num(min(numbers))} "
                f"max={_fmt_num(max(numbers))} mean={_fmt_num(statistics.fmean(numbers))}"
            )

        result = "\n".join(pieces)
        return result if len(result) < len(text) else text

    def _numeric_columns(self, data: list[list[str]], width: int) -> dict[int, list[float | None]]:
        """Columns where every non-empty cell parses as a number (>= 2 values)."""
        numeric: dict[int, list[float | None]] = {}
        for col in range(width):
            values: list[float | None] = []
            ok = True
            for row in data:
                if not row[col].strip():
                    values.append(None)
                    continue
                number = _to_float(row[col])
                if number is None:
                    ok = False
                    break
                values.append(number)
            if ok and sum(v is not None for v in values) >= 2:
                numeric[col] = values
        return numeric

    def _outliers(
        self,
        data: list[list[str]],
        numeric_cols: dict[int, list[float | None]],
        exclude: set[int],
    ) -> set[int]:
        """Middle rows worth keeping: |z| > 2.5 or min/max holders per column."""
        scored: dict[int, float] = {}
        for values in numeric_cols.values():
            numbers = [v for v in values if v is not None]
            if len(numbers) < 3:
                continue
            mean = statistics.fmean(numbers)
            stdev = statistics.pstdev(numbers)
            if stdev == 0:
                continue  # constant column: no outliers, min/max meaningless
            lo, hi = min(numbers), max(numbers)
            for idx, value in enumerate(values):
                if value is None or idx in exclude:
                    continue
                z = abs(value - mean) / stdev
                if z > _Z_THRESHOLD or value in (lo, hi):
                    scored[idx] = max(scored.get(idx, 0.0), z)
        budget = max(0, self.config.csv_max_outliers)
        ranked = sorted(scored, key=lambda i: scored[i], reverse=True)
        return set(ranked[:budget])

    def _marker_line(self, original: str, elided: int) -> str:
        """One-line CCR sentinel (text_marker format) storing the full table."""
        key = self.store.put(original) if self.store is not None else content_hash(original)
        return f"{TEXT_PREFIX} {elided} {_REASON} {make_marker(key, elided, _REASON)}"
