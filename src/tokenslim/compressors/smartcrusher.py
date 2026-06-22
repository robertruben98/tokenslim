"""SmartCrusher — statistical JSON-array compressor.

The single biggest win in agent transcripts: tool outputs that are long
homogeneous arrays (DB rows, list endpoints, search hits). SmartCrusher keeps
the first/last N items, drops the statistically-redundant middle, and appends a
CCR sentinel recording how many rows were elided and their content hash.

Two safety rails make the elision lossless *where it matters*:

* **Error preservation** — any item whose serialised form contains an error
  keyword is always kept, wherever it sits in the array.
* **Rare-value preservation** — for low-cardinality "status"-like fields, every
  item carrying a rare value is kept so anomalies (one ``cancelled`` among a
  thousand ``ok``) survive. There is no early short-circuit when a field has
  many distinct codes.

Sub-issues covered: array classifier + per-field analysis (#21), error &
rare-value preservation (#22), core crusher (#20).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..ccr import json_sentinel
from ..config import Config
from ..detector import ContentType

if TYPE_CHECKING:
    from ..store import CCRStore

__all__ = ["SmartCrusher", "ArrayKind", "FieldStats", "classify_array", "analyze_fields"]


class ArrayKind(str, Enum):
    OBJECTS = "objects"
    NUMBERS = "numbers"
    STRINGS = "strings"
    MIXED = "mixed"
    EMPTY = "empty"


@dataclass(frozen=True)
class FieldStats:
    """Per-field statistics over an array of objects."""

    name: str
    count: int
    cardinality: int
    entropy: float
    is_id_like: bool
    is_score_like: bool
    is_status_like: bool
    rare_values: frozenset[Any]


def classify_array(items: list[Any]) -> ArrayKind:
    """Classify the element type of a JSON array."""
    if not items:
        return ArrayKind.EMPTY
    kinds = set()
    for item in items:
        if isinstance(item, dict):
            kinds.add("object")
        elif isinstance(item, bool):
            kinds.add("string")  # treat bools like low-card categoricals
        elif isinstance(item, (int, float)):
            kinds.add("number")
        elif isinstance(item, str):
            kinds.add("string")
        else:
            kinds.add("other")
    if kinds == {"object"}:
        return ArrayKind.OBJECTS
    if kinds == {"number"}:
        return ArrayKind.NUMBERS
    if kinds == {"string"}:
        return ArrayKind.STRINGS
    return ArrayKind.MIXED


def _entropy(values: list[Any]) -> float:
    """Shannon entropy (bits) of a value list."""
    n = len(values)
    if n == 0:
        return 0.0
    counts = Counter(json.dumps(v, sort_keys=True, default=str) for v in values)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# Field-name hints used alongside structural stats.
_ID_HINT = ("id", "uuid", "key", "pk", "_id", "guid")
_SCORE_HINT = ("score", "rank", "weight", "confidence", "prob", "rating", "distance")
_STATUS_HINT = ("status", "state", "level", "type", "code", "result", "severity", "kind")


def _name_matches(name: str, hints: tuple[str, ...]) -> bool:
    low = name.lower()
    return any(low == h or low.endswith("_" + h) or low.endswith(h) for h in hints)


def analyze_fields(
    items: list[dict[str, Any]], rare_threshold: float = 0.05
) -> dict[str, FieldStats]:
    """Compute per-field stats for an array of objects.

    A *status-like* field is a low-cardinality categorical (few distinct values
    relative to row count); its rare values (frequency below ``rare_threshold``)
    are flagged so the crusher can preserve them.
    """
    stats: dict[str, FieldStats] = {}
    # Union of keys; only fields present on a majority of rows are "columns".
    key_counts: Counter[str] = Counter()
    for item in items:
        key_counts.update(item.keys())

    for name, present in key_counts.items():
        values = [item[name] for item in items if name in item]
        # Only hashable/categorical-friendly values participate in cardinality.
        try:
            distinct = {json.dumps(v, sort_keys=True, default=str) for v in values}
        except TypeError:
            distinct = set()
        cardinality = len(distinct)
        ent = _entropy(values)

        is_id = _name_matches(name, _ID_HINT) or (cardinality == present and present > 1)
        is_score = _name_matches(name, _SCORE_HINT) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
        )
        # Status-like: explicitly hinted, OR genuinely low cardinality and not
        # an id. No early short-circuit on "too many distinct" — we still scan
        # for rare values even when there are 50+ codes.
        low_card = present > 0 and cardinality <= max(2, present * 0.2)
        is_status = (_name_matches(name, _STATUS_HINT) or low_card) and not is_id

        rare: set[Any] = set()
        if is_status:
            freq = Counter(json.dumps(v, sort_keys=True, default=str) for v in values)
            cutoff = max(1, math.floor(present * rare_threshold))
            raw_by_key = {json.dumps(v, sort_keys=True, default=str): v for v in values}
            for key, c in freq.items():
                raw = raw_by_key[key]
                # Only scalar (hashable) status values participate in rare-value
                # preservation; nested structures aren't categoricals.
                if c <= cutoff and isinstance(raw, (str, int, float, bool, type(None))):
                    rare.add(raw)

        stats[name] = FieldStats(
            name=name,
            count=present,
            cardinality=cardinality,
            entropy=ent,
            is_id_like=is_id,
            is_score_like=is_score,
            is_status_like=is_status,
            rare_values=frozenset(rare),
        )
    return stats


class SmartCrusher:
    """Configurable JSON-array compressor."""

    name = "smartcrusher"

    def __init__(self, config: Config | None = None, store: CCRStore | None = None) -> None:
        self.config = config or Config()
        self.store = store

    # -- public callable -------------------------------------------------

    def __call__(self, text: str, content_type: ContentType = ContentType.JSON) -> str:
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return text

        crushed = self._crush_value(data)
        if crushed is _UNCHANGED:
            # Nothing crushed; still strip whitespace for the cheap win.
            return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        return json.dumps(crushed, separators=(",", ":"), ensure_ascii=False)

    # -- recursion -------------------------------------------------------

    def _crush_value(self, value: Any) -> Any:
        """Return a crushed copy, or the ``_UNCHANGED`` sentinel if no change."""
        if isinstance(value, list):
            return self._crush_array(value)
        if isinstance(value, dict):
            changed = False
            out = {}
            for k, v in value.items():
                cv = self._crush_value(v)
                if cv is _UNCHANGED:
                    out[k] = v
                else:
                    out[k] = cv
                    changed = True
            return out if changed else _UNCHANGED
        return _UNCHANGED

    def _crush_array(self, items: list[Any]) -> Any:
        head = self.config.crush_keep_head
        tail = self.config.crush_keep_tail
        n = len(items)

        # Too short to bother, or keeping head+tail wouldn't drop anything.
        if n < self.config.crush_min_items or n <= head + tail + 1:
            # Still recurse into nested arrays/objects.
            return self._maybe_recurse_children(items)

        kind = classify_array(items)
        # Only homogeneous arrays get statistically crushed. Mixed/empty arrays
        # just recurse into their children.
        if kind in (ArrayKind.MIXED, ArrayKind.EMPTY):
            return self._maybe_recurse_children(items)

        field_stats = analyze_fields(items) if kind is ArrayKind.OBJECTS else {}

        keep_indices = self._select_keep_indices(items, field_stats, head, tail)
        if len(keep_indices) >= n:
            return self._maybe_recurse_children(items)

        dropped = [items[i] for i in range(n) if i not in keep_indices]
        result: list[Any] = []
        sentinel_emitted = False
        for i in range(n):
            if i in keep_indices:
                result.append(items[i])
            elif not sentinel_emitted:
                # Emit a single sentinel at the first gap (the elided middle).
                if self.config.ccr:
                    result.append(
                        json_sentinel(
                            dropped,
                            total=n,
                            kept=len(keep_indices),
                            store=self.store,
                        )
                    )
                sentinel_emitted = True
        return result

    def _maybe_recurse_children(self, items: list[Any]) -> Any:
        changed = False
        out = []
        for item in items:
            cv = self._crush_value(item)
            if cv is _UNCHANGED:
                out.append(item)
            else:
                out.append(cv)
                changed = True
        return out if changed else _UNCHANGED

    # -- selection -------------------------------------------------------

    def _select_keep_indices(
        self,
        items: list[Any],
        field_stats: dict[str, FieldStats],
        head: int,
        tail: int,
    ) -> set[int]:
        n = len(items)
        keep: set[int] = set(range(head)) | set(range(n - tail, n))

        for i, item in enumerate(items):
            if i in keep:
                continue
            if self._is_error_item(item):
                keep.add(i)
                continue
            if field_stats and self._has_rare_value(item, field_stats):
                keep.add(i)
        return keep

    def _is_error_item(self, item: Any) -> bool:
        """True if the item's serialised form contains an error keyword."""
        if not self.config.error_keywords:
            return False
        blob = (
            item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, default=str)
        ).lower()
        return any(kw in blob for kw in self.config.error_keywords)

    def _has_rare_value(self, item: Any, field_stats: dict[str, FieldStats]) -> bool:
        if not isinstance(item, dict):
            return False
        for name, fs in field_stats.items():
            if not fs.is_status_like or not fs.rare_values:
                continue
            if name in item and item[name] in fs.rare_values:
                return True
        return False


# Distinct sentinel so ``None`` results aren't mistaken for "no change".
class _Unchanged:
    __slots__ = ()


_UNCHANGED = _Unchanged()
