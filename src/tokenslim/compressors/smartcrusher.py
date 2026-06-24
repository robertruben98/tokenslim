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
from ..relevance import tokenize

_STOP_WORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "arent",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "cant",
    "cannot",
    "could",
    "couldnt",
    "did",
    "didnt",
    "do",
    "does",
    "doesnt",
    "doing",
    "dont",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadnt",
    "has",
    "hasnt",
    "have",
    "havent",
    "having",
    "he",
    "hed",
    "hell",
    "hes",
    "her",
    "here",
    "heres",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "hows",
    "i",
    "id",
    "ill",
    "im",
    "ive",
    "if",
    "in",
    "into",
    "is",
    "isnt",
    "it",
    "its",
    "itself",
    "lets",
    "me",
    "more",
    "most",
    "mustnt",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shant",
    "she",
    "shed",
    "shell",
    "shes",
    "should",
    "shouldnt",
    "so",
    "some",
    "such",
    "than",
    "that",
    "thats",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "theres",
    "these",
    "they",
    "theyd",
    "theyll",
    "theyre",
    "theyve",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasnt",
    "we",
    "wed",
    "well",
    "were",
    "weve",
    "werent",
    "what",
    "whats",
    "when",
    "whens",
    "where",
    "wheres",
    "which",
    "while",
    "who",
    "whos",
    "whom",
    "why",
    "whys",
    "with",
    "wont",
    "would",
    "wouldnt",
    "you",
    "youd",
    "youll",
    "youre",
    "youve",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

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

    def _split_budget(self, k: int) -> tuple[int, int]:
        if k <= 0:
            return 0, 0
        if k == 1:
            return 1, 0
        head_ratio = self.config.crush_keep_head
        tail_ratio = self.config.crush_keep_tail
        total = head_ratio + tail_ratio
        if total == 0:
            return 0, 0
        head = round(k * head_ratio / total)
        head = max(0, min(k, head))
        tail = k - head
        return head, tail

    def _crush_array(self, items: list[Any]) -> Any:
        n = len(items)
        if self.config.max_items_after_crush is not None:
            k = min(self.config.max_items_after_crush, n)
            head, tail = self._split_budget(k)
            if n <= k:
                return self._maybe_recurse_children(items)
        else:
            head = self.config.crush_keep_head
            tail = self.config.crush_keep_tail
            if n < self.config.crush_min_items or n <= head + tail + 1:
                return self._maybe_recurse_children(items)

        kind = classify_array(items)
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
        keep: set[int] = set(range(head)) | set(range(max(0, n - tail), n))

        # Query-anchor relevance keep
        anchors = set()
        if self.config.query:
            query_terms = tokenize(self.config.query)
            anchors = {t for t in query_terms if t not in _STOP_WORDS}
            if not anchors:
                anchors = set(query_terms)

        for i, item in enumerate(items):
            if i in keep:
                continue
            if self._is_error_item(item):
                keep.add(i)
                continue
            if field_stats and self._has_rare_value(item, field_stats):
                keep.add(i)
                continue
            if anchors:
                blob = (
                    item
                    if isinstance(item, str)
                    else json.dumps(item, ensure_ascii=False, default=str)
                ).lower()
                item_tokens = set(tokenize(blob))
                if anchors.intersection(item_tokens):
                    keep.add(i)

        # Anomaly detection (Z-score outliers + variance change points)
        # 1. Raw numbers array
        is_num_array = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in items)
        if not field_stats and is_num_array:
            keep.update(self._detect_numeric_anomalies(items))

        # 2. Numeric fields in object arrays
        if field_stats:
            for name, fs in field_stats.items():
                if fs.is_id_like:
                    continue
                row_vals = []
                for row_idx, item in enumerate(items):
                    if isinstance(item, dict) and name in item:
                        val = item[name]
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            row_vals.append((row_idx, float(val)))
                if len(row_vals) >= 2:
                    keep.update(self._detect_numeric_anomalies_from_indexed(row_vals))

        return keep

    def _detect_numeric_anomalies(self, nums: list[float | int]) -> set[int]:
        indexed = [(i, float(x)) for i, x in enumerate(nums)]
        return self._detect_numeric_anomalies_from_indexed(indexed)

    def _detect_numeric_anomalies_from_indexed(self, indexed: list[tuple[int, float]]) -> set[int]:
        keep: set[int] = set()
        n = len(indexed)
        if n < 2:
            return keep

        vals = [val for _, val in indexed]
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / n
        std = math.sqrt(var)

        # 1. Z-score outliers (>2 sigma)
        if std > 0:
            for idx, val in indexed:
                if abs(val - mean) / std > 2.0:
                    keep.add(idx)

        # 2. Variance change-point detection (CSS algorithm)
        if n >= 4 and var > 0:
            y = [val - mean for val in vals]
            C = []
            curr = 0.0
            for val in y:
                curr += val * val
                C.append(curr)

            total_sum_sq = C[-1]
            if total_sum_sq > 0:
                max_d = -1.0
                k_max = -1
                for k in range(n - 1):
                    d_k = abs(C[k] / total_sum_sq - (k + 1) / n)
                    if d_k > max_d:
                        max_d = d_k
                        k_max = k

                # Check if change-point is significant
                if max_d > 0.15 and k_max != -1:
                    keep.add(indexed[k_max][0])
                    keep.add(indexed[k_max + 1][0])

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
