"""CCR — Compressed-Content-Record markers & sentinels.

When a compressor drops material it leaves behind a small, machine-readable
marker recording *what* and *how much* was removed plus a stable content hash of
the dropped original. With a :class:`~tokenslim.store.CCRStore` wired in, the
original is also cached under that hash so it can be retrieved on demand — the
marker is the model's pointer back to the full data.

The canonical marker is a compact text token::

    <<ccr:HASH N reason>>

* ``HASH`` — content hash of the dropped material (the store key).
* ``N`` — how many items/lines were dropped.
* ``reason`` — short tag (e.g. ``middle-elided``, ``lines-elided``).

For JSON arrays the marker is wrapped in a sentinel object so it survives
re-serialisation::

    {"_ccr_dropped": "<<ccr:HASH N reason>>", "__tokenslim_ccr__": {...detail...}}

CCR is intentionally tiny — a marker costs a handful of tokens against the
hundreds-to-thousands it represents.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .store import CCRStore

__all__ = [
    "content_hash",
    "make_marker",
    "parse_marker",
    "find_markers",
    "strip_markers",
    "CCRMarker",
    "json_sentinel",
    "text_marker",
    "SENTINEL_KEY",
    "DROPPED_KEY",
    "TEXT_PREFIX",
]

# Stable keys/prefixes so markers are greppable and unambiguous in any payload.
SENTINEL_KEY = "__tokenslim_ccr__"
DROPPED_KEY = "_ccr_dropped"
TEXT_PREFIX = "[tokenslim:ccr]"

# Canonical marker: <<ccr:HASH N reason>>. Hash is hex; N is an int; reason is an
# optional run of non-space token chars.
_MARKER_RE = re.compile(r"<<ccr:(?P<hash>[0-9a-f]+)\s+(?P<count>\d+)(?:\s+(?P<reason>[^\s>]+))?>>")


@dataclass(frozen=True)
class CCRMarker:
    """A parsed CCR marker."""

    hash: str
    count: int
    reason: str


def content_hash(payload: Any) -> str:
    """Short, stable hash of ``payload`` (serialised deterministically)."""
    if isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        data = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def make_marker(hash: str, count: int, reason: str = "elided") -> str:
    """Build the canonical ``<<ccr:HASH N reason>>`` marker token."""
    return f"<<ccr:{hash} {count} {reason}>>"


def parse_marker(text: str) -> CCRMarker | None:
    """Parse the first CCR marker in ``text``; ``None`` if there is none."""
    m = _MARKER_RE.search(text)
    if m is None:
        return None
    return CCRMarker(
        hash=m.group("hash"),
        count=int(m.group("count")),
        reason=m.group("reason") or "elided",
    )


def find_markers(text: str) -> list[CCRMarker]:
    """Return every CCR marker found in ``text``."""
    return [
        CCRMarker(
            hash=m.group("hash"), count=int(m.group("count")), reason=m.group("reason") or "elided"
        )
        for m in _MARKER_RE.finditer(text)
    ]


def strip_markers(text: str) -> str:
    """Remove all CCR marker tokens from ``text`` (collapsing leftover spaces)."""
    cleaned = _MARKER_RE.sub("", text)
    # Tidy the double spaces / dangling spaces a removed inline marker leaves.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return re.sub(r"[ \t]+(\n|$)", r"\1", cleaned)


def _store_and_hash(payload: Any, store: CCRStore | None, fallback: str) -> str:
    """Persist ``payload`` to ``store`` (if given) and return the key.

    Falls back to a plain content hash when no store is provided, so markers are
    still meaningful (auditable) even without a backing cache.
    """
    if store is None:
        return fallback
    serialised = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False, default=str)
    )
    return store.put(serialised)


def json_sentinel(
    dropped_items: list[Any],
    *,
    total: int,
    kept: int,
    reason: str = "middle-elided",
    store: CCRStore | None = None,
) -> dict[str, Any]:
    """Build a JSON sentinel object describing dropped array items.

    When ``store`` is given, the dropped items are persisted and the sentinel's
    hash is the store key, so the rows are retrievable via :func:`retrieve`.
    """
    fallback = content_hash(dropped_items)
    key = _store_and_hash(dropped_items, store, fallback)
    n = len(dropped_items)
    return {
        DROPPED_KEY: make_marker(key, n, reason),
        SENTINEL_KEY: {
            "dropped": n,
            "kept": kept,
            "total": total,
            "reason": reason,
            "hash": key,
        },
    }


def text_marker(
    dropped_lines: list[str],
    *,
    reason: str = "lines-elided",
    store: CCRStore | None = None,
) -> str:
    """Build a one-line text sentinel describing dropped lines.

    The line is the human-readable prefix followed by the canonical marker so it
    is both readable and machine-parseable / retrievable.
    """
    payload = "\n".join(dropped_lines)
    fallback = content_hash(dropped_lines)
    # Hash the joined text when storing so retrieve() returns the exact lines.
    key = store.put(payload) if store is not None else fallback
    n = len(dropped_lines)
    return f"{TEXT_PREFIX} {n} {reason} {make_marker(key, n, reason)}"
