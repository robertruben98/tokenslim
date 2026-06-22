"""CCR — Compressed-Content-Record sentinels.

When a compressor drops material it leaves behind a small, machine-readable
marker recording *what* and *how much* was removed, plus a stable content hash
of the dropped material. This keeps compression auditable: a downstream tool
(or a human) can see that 4,000 array rows were elided rather than silently
losing them, and the hash lets a cache/store reconcile the original later.

CCR is intentionally tiny — a sentinel costs a handful of tokens against the
hundreds-to-thousands it represents.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["content_hash", "json_sentinel", "text_marker"]

# Stable prefix so sentinels are greppable and unambiguous in any payload.
SENTINEL_KEY = "__tokenslim_ccr__"
TEXT_PREFIX = "[tokenslim:ccr]"


def content_hash(payload: Any) -> str:
    """Short, stable hash of ``payload`` (serialised deterministically)."""
    if isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        data = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def json_sentinel(
    dropped_items: list[Any],
    *,
    total: int,
    kept: int,
    reason: str = "middle-elided",
) -> dict[str, Any]:
    """Build a JSON sentinel object describing dropped array items."""
    return {
        SENTINEL_KEY: {
            "dropped": len(dropped_items),
            "kept": kept,
            "total": total,
            "reason": reason,
            "hash": content_hash(dropped_items),
        }
    }


def text_marker(dropped_lines: list[str], *, reason: str = "lines-elided") -> str:
    """Build a one-line text sentinel describing dropped lines."""
    return f"{TEXT_PREFIX} {len(dropped_lines)} {reason} (hash={content_hash(dropped_lines)})"
