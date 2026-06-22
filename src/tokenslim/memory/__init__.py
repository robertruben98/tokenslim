"""tokenslim.memory — per-project memory + compressed inter-agent handoffs (M3).

Two surfaces:

* :class:`MemoryStore` — durable, per-project knowledge base (SQLite records +
  semantic recall over a pluggable embedder). Strict per-project scoping.
* :class:`SharedContext` — compressed working-context handoffs between agents,
  built on the core :func:`tokenslim.compress` pipeline.

Both are dependency-free by default; heavy vector/graph backends are reserved as
future optional extras.
"""

from __future__ import annotations

from .embedder import Embedder, HashingEmbedder, cosine
from .shared import Handle, SharedContext
from .store import MemoryRecord, MemoryStore

__all__ = [
    "MemoryStore",
    "MemoryRecord",
    "SharedContext",
    "Handle",
    "Embedder",
    "HashingEmbedder",
    "cosine",
]
