"""SharedContext — compressed working-context handoffs between agents.

One agent assembles a working context (notes, findings, a plan) and ``put``s it
under a key; another agent (Claude, Codex, Gemini, a later run of the same one)
``get``s it back. The blob is run through the core :func:`tokenslim.compress`
pipeline before storage, so a large handoff costs few tokens to keep around and
the *compressed* form is what a reader pulls by default.

Design choices:

* **Compression reuses the core engine.** We wrap the content in a one-message
  array, call :func:`compress`, and keep both the compressed text and the
  original so ``get(full=True)`` can reconstruct exactly what was put in.
* **Auto-dedup.** ``put``-ing content that fully contains an already-stored
  entry's text drops the redundant entry; storing a substring of an existing
  entry is itself skipped. This keeps overlapping handoffs from piling up.
* **TTL + size cap.** Expired entries are evicted lazily on access; when the
  number of live entries exceeds ``max_entries`` the oldest are dropped.

Everything is in-process and dependency-free. A persistent/distributed backend
can sit behind the same ``put``/``get`` surface later.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from ..compress import Message, compress
from ..config import Config

__all__ = ["Handle", "SharedContext"]


@dataclass(frozen=True)
class Handle:
    """A lightweight reference to a stored context entry.

    Returned by :meth:`SharedContext.put`; carries enough metadata to describe
    the handoff without shipping the full payload around.
    """

    key: str
    compressed: str
    orig_tokens: int
    new_tokens: int
    created_at: float

    @property
    def ratio(self) -> float:
        """Fraction of tokens removed by compression (0.0 if unmeasured)."""
        if self.orig_tokens == 0:
            return 0.0
        return 1.0 - (self.new_tokens / self.orig_tokens)


@dataclass
class _Entry:
    key: str
    original: str
    compressed: str
    orig_tokens: int
    new_tokens: int
    created_at: float


class SharedContext:
    """In-process, compressed key/value store for inter-agent handoffs.

    Args:
        config: Config passed through to :func:`compress`. Defaults to the
            library defaults.
        ttl: Seconds an entry stays live; ``None`` keeps entries forever.
        max_entries: Hard cap on live entries; oldest evicted past this.
    """

    def __init__(
        self,
        config: Config | None = None,
        ttl: float | None = None,
        max_entries: int = 128,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._config = config
        self._ttl = ttl
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}

    def _compress_text(self, content: str) -> tuple[str, int, int]:
        """Run ``content`` through the core pipeline; return (compressed, orig, new)."""
        messages: list[Message] = [{"role": "user", "content": content}]
        out, stats = compress(messages, options=self._config)
        compressed = out[0].get("content", content)
        if not isinstance(compressed, str):
            # Compressor produced a block list; fall back to the original text.
            compressed = content
        return compressed, stats.orig_tokens, stats.new_tokens

    def put(self, key: str, content: str) -> Handle:
        """Store ``content`` under ``key`` (compressed) and return a handle.

        Overlapping entries are deduped: if ``content`` is a substring of an
        existing entry it is skipped (the superset is kept); if it is a superset
        of existing entries those are removed in its favour.
        """
        if not key:
            raise ValueError("key is required")
        if not isinstance(content, str) or not content:
            raise ValueError("content must be a non-empty string")

        compressed, orig_tokens, new_tokens = self._compress_text(content)
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            # Dedup: drop any live entry whose text is fully contained in this
            # one (this put supersedes them).
            for other_key, other in list(self._entries.items()):
                if other_key == key:
                    continue
                if other.original and other.original in content:
                    del self._entries[other_key]
            self._entries[key] = _Entry(
                key=key,
                original=content,
                compressed=compressed,
                orig_tokens=orig_tokens,
                new_tokens=new_tokens,
                created_at=now,
            )
            self._enforce_cap()
        return Handle(
            key=key,
            compressed=compressed,
            orig_tokens=orig_tokens,
            new_tokens=new_tokens,
            created_at=now,
        )

    def get(self, key: str, full: bool = False) -> str | None:
        """Return the entry for ``key``: compressed by default, original if ``full``.

        ``None`` if the key is unknown or its entry has expired.
        """
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            entry = self._entries.get(key)
            if entry is None:
                return None
            return entry.original if full else entry.compressed

    def handle(self, key: str) -> Handle | None:
        """Return the :class:`Handle` for ``key`` without its payload."""
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            entry = self._entries.get(key)
            if entry is None:
                return None
            return Handle(
                key=entry.key,
                compressed=entry.compressed,
                orig_tokens=entry.orig_tokens,
                new_tokens=entry.new_tokens,
                created_at=entry.created_at,
            )

    def delete(self, key: str) -> bool:
        """Remove ``key``; return whether it existed."""
        with self._lock:
            return self._entries.pop(key, None) is not None

    def keys(self) -> list[str]:
        """Live (non-expired) keys, oldest first."""
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            ordered = sorted(self._entries.values(), key=lambda e: e.created_at)
            return [e.key for e in ordered]

    def _evict_expired(self, now: float) -> None:
        """Drop entries past their TTL. Caller holds the lock."""
        if self._ttl is None:
            return
        for key, entry in list(self._entries.items()):
            if now - entry.created_at > self._ttl:
                del self._entries[key]

    def _enforce_cap(self) -> None:
        """Drop the oldest entries until within ``max_entries``. Caller holds the lock."""
        if len(self._entries) <= self._max_entries:
            return
        ordered = sorted(self._entries.values(), key=lambda e: e.created_at)
        excess = len(self._entries) - self._max_entries
        for entry in ordered[:excess]:
            del self._entries[entry.key]

    def __len__(self) -> int:
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            return len(self._entries)
