"""CCR store — content-addressed cache for dropped originals.

Compress-Cache-Retrieve: when a compressor elides material it ``put()``s the
original into a store keyed by its content hash, embeds that hash in a marker,
and the model (or a tool) can later ``get()`` the full original back. That makes
lossy compression *safe* — nothing is truly lost, just moved out of the prompt.

Two backends ship here:

* :class:`InMemoryCCRStore` — a process-local dict. Default; ideal for tests and
  single-process use.
* :class:`SQLiteCCRStore` — a local SQLite file with ``created_at`` and optional
  TTL eviction, so records survive across processes.

A Redis backend (distributed) lands later behind this same interface.

The store is *content-addressed*: ``put`` is idempotent — storing the same bytes
twice yields the same hash and a single row. The hash matches
:func:`tokenslim.ccr.content_hash` so a sentinel's hash and its store key agree.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import Protocol, runtime_checkable

from .ccr import content_hash
from .config import Config

__all__ = [
    "CCRStore",
    "InMemoryCCRStore",
    "SQLiteCCRStore",
    "RedisCCRStore",
    "get_store",
]


@runtime_checkable
class CCRStore(Protocol):
    """Content-addressed key/value store for dropped originals."""

    def put(self, original: str) -> str:
        """Store ``original`` and return its content hash (the retrieval key)."""
        ...

    def get(self, hash: str) -> str | None:
        """Return the original for ``hash``, or ``None`` if unknown/expired."""
        ...


class InMemoryCCRStore:
    """Process-local, thread-safe in-memory CCR store."""

    def __init__(self, ttl: int | None = None) -> None:
        self._ttl = ttl
        self._lock = threading.Lock()
        # hash -> (original, created_at)
        self._data: dict[str, tuple[str, float]] = {}

    def put(self, original: str) -> str:
        key = content_hash(original)
        with self._lock:
            self._data[key] = (original, time.time())
        return key

    def get(self, hash: str) -> str | None:
        with self._lock:
            entry = self._data.get(hash)
            if entry is None:
                return None
            original, created_at = entry
            if self._ttl is not None and time.time() - created_at > self._ttl:
                del self._data[hash]
                return None
            return original

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class SQLiteCCRStore:
    """File-backed CCR store with created_at + optional TTL eviction."""

    def __init__(self, path: str, ttl: int | None = None) -> None:
        self.path = path
        self._ttl = ttl
        self._lock = threading.Lock()
        # check_same_thread=False + our own lock keeps it usable across threads.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS ccr ("
            "  hash TEXT PRIMARY KEY,"
            "  original TEXT NOT NULL,"
            "  created_at REAL NOT NULL"
            ")"
        )
        self._conn.commit()

    def put(self, original: str) -> str:
        key = content_hash(original)
        with self._lock:
            # INSERT OR REFRESH: keep it idempotent but bump created_at so a
            # re-stored record's TTL clock restarts.
            self._conn.execute(
                "INSERT INTO ccr (hash, original, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT(hash) DO UPDATE SET created_at = excluded.created_at",
                (key, original, time.time()),
            )
            self._conn.commit()
        return key

    def get(self, hash: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT original, created_at FROM ccr WHERE hash = ?", (hash,)
            ).fetchone()
            if row is None:
                return None
            original, created_at = row
            if self._ttl is not None and time.time() - created_at > self._ttl:
                self._conn.execute("DELETE FROM ccr WHERE hash = ?", (hash,))
                self._conn.commit()
                return None
            return original

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM ccr").fetchone()[0]


class RedisCCRStore:
    """Distributed CCR store backed by Redis."""

    def __init__(self, url: str, ttl: int | None = None) -> None:
        try:
            import redis
        except ImportError as e:
            raise ImportError(
                "The 'redis' package is required to use the Redis CCR backend. "
                "Install it with `pip install redis`."
            ) from e
        self.url = url
        self._ttl = ttl
        self._client = redis.Redis.from_url(url, decode_responses=True)

    def put(self, original: str) -> str:
        key = content_hash(original)
        redis_key = f"tokenslim:ccr:{key}"
        if self._ttl is not None:
            self._client.setex(redis_key, self._ttl, original)
        else:
            self._client.set(redis_key, original)
        return key

    def get(self, hash: str) -> str | None:
        redis_key = f"tokenslim:ccr:{hash}"
        val = self._client.get(redis_key)
        if isinstance(val, (str, bytes)):
            return val.decode("utf-8") if isinstance(val, bytes) else val
        return None

    def __len__(self) -> int:
        return sum(1 for _ in self._client.scan_iter("tokenslim:ccr:*"))


def get_store(config: Config) -> CCRStore:
    """Build the CCR store selected by ``config.ccr_backend``."""
    backend = (config.ccr_backend or "memory").lower()
    if backend == "memory":
        return InMemoryCCRStore(ttl=config.ccr_ttl)
    if backend == "sqlite":
        return SQLiteCCRStore(config.ccr_path, ttl=config.ccr_ttl)
    if backend == "redis":
        return RedisCCRStore(config.redis_url, ttl=config.ccr_ttl)
    raise ValueError(f"unknown CCR backend {backend!r} (expected 'memory', 'sqlite', or 'redis')")
