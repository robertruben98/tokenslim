"""Per-project persistent memory store — SQLite records + semantic recall.

A :class:`MemoryStore` is a long-lived, per-project knowledge base for agents:
durable facts, decisions and snippets that should outlive a single run. Records
live in SQLite (so they survive process restarts); recall is semantic, ranked by
cosine similarity over a pluggable :class:`~tokenslim.memory.embedder.Embedder`
(default: the dependency-free :class:`HashingEmbedder`).

Scoping is strict and mandatory. Every record carries a ``project`` key and
every query is filtered by it at the SQL level, so a search in project A can
never surface a row written under project B. There is no global / unscoped read.

Storage layout (one SQLite file, may hold many projects):

* ``memory`` — id, project, text, metadata (JSON), embedding (JSON), created_at.
* ``memory_fts`` — an FTS5 virtual table mirroring ``text`` for full-text
  prefiltering, *when the SQLite build has FTS5*. We detect this at open time
  and silently fall back to a Python-side scan when it is unavailable, so the
  store works on every interpreter.

Heavy vector backends (HNSW / Qdrant / Neo4j / mem0) are intentionally out of
scope here — they land later as optional extras behind this same API.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .embedder import Embedder, HashingEmbedder, cosine

__all__ = ["MemoryRecord", "MemoryStore"]


@dataclass(frozen=True)
class MemoryRecord:
    """A single stored memory and (on search) its relevance score."""

    id: str
    project: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    score: float = 0.0


def _fts5_available(conn: sqlite3.Connection) -> bool:
    """True when this SQLite build can create FTS5 virtual tables."""
    try:
        conn.execute("CREATE VIRTUAL TABLE _fts_probe USING fts5(x)")
        conn.execute("DROP TABLE _fts_probe")
        return True
    except sqlite3.OperationalError:
        return False


class MemoryStore:
    """Per-project persistent memory with semantic + full-text recall.

    Args:
        path: SQLite file path. Use ``":memory:"`` for an ephemeral store.
        embedder: Embedder for semantic ranking. Defaults to
            :class:`HashingEmbedder` (no extra deps). Inject a real model here
            for better recall; existing rows keep their stored embeddings.
    """

    def __init__(self, path: str = ":memory:", embedder: Embedder | None = None) -> None:
        self.path = path
        self._embedder: Embedder = embedder or HashingEmbedder()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._fts = _fts5_available(self._conn)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS memory ("
                "  id TEXT PRIMARY KEY,"
                "  project TEXT NOT NULL,"
                "  text TEXT NOT NULL,"
                "  metadata TEXT NOT NULL,"
                "  embedding TEXT NOT NULL,"
                "  created_at REAL NOT NULL"
                ")"
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_project ON memory(project)")
            if self._fts:
                # contentless-ish FTS mirror keyed by rowid==id for prefiltering.
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts "
                    "USING fts5(id UNINDEXED, project UNINDEXED, text)"
                )
            self._conn.commit()

    @property
    def fts_enabled(self) -> bool:
        """Whether FTS5 full-text prefiltering is active for this store."""
        return self._fts

    def add(
        self,
        text: str,
        project: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store ``text`` under ``project`` and return its new record id."""
        if not project:
            raise ValueError("project is required (no unscoped memory)")
        if not text:
            raise ValueError("text must be non-empty")
        rec_id = uuid.uuid4().hex
        embedding = self._embedder.embed(text)
        meta_json = json.dumps(metadata or {})
        with self._lock:
            self._conn.execute(
                "INSERT INTO memory (id, project, text, metadata, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (rec_id, project, text, meta_json, json.dumps(embedding), time.time()),
            )
            if self._fts:
                self._conn.execute(
                    "INSERT INTO memory_fts (id, project, text) VALUES (?, ?, ?)",
                    (rec_id, project, text),
                )
            self._conn.commit()
        return rec_id

    def get(self, id: str) -> MemoryRecord | None:
        """Return the record for ``id``, or ``None`` if unknown."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, project, text, metadata, created_at FROM memory WHERE id = ?",
                (id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def search(self, query: str, project: str, k: int = 5) -> list[MemoryRecord]:
        """Return the ``k`` records in ``project`` most relevant to ``query``.

        Ranking is cosine similarity over embeddings. When FTS5 is available the
        candidate set is prefiltered by a full-text match (falling back to the
        whole project when the text query matches nothing), so large projects
        don't pay a full scan on every call. Results are scoped to ``project``.
        """
        if not project:
            raise ValueError("project is required (no unscoped search)")
        if k <= 0:
            return []
        rows = self._candidate_rows(query, project)
        if not rows:
            return []
        q_vec = self._embedder.embed(query)
        scored: list[MemoryRecord] = []
        for row in rows:
            try:
                emb = json.loads(row["embedding"])
            except (TypeError, ValueError):
                emb = []
            score = cosine(q_vec, emb) if len(emb) == len(q_vec) else 0.0
            scored.append(self._row_to_record(row, score=score))
        scored.sort(key=lambda r: (r.score, r.created_at), reverse=True)
        return scored[:k]

    def _candidate_rows(self, query: str, project: str) -> list[sqlite3.Row]:
        """Project-scoped candidate rows for ranking (FTS prefilter if usable)."""
        with self._lock:
            if self._fts and query.strip():
                match = self._fts_query(query)
                if match:
                    rows = self._conn.execute(
                        "SELECT m.id, m.project, m.text, m.metadata, m.embedding, m.created_at "
                        "FROM memory_fts f JOIN memory m ON m.id = f.id "
                        "WHERE f.project = ? AND memory_fts MATCH ? ",
                        (project, match),
                    ).fetchall()
                    if rows:
                        return rows
            # Fallback / no-match: rank the whole project by embedding.
            return self._conn.execute(
                "SELECT id, project, text, metadata, embedding, created_at "
                "FROM memory WHERE project = ?",
                (project,),
            ).fetchall()

    @staticmethod
    def _fts_query(query: str) -> str:
        """Turn free text into a safe FTS5 OR-of-prefixes match expression."""
        import re

        terms = re.findall(r"[A-Za-z0-9]+", query)
        if not terms:
            return ""
        return " OR ".join(f'"{t}"*' for t in terms)

    @staticmethod
    def _row_to_record(row: sqlite3.Row, score: float = 0.0) -> MemoryRecord:
        try:
            metadata = json.loads(row["metadata"])
        except (TypeError, ValueError):
            metadata = {}
        return MemoryRecord(
            id=row["id"],
            project=row["project"],
            text=row["text"],
            metadata=metadata,
            created_at=row["created_at"],
            score=score,
        )

    def count(self, project: str | None = None) -> int:
        """Row count, optionally restricted to one ``project``."""
        with self._lock:
            if project is None:
                return self._conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
            return self._conn.execute(
                "SELECT COUNT(*) FROM memory WHERE project = ?", (project,)
            ).fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __len__(self) -> int:
        return self.count()
