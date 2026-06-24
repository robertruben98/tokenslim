"""Project-scoped persistent memory store.

SQLite for record metadata and contents, with a fast NumPy-based cosine
similarity index for semantic vector search, plus a BM25 fallback for keyword recall.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np

from .relevance import BM25Scorer

__all__ = ["ProjectMemoryStore"]


class ProjectMemoryStore:
    """Persistent project-scoped memory store.

    Saves records locally inside a `.tokenslim/` directory at the project root
    to prevent cross-project bleed. Uses SQLite for durability and NumPy/BM25
    for semantic/keyword search.
    """

    def __init__(
        self,
        db_path: str | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        if db_path is None:
            # Traverses up to find project root (marked by .git or pyproject.toml)
            root = os.getcwd()
            curr = root
            while True:
                if os.path.exists(os.path.join(curr, ".git")) or os.path.exists(
                    os.path.join(curr, "pyproject.toml")
                ):
                    root = curr
                    break
                parent = os.path.dirname(curr)
                if parent == curr:
                    break
                curr = parent

            dot_dir = os.path.join(root, ".tokenslim")
            os.makedirs(dot_dir, exist_ok=True)
            db_path = os.path.join(dot_dir, "memory.db")

        self.db_path = db_path
        self.embed_fn = embed_fn

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS memory ("
            "  id TEXT PRIMARY KEY,"
            "  content TEXT NOT NULL,"
            "  metadata TEXT,"
            "  embedding TEXT,"
            "  created_at REAL NOT NULL"
            ")"
        )
        self._conn.commit()

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Add a document to the memory store, optionally providing its embedding.

        If embed_fn is configured and no embedding is provided, it will generate it.
        Returns the unique ID of the inserted record.
        """
        record_id = str(uuid.uuid4())
        meta_str = json.dumps(metadata or {})

        if embedding is None and self.embed_fn is not None:
            embedding = self.embed_fn(content)

        embed_str = json.dumps(embedding) if embedding is not None else None

        self._conn.execute(
            "INSERT INTO memory (id, content, metadata, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (record_id, content, meta_str, embed_str, time.time()),
        )
        self._conn.commit()
        return record_id

    def get(self, record_id: str) -> dict[str, Any] | None:
        """Retrieve a stored record by its ID."""
        row = self._conn.execute(
            "SELECT content, metadata, embedding, created_at FROM memory WHERE id = ?",
            (record_id,),
        ).fetchone()
        if row is None:
            return None

        content, meta_str, embed_str, created_at = row
        return {
            "id": record_id,
            "content": content,
            "metadata": json.loads(meta_str) if meta_str else {},
            "embedding": json.loads(embed_str) if embed_str else None,
            "created_at": created_at,
        }

    def delete(self, record_id: str) -> bool:
        """Delete a record by its ID. Returns True if deleted, False if not found."""
        cursor = self._conn.execute("DELETE FROM memory WHERE id = ?", (record_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def search(
        self,
        query: str,
        limit: int = 5,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the store for matching memories.

        If a query embedding is provided (or generated), it uses cosine similarity
        over the database embeddings. Otherwise, falls back to BM25 keyword search.
        """
        if query_embedding is None and self.embed_fn is not None:
            query_embedding = self.embed_fn(query)

        rows = self._conn.execute(
            "SELECT id, content, metadata, embedding, created_at FROM memory"
        ).fetchall()

        if not rows:
            return []

        results = []
        if query_embedding is not None:
            # Vector cosine similarity search
            q_vec = np.array(query_embedding, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)

            for record_id, content, meta_str, embed_str, created_at in rows:
                if not embed_str:
                    continue
                v = np.array(json.loads(embed_str), dtype=np.float32)
                v_norm = np.linalg.norm(v)

                if q_norm > 0 and v_norm > 0:
                    similarity = float(np.dot(q_vec, v) / (q_norm * v_norm))
                else:
                    similarity = 0.0

                results.append(
                    {
                        "id": record_id,
                        "content": content,
                        "metadata": json.loads(meta_str) if meta_str else {},
                        "created_at": created_at,
                        "score": similarity,
                    }
                )
            # Sort by similarity descending
            results.sort(key=lambda x: x["score"], reverse=True)
        else:
            # Fallback to BM25 keyword search
            candidates = [r[1] for r in rows]
            scorer = BM25Scorer()
            scores = scorer.score(query, candidates)

            for score, (record_id, content, meta_str, _, created_at) in zip(scores, rows):
                results.append(
                    {
                        "id": record_id,
                        "content": content,
                        "metadata": json.loads(meta_str) if meta_str else {},
                        "created_at": created_at,
                        "score": score,
                    }
                )
            # Sort by score descending
            results.sort(key=lambda x: x["score"], reverse=True)

        return results[:limit]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
