"""Pluggable text embedders for semantic recall — zero heavy deps.

The memory store ranks recall results by cosine similarity over embeddings.
To keep the default install dependency-free (no model download, no numpy),
the built-in :class:`HashingEmbedder` projects a bag-of-words into a fixed-size
vector via feature hashing. It is deterministic, fast, and good enough to rank
topically-related text above unrelated text — which is all the tests and the
default experience need.

A real embedder (sentence-transformers, an API client, etc.) can be injected
anywhere an :class:`Embedder` is accepted, as long as it returns a fixed-length
list of floats for a string.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

__all__ = ["Embedder", "HashingEmbedder", "cosine"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Maps text to a fixed-length dense vector."""

    @property
    def dim(self) -> int:
        """Dimensionality of the returned vectors."""
        ...

    def embed(self, text: str) -> list[float]:
        """Return the embedding for ``text`` (length == :attr:`dim`)."""
        ...


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens — the unit of the bag-of-words model."""
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Feature-hashing bag-of-words embedder (the dependency-free default).

    Each token is hashed into one of ``dim`` buckets with a signed weight, the
    bucket counts are accumulated and L2-normalised. Two texts that share many
    tokens land close together under cosine similarity; unrelated texts do not.
    Deterministic across processes (uses BLAKE2b, not Python's salted ``hash``).
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _bucket(self, token: str) -> tuple[int, float]:
        """Hash a token to ``(bucket_index, sign)``."""
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "big")
        index = h % self._dim
        sign = 1.0 if (h >> 63) & 1 else -1.0
        return index, sign

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in _tokenize(text):
            index, sign = self._bucket(token)
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is zero)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
