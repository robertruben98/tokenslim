"""Relevance scoring — query-aware selection.

A zero-dependency BM25 scorer ranks candidate strings (log lines, search hits,
JSON rows) by relevance to a query. Compressors use it to keep what the user is
actually asking about, not just the head/tail of the input.

The :class:`Scorer` protocol leaves room for an embedding/hybrid scorer to drop
in later behind the same interface.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, runtime_checkable

__all__ = ["BM25Scorer", "Scorer", "tokenize"]

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/identifier tokenizer shared by the scorer."""
    return _TOKEN_RE.findall(text.lower())


@runtime_checkable
class Scorer(Protocol):
    """Scores each candidate string against the query (higher = more relevant)."""

    def score(self, query: str, candidates: list[str]) -> list[float]: ...


class BM25Scorer:
    """Okapi BM25 over a candidate corpus.

    ``k1`` controls term-frequency saturation; ``b`` controls length
    normalisation. Defaults are the standard BM25 values.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return [0.0] * len(candidates)

        docs = [tokenize(c) for c in candidates]
        doc_lens = [len(d) for d in docs]
        n_docs = len(docs)
        avgdl = (sum(doc_lens) / n_docs) or 1.0

        # Document frequency per query term.
        df: Counter[str] = Counter()
        unique_query_terms = set(query_terms)
        for doc in docs:
            present = unique_query_terms.intersection(doc)
            for term in present:
                df[term] += 1

        # BM25 idf with the +1 smoothing that keeps it non-negative.
        idf = {
            term: math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            for term in unique_query_terms
        }

        scores: list[float] = []
        for doc, dl in zip(docs, doc_lens):
            tf = Counter(doc)
            s = 0.0
            for term in unique_query_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * dl / avgdl)
                s += idf[term] * (freq * (self.k1 + 1)) / denom
            scores.append(s)
        return scores
