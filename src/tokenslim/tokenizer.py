"""Pluggable token counting.

The default counter is a fast, dependency-free heuristic estimator. When the
optional ``tiktoken`` extra is installed an accurate OpenAI tokenizer is used
for models it recognises; otherwise we transparently fall back to the heuristic.

All ratio math in the library funnels through :func:`count_tokens` so that
estimates stay consistent regardless of which backend is active.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol

__all__ = ["count_tokens", "Tokenizer", "HeuristicTokenizer", "get_tokenizer"]

# Rough word/punctuation splitter used by the heuristic backend.
_TOKEN_SPLIT_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Tokenizer(Protocol):
    """Anything that can turn text into an integer token count."""

    def count(self, text: str) -> int: ...


class HeuristicTokenizer:
    """Dependency-free token estimator.

    Combines a character-based estimate (~4 chars/token, the common rule of
    thumb for English) with a word/punctuation split, then averages them. This
    tracks real BPE tokenizers closely enough for compression-ratio decisions
    without pulling in a heavy dependency.
    """

    name = "heuristic"

    def count(self, text: str) -> int:
        if not text:
            return 0
        char_estimate = len(text) / 4.0
        piece_estimate = float(len(_TOKEN_SPLIT_RE.findall(text)))
        # Average the two signals; never report fewer than 1 token for
        # non-empty input.
        return max(1, round((char_estimate + piece_estimate) / 2.0))


class _TiktokenTokenizer:
    """Adapter around a tiktoken encoding."""

    def __init__(self, encoding, name: str) -> None:
        self._encoding = encoding
        self.name = name

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))


@lru_cache(maxsize=32)
def get_tokenizer(model: str | None = None) -> Tokenizer:
    """Return the best available tokenizer for ``model``.

    Falls back to the heuristic estimator when ``tiktoken`` is not installed or
    does not know the model. Results are cached per model.
    """
    if model:
        try:
            import tiktoken
        except ImportError:
            pass
        else:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            return _TiktokenTokenizer(encoding, f"tiktoken:{model}")
    return HeuristicTokenizer()


def count_tokens(text: str, model: str | None = None) -> int:
    """Count tokens in ``text`` using the best backend for ``model``."""
    return get_tokenizer(model).count(text)
