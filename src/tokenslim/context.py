"""SharedContext — serialized and compressed inter-agent context handoffs.

Allows serializing and compressing facts, states, and logs to pass context between
agents while auto-deduplicating overlapping facts via Jaccard similarity.
"""

from __future__ import annotations

import json
import re

from .compressors.text import TextCompressor
from .config import Config

__all__ = ["SharedContext"]

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "is",
    "was",
    "were",
    "are",
    "be",
    "been",
    "it",
    "this",
    "that",
    "he",
    "she",
    "they",
    "we",
    "i",
    "you",
    "his",
    "her",
    "their",
    "our",
    "my",
    "your",
    "them",
    "us",
    "him",
    "me",
    "as",
    "by",
    "from",
    "about",
    "into",
    "through",
    "over",
    "under",
    "again",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "should",
    "would",
    "now",
}


class SharedContext:
    """Manages context stashes for inter-agent communication with auto-deduplication."""

    def __init__(self, items: list[str] | None = None, threshold: float = 0.5) -> None:
        self.items: list[str] = []
        self.threshold = threshold
        if items:
            for item in items:
                self.add_item(item)

    def add_item(self, item: str) -> bool:
        """Add a fact/log to the context.

        Returns True if added, False if dropped due to overlapping/redundancy.
        """
        stripped = item.strip()
        if not stripped:
            return False

        # Compare Jaccard word overlap against existing items
        for existing in self.items:
            sim = self._similarity(stripped, existing)
            if sim >= self.threshold:
                # Merge/keep the longer one to retain maximum detail
                if len(stripped) > len(existing):
                    idx = self.items.index(existing)
                    self.items[idx] = stripped
                return False

        self.items.append(stripped)
        return True

    def _similarity(self, text1: str, text2: str) -> float:
        words1 = set(re.findall(r"\b[a-zA-Z0-9]+\b", text1.lower())) - _STOPWORDS
        words2 = set(re.findall(r"\b[a-zA-Z0-9]+\b", text2.lower())) - _STOPWORDS
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    def serialize(self, compress: bool = True, target_ratio: float = 0.5) -> str:
        """Serialize the context items to a compact string."""
        if not compress:
            return json.dumps({"version": "1.0", "items": self.items}, ensure_ascii=False)

        config = Config(target_ratio=target_ratio, min_bytes=0, ccr=False)
        text_comp = TextCompressor(config)
        compressed_items = [text_comp(item) for item in self.items]

        return json.dumps(
            {"version": "1.0", "items": compressed_items},
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def deserialize(cls, payload: str) -> SharedContext:
        """Load context items back from a serialized string."""
        data = json.loads(payload)
        items = data.get("items", [])
        return cls(items=items)
