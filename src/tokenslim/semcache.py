"""Optional semantic response cache with a safety-first lexical guard.

Return a previously stored response when a new prompt is an exact match or
semantically close enough (cosine similarity over embeddings) to a cached one.

The design follows a real-embedding calibration experiment (RTX 5070 Ti,
three sentence-transformers models, 120 hand-written EN+ES prompt pairs):

- Naive cosine at the commonly proposed 0.95 threshold serves a WRONG cached
  answer on 5-10% of adversarial near-miss pairs (all-MiniLM-L6-v2 5%,
  all-mpnet-base-v2 10%, bge-small-en-v1.5 7.5%).
- The surviving false positives are exactly the dangerous class: date swaps
  ("June 5th" vs "July 5th") and polarity flips ("enable dark mode" vs
  "disable dark mode") score 0.96-0.98 — embeddings barely register them. No
  cosine threshold reaches zero near-miss FPs at useful recall (recall
  collapses to 0-5% at the 0.98-0.99 needed for zero FPs).
- Defaults here: threshold **0.96** plus a cheap lexical guard (numeric/date
  token equality + antonym/negation flip detection). The guard killed every
  observed high-similarity false positive in the experiment; its failure mode
  is an occasional extra cache miss, never a wrong answer.

The core stays dependency-free: bring any embedding backend satisfying the
:class:`Embedder` protocol. ``pip install tokenslim[semantic]`` enables
:class:`SentenceTransformerEmbedder` (recommended model
``BAAI/bge-small-en-v1.5`` — the experiment's best safety/recall trade-off at
threshold 0.96; prefer a multilingual model for heavy non-English traffic).

Nothing in :func:`tokenslim.compress` uses this — the cache is opt-in::

    from tokenslim import SemanticCache, SentenceTransformerEmbedder

    cache = SemanticCache(SentenceTransformerEmbedder())
    if (hit := cache.get(prompt)) is not None:
        return hit.response
    response = call_llm(prompt)
    cache.put(prompt, response)

The cache never raises from :meth:`SemanticCache.get` / :meth:`SemanticCache.put`:
embedder failures degrade to a miss / a skipped insert. It is not thread-safe.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "ANTONYM_PAIRS",
    "CacheHit",
    "Embedder",
    "HTTPEmbedder",
    "SemanticCache",
    "SentenceTransformerEmbedder",
]

# Antonym / polarity pairs whose one-sided swap between two prompts flips the
# meaning while cosine similarity stays 0.96-0.98 (observed in the calibration
# experiment). Matching is exact-word on lowercased letter tokens.
ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("enable", "disable"),
    ("on", "off"),
    ("add", "remove"),
    ("always", "never"),
    ("con", "sin"),
    ("increase", "decrease"),
    ("create", "delete"),
    ("start", "stop"),
)

# Words whose mere presence difference flips polarity ("is it safe" vs
# "is it not safe").
_NEGATION_WORDS: tuple[str, ...] = ("not",)

# ISO dates are kept whole (alternation order matters); other digit runs keep
# an optional decimal part so "2.5" is one token.
_NUMERIC_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d+(?:\.\d+)?")
# Letter-only tokens (unicode-aware, so accented Spanish words stay whole).
_WORD_RE = re.compile(r"[^\W\d_]+")

_MONTHS = frozenset(
    {
        # English
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        # Spanish
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    }
)


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to embedding vectors (one vector per input text)."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class CacheHit:
    """A served cache entry: the response plus provenance for auditing."""

    response: str
    similarity: float
    key_prompt: str


@dataclass
class _Entry:
    prompt: str
    response: str
    vector: np.ndarray


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _numeric_tokens(text: str) -> set[str]:
    """Numbers, ISO dates and month names — the tokens that must match exactly."""
    tokens = set(_NUMERIC_RE.findall(text))
    tokens |= _words(text) & _MONTHS
    return tokens


def _lexical_guard(a: str, b: str) -> bool:
    """Return True when serving ``b``'s cached response for prompt ``a`` is safe.

    Rejects (returns False) when the numeric/date token sets differ, or when an
    antonym/negation flip is detected — the two failure classes that survive
    any useful cosine threshold (experiment: date swaps and enable/disable
    flips score 0.96-0.98). A False here only costs a cache miss.
    """
    if _numeric_tokens(a) != _numeric_tokens(b):
        return False
    words_a, words_b = _words(a), _words(b)
    for negation in _NEGATION_WORDS:
        if (negation in words_a) != (negation in words_b):
            return False
    for x, y in ANTONYM_PAIRS:
        forward = x in words_a and y in words_b and y not in words_a and x not in words_b
        backward = y in words_a and x in words_b and x not in words_a and y not in words_b
        if forward or backward:
            return False
    return True


def _hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class SemanticCache:
    """LRU prompt→response cache keyed by exact match, then embedding similarity.

    ``threshold`` defaults to 0.96 (see module docstring; ``Config.
    semantic_cache_threshold`` carries the same default for wiring). ``guard``
    keeps the lexical guard on — disable it only if wrong-but-similar answers
    are acceptable. ``max_entries`` bounds memory; least-recently-used entries
    are evicted first (both hits and puts refresh recency).
    """

    def __init__(
        self,
        embedder: Embedder,
        threshold: float = 0.96,
        max_entries: int = 1024,
        guard: bool = True,
    ) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.max_entries = max(1, max_entries)
        self.guard = guard
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Drop all cached entries."""
        self._entries.clear()

    def get(self, prompt: str) -> CacheHit | None:
        """Return the best safe hit for ``prompt``, or None. Never raises."""
        key = _hash(prompt)
        entry = self._entries.get(key)
        if entry is not None and entry.prompt == prompt:
            # Exact-match fast path: no embedding call, similarity 1.0.
            self._entries.move_to_end(key)
            return CacheHit(entry.response, 1.0, entry.prompt)
        if not self._entries:
            return None
        try:
            query = self._embed_one(prompt)
            if query is None:
                return None
            keys = list(self._entries.keys())
            matrix = np.stack([self._entries[k].vector for k in keys])
            sims = matrix @ query
            for idx in np.argsort(sims)[::-1]:
                similarity = float(sims[idx])
                if similarity < self.threshold:
                    break
                candidate = self._entries[keys[idx]]
                if self.guard and not _lexical_guard(prompt, candidate.prompt):
                    continue  # unsafe near-miss; try the next-best candidate
                self._entries.move_to_end(keys[idx])
                return CacheHit(candidate.response, similarity, candidate.prompt)
        except Exception:
            return None
        return None

    def put(self, prompt: str, response: str) -> None:
        """Store ``response`` under ``prompt``, evicting LRU entries. Never raises."""
        key = _hash(prompt)
        existing = self._entries.get(key)
        if existing is not None:
            existing.response = response
            self._entries.move_to_end(key)
            return
        vector = self._embed_one(prompt)
        if vector is None:
            return
        self._entries[key] = _Entry(prompt, response, vector)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def _embed_one(self, prompt: str) -> np.ndarray | None:
        """Embed and L2-normalize one prompt; None on embedder failure."""
        try:
            raw = self.embedder.embed([prompt])[0]
            vector = np.asarray(raw, dtype=np.float64)
        except Exception:
            return None
        if vector.ndim != 1:
            return None
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0.0:
            return None
        return vector / norm


class SentenceTransformerEmbedder:
    """:class:`Embedder` backed by sentence-transformers (optional extra).

    Requires ``pip install tokenslim[semantic]``. The default model is the
    calibration experiment's recommendation for threshold 0.96. Thresholds are
    NOT transferable between models — recalibrate if you swap models.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install it with: pip install tokenslim[semantic]"
            ) from exc
        self._model = SentenceTransformer(model_name, device=device)  # pragma: no cover

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vec)) for vec in vectors]


class HTTPEmbedder:
    """:class:`Embedder` backed by a remote embedding HTTP service.

    Lets the cache use a GPU on another machine without adding heavy local
    dependencies. The service contract is one endpoint:

    ``POST {base_url}/embed`` with body ``{"texts": ["...", ...]}`` returning
    ``{"embeddings": [[...], ...]}`` — one vector per input text.

    Network or protocol failures raise :class:`OSError`; callers that must
    never fail should catch it and skip caching for that call.
    """

    def __init__(self, base_url: str, timeout: float = 10.0):
        self._url = base_url.rstrip("/") + "/embed"
        self._timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        import json
        import urllib.request

        body = json.dumps({"texts": texts}).encode()
        request = urllib.request.Request(
            self._url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode())
        except ValueError as exc:  # malformed JSON body
            raise OSError(f"embedding service at {self._url} returned invalid JSON") from exc
        except OSError as exc:
            raise OSError(f"embedding service at {self._url} failed: {exc}") from exc
        embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OSError(f"embedding service at {self._url} returned a malformed response")
        return [[float(value) for value in vector] for vector in embeddings]
