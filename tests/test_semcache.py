"""Tests for the optional semantic cache (semcache.py)."""

from __future__ import annotations

import math

import pytest

from tokenslim import CacheHit, Embedder, SemanticCache, SentenceTransformerEmbedder
from tokenslim.config import load_config
from tokenslim.semcache import ANTONYM_PAIRS, _lexical_guard


class FakeEmbedder:
    """Deterministic embedder with hand-set vectors per exact text."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = dict(vectors)
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [list(self.vectors[t]) for t in texts]


class ExplodingEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding backend down")


def _vec(sim: float) -> list[float]:
    """Unit vector whose cosine with [1, 0, 0] is exactly ``sim``."""
    return [sim, math.sqrt(1.0 - sim * sim), 0.0]


def _cache(vectors: dict[str, list[float]], **kwargs) -> tuple[SemanticCache, FakeEmbedder]:
    embedder = FakeEmbedder(vectors)
    return SemanticCache(embedder, **kwargs), embedder


# --- Embedder protocol ---


def test_embedder_protocol_conformance():
    assert isinstance(FakeEmbedder({}), Embedder)
    assert isinstance(ExplodingEmbedder(), Embedder)
    assert issubclass(SentenceTransformerEmbedder, object)  # importable without the extra

    class NotAnEmbedder:
        def encode(self, texts):
            return []

    assert not isinstance(NotAnEmbedder(), Embedder)


# --- Hit / miss around the threshold ---


def test_hit_above_threshold():
    cache, _ = _cache({"what is python": _vec(1.0), "tell me about python": _vec(0.97)})
    cache.put("what is python", "A programming language.")
    hit = cache.get("tell me about python")
    assert hit is not None, "0.97 >= 0.96 must hit"
    assert isinstance(hit, CacheHit)
    assert hit.response == "A programming language."
    assert hit.key_prompt == "what is python"
    assert hit.similarity == pytest.approx(0.97, abs=1e-9)


def test_miss_below_threshold():
    cache, _ = _cache({"what is python": _vec(1.0), "what is a snake": _vec(0.90)})
    cache.put("what is python", "A programming language.")
    assert cache.get("what is a snake") is None, "0.90 < 0.96 must miss"


def test_custom_threshold():
    cache, _ = _cache({"what is python": _vec(1.0), "what is a snake": _vec(0.90)}, threshold=0.85)
    cache.put("what is python", "A programming language.")
    assert cache.get("what is a snake") is not None


# --- Exact-match fast path ---


def test_exact_match_skips_embedding():
    cache, embedder = _cache({"hello world": _vec(1.0)})
    cache.put("hello world", "hi")
    assert len(embedder.calls) == 1  # one embed for the put
    hit = cache.get("hello world")
    assert hit is not None
    assert hit.similarity == 1.0
    assert hit.response == "hi"
    assert hit.key_prompt == "hello world"
    assert len(embedder.calls) == 1, "exact match must not embed again"


def test_get_on_empty_cache_never_embeds():
    cache, embedder = _cache({})
    assert cache.get("anything") is None
    assert embedder.calls == []


def test_put_same_prompt_updates_response():
    cache, embedder = _cache({"q": _vec(1.0)})
    cache.put("q", "old")
    cache.put("q", "new")
    assert len(cache) == 1
    assert len(embedder.calls) == 1, "re-put of the same prompt must not re-embed"
    hit = cache.get("q")
    assert hit is not None and hit.response == "new"


# --- LRU eviction ---


def test_lru_eviction_at_max_entries():
    vectors = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0], "c": [0.0, 0.0, 1.0]}
    cache, _ = _cache(vectors, max_entries=2)
    cache.put("a", "ra")
    cache.put("b", "rb")
    cache.put("c", "rc")
    assert len(cache) == 2
    assert cache.get("a") is None, "oldest entry must be evicted"
    assert cache.get("b") is not None
    assert cache.get("c") is not None


def test_get_refreshes_lru_order():
    vectors = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0], "c": [0.0, 0.0, 1.0]}
    cache, _ = _cache(vectors, max_entries=2)
    cache.put("a", "ra")
    cache.put("b", "rb")
    assert cache.get("a") is not None  # touch "a" so "b" becomes LRU
    cache.put("c", "rc")
    assert cache.get("b") is None, "least-recently-used entry must be evicted"
    assert cache.get("a") is not None


# --- Lexical guard: the experiment's dangerous near-miss classes ---


def test_guard_rejects_date_swap_even_at_sim_099():
    cache, _ = _cache(
        {
            "book me a flight on June 5th": _vec(1.0),
            "book me a flight on July 5th": _vec(0.99),
        }
    )
    cache.put("book me a flight on June 5th", "Booked for June 5th.")
    assert cache.get("book me a flight on July 5th") is None


def test_guard_rejects_number_swap_even_at_sim_099():
    cache, _ = _cache(
        {"set the timeout to 30 seconds": _vec(1.0), "set the timeout to 60 seconds": _vec(0.99)}
    )
    cache.put("set the timeout to 30 seconds", "timeout=30")
    assert cache.get("set the timeout to 60 seconds") is None


def test_guard_rejects_iso_date_swap():
    cache, _ = _cache({"deploy on 2026-07-03": _vec(1.0), "deploy on 2026-07-04": _vec(0.99)})
    cache.put("deploy on 2026-07-03", "scheduled")
    assert cache.get("deploy on 2026-07-04") is None


def test_guard_rejects_antonym_flip_even_at_sim_099():
    cache, _ = _cache(
        {"enable dark mode in Slack": _vec(1.0), "disable dark mode in Slack": _vec(0.99)}
    )
    cache.put("enable dark mode in Slack", "Toggle it on in preferences.")
    assert cache.get("disable dark mode in Slack") is None


def test_guard_rejects_negation_flip():
    cache, _ = _cache(
        {"is this file safe to delete": _vec(1.0), "is this file not safe to delete": _vec(0.99)}
    )
    cache.put("is this file safe to delete", "yes")
    assert cache.get("is this file not safe to delete") is None


def test_guard_rejects_spanish_con_sin_flip():
    cache, _ = _cache(
        {"café con leche por favor": _vec(1.0), "café sin leche por favor": _vec(0.99)}
    )
    cache.put("café con leche por favor", "con leche")
    assert cache.get("café sin leche por favor") is None


def test_guard_allows_benign_paraphrase_with_matching_numbers():
    cache, _ = _cache({"list 5 uses of python": _vec(1.0), "give me 5 uses of python": _vec(0.97)})
    cache.put("list 5 uses of python", "here are 5")
    hit = cache.get("give me 5 uses of python")
    assert hit is not None and hit.response == "here are 5"


def test_guard_disabled_bypasses_lexical_checks():
    cache, _ = _cache(
        {"enable dark mode in Slack": _vec(1.0), "disable dark mode in Slack": _vec(0.99)},
        guard=False,
    )
    cache.put("enable dark mode in Slack", "Toggle it on in preferences.")
    hit = cache.get("disable dark mode in Slack")
    assert hit is not None, "guard=False must serve the raw cosine hit"
    assert hit.similarity == pytest.approx(0.99, abs=1e-9)


def test_guard_falls_through_to_next_safe_candidate():
    vectors = {
        "set the timeout to 60 seconds": _vec(0.99),  # closer, but number differs
        "please set the timeout to 30 seconds": _vec(0.97),  # safe paraphrase
        "set the timeout to 30 seconds": _vec(1.0),  # query
    }
    cache, _ = _cache(vectors)
    cache.put("set the timeout to 60 seconds", "timeout=60")
    cache.put("please set the timeout to 30 seconds", "timeout=30")
    hit = cache.get("set the timeout to 30 seconds")
    assert hit is not None, "guard must skip the unsafe best match, not abort"
    assert hit.response == "timeout=30"
    assert hit.key_prompt == "please set the timeout to 30 seconds"
    assert hit.similarity == pytest.approx(0.97, abs=1e-9)


def test_lexical_guard_direct():
    assert _lexical_guard("same words 42", "same words 42")
    assert _lexical_guard("how do I sort a list", "how can I sort a list")
    assert not _lexical_guard("meeting in June", "meeting in July")
    assert not _lexical_guard("always retry", "never retry")
    assert isinstance(ANTONYM_PAIRS, tuple)
    assert ("enable", "disable") in ANTONYM_PAIRS
    assert ("con", "sin") in ANTONYM_PAIRS


# --- Never-raise behaviour ---


def test_embedder_failure_degrades_to_miss():
    cache = SemanticCache(ExplodingEmbedder())
    cache.put("q", "r")  # swallowed: nothing stored
    assert len(cache) == 0
    assert cache.get("q") is None


def test_embedder_failure_on_get_after_good_put():
    embedder = FakeEmbedder({"q": _vec(1.0)})
    cache = SemanticCache(embedder)
    cache.put("q", "r")
    embedder.vectors.clear()  # unknown prompt -> KeyError inside embed
    assert cache.get("other prompt") is None
    assert cache.get("q") is not None, "exact path still works without embedding"


def test_zero_vector_is_rejected():
    cache, _ = _cache({"q": [0.0, 0.0, 0.0]})
    cache.put("q", "r")
    assert len(cache) == 0


def test_clear():
    cache, _ = _cache({"q": _vec(1.0)})
    cache.put("q", "r")
    cache.clear()
    assert len(cache) == 0
    assert cache.get("q") is None


# --- Config knob & optional extra ---


def test_config_semantic_cache_threshold():
    assert load_config().semantic_cache_threshold == 0.96
    cfg = load_config(env={"TOKENSLIM_SEMANTIC_CACHE_THRESHOLD": "0.9"})
    assert cfg.semantic_cache_threshold == 0.9


def test_sentence_transformer_embedder_requires_extra():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("sentence-transformers is installed; ImportError path untestable")
    with pytest.raises(ImportError, match=r"tokenslim\[semantic\]"):
        SentenceTransformerEmbedder()
