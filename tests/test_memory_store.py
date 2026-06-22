"""Tests for the per-project memory store (issue #39)."""

from __future__ import annotations

import os

import pytest

from tokenslim.memory import HashingEmbedder, MemoryStore, cosine


def test_add_get_roundtrip() -> None:
    store = MemoryStore()
    rec_id = store.add("the deploy uses blue-green rollout", "projA", {"tag": "ops"})
    rec = store.get(rec_id)
    assert rec is not None
    assert rec.id == rec_id
    assert rec.project == "projA"
    assert rec.text == "the deploy uses blue-green rollout"
    assert rec.metadata == {"tag": "ops"}
    assert rec.created_at > 0


def test_get_unknown_returns_none() -> None:
    store = MemoryStore()
    assert store.get("does-not-exist") is None


def test_per_project_isolation_in_search() -> None:
    store = MemoryStore()
    store.add("the database connection pool size is 20", "projA")
    store.add("the database connection pool size is 20", "projB")
    a_id = store.add("alpha-only secret token rotation policy", "projA")

    # A query that matches content in both projects must only return projA rows.
    results = store.search("database connection pool", "projA", k=10)
    assert results, "expected at least one hit in projA"
    assert all(r.project == "projA" for r in results)
    assert all(r.id != "" for r in results)

    # The projA-only fact must never appear when searching projB.
    b_results = store.search("alpha-only secret token rotation", "projB", k=10)
    assert all(r.project == "projB" for r in b_results)
    assert all(r.id != a_id for r in b_results)


def test_search_ranks_relevant_higher() -> None:
    store = MemoryStore()
    store.add("kubernetes pod autoscaling and resource limits", "p")
    store.add("the cat sat on the warm windowsill all afternoon", "p")
    store.add("horizontal pod autoscaler scales kubernetes deployments", "p")

    results = store.search("kubernetes autoscaling pods", "p", k=3)
    # Topical rows rank above the unrelated cat sentence. With FTS5 the cat row
    # is prefiltered out entirely (no shared terms); either way it must never
    # outrank a kubernetes row.
    assert results, "expected hits"
    assert "kubernetes" in results[0].text
    assert results[0].score >= results[-1].score
    cat_scores = [r.score for r in results if "cat sat" in r.text]
    k8s_scores = [r.score for r in results if "kubernetes" in r.text]
    if cat_scores:
        assert max(k8s_scores) > max(cat_scores)


def test_embedding_fallback_ranks_all_when_no_fts_match() -> None:
    # A query with no token overlap forces the pure-embedding fallback path
    # (FTS MATCH finds nothing -> whole project is ranked), so every row is a
    # candidate. Semantic ranking still applies.
    store = MemoryStore()
    store.add("kubernetes pod autoscaling and resource limits", "p")
    store.add("the cat sat on the warm windowsill all afternoon", "p")
    results = store.search("zzz qqq nonmatching", "p", k=5)
    assert len(results) == 2  # fallback considered the whole project
    assert all(r.project == "p" for r in results)


def test_search_requires_project() -> None:
    store = MemoryStore()
    with pytest.raises(ValueError):
        store.search("anything", "", k=3)


def test_add_requires_project_and_text() -> None:
    store = MemoryStore()
    with pytest.raises(ValueError):
        store.add("text", "")
    with pytest.raises(ValueError):
        store.add("", "proj")


def test_search_k_zero_returns_empty() -> None:
    store = MemoryStore()
    store.add("something", "p")
    assert store.search("something", "p", k=0) == []


def test_persistence_across_reopen(tmp_path) -> None:
    db = os.path.join(str(tmp_path), "mem.sqlite3")
    store = MemoryStore(path=db)
    rec_id = store.add("a durable fact that survives restarts", "proj")
    store.close()

    reopened = MemoryStore(path=db)
    rec = reopened.get(rec_id)
    assert rec is not None
    assert rec.text == "a durable fact that survives restarts"
    hits = reopened.search("durable fact survives", "proj", k=5)
    assert any(h.id == rec_id for h in hits)
    reopened.close()


def test_count_scoping() -> None:
    store = MemoryStore()
    store.add("x", "a")
    store.add("y", "a")
    store.add("z", "b")
    assert store.count("a") == 2
    assert store.count("b") == 1
    assert store.count() == 3
    assert len(store) == 3


def test_injected_embedder_is_used() -> None:
    # A custom embedder with a distinct dim should be honoured end to end.
    store = MemoryStore(embedder=HashingEmbedder(dim=32))
    store.add("injected embedder content here", "p")
    hits = store.search("injected embedder content", "p", k=1)
    assert len(hits) == 1


def test_embedder_cosine_properties() -> None:
    emb = HashingEmbedder(dim=64)
    v1 = emb.embed("shared overlapping words here")
    v2 = emb.embed("shared overlapping words here too")
    v3 = emb.embed("completely unrelated zzz qqq")
    assert cosine(v1, v1) == pytest.approx(1.0, abs=1e-9)
    assert cosine(v1, v2) > cosine(v1, v3)
    assert emb.embed("") == [0.0] * 64
