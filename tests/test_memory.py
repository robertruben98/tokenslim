import os
import tempfile

import pytest

from tokenslim.memory import ProjectMemoryStore


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_add_and_get(temp_db):
    store = ProjectMemoryStore(db_path=temp_db)
    doc_id = store.add("Hello world", metadata={"source": "test"})

    record = store.get(doc_id)
    assert record is not None
    assert record["content"] == "Hello world"
    assert record["metadata"] == {"source": "test"}
    assert record["embedding"] is None

    store.close()


def test_delete(temp_db):
    store = ProjectMemoryStore(db_path=temp_db)
    doc_id = store.add("To be deleted")
    assert store.get(doc_id) is not None

    deleted = store.delete(doc_id)
    assert deleted is True
    assert store.get(doc_id) is None

    deleted_again = store.delete(doc_id)
    assert deleted_again is False

    store.close()


def test_cosine_similarity_search(temp_db):
    store = ProjectMemoryStore(db_path=temp_db)

    # Add records with embeddings
    # Document 1: [1, 0] (x-axis)
    # Document 2: [0, 1] (y-axis)
    id1 = store.add("Document 1", embedding=[1.0, 0.0])
    id2 = store.add("Document 2", embedding=[0.0, 1.0])

    # Search with query embedding closer to Document 1: [0.9, 0.1]
    res1 = store.search("Query 1", query_embedding=[0.9, 0.1])
    assert len(res1) == 2
    assert res1[0]["id"] == id1
    assert res1[0]["score"] > 0.8

    # Search with query embedding closer to Document 2: [0.1, 0.9]
    res2 = store.search("Query 2", query_embedding=[0.1, 0.9])
    assert len(res2) == 2
    assert res2[0]["id"] == id2
    assert res2[0]["score"] > 0.8

    store.close()


def test_bm25_search_fallback(temp_db):
    store = ProjectMemoryStore(db_path=temp_db)

    id1 = store.add("The quick brown fox jumps over the lazy dog")
    id2 = store.add("Artificial intelligence and deep machine learning models")

    # Search for fox (should match id1)
    res1 = store.search("fox jumps")
    assert len(res1) == 2
    assert res1[0]["id"] == id1
    assert res1[0]["score"] > 0.0

    # Search for learning (should match id2)
    res2 = store.search("deep learning models")
    assert len(res2) == 2
    assert res2[0]["id"] == id2
    assert res2[0]["score"] > 0.0

    store.close()


def test_embed_fn_callback(temp_db):
    def fake_embed(text):
        if "apple" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]

    store = ProjectMemoryStore(db_path=temp_db, embed_fn=fake_embed)

    id1 = store.add("an apple tree")
    id2 = store.add("a banana boat")

    r1 = store.get(id1)
    r2 = store.get(id2)

    assert r1["embedding"] == [1.0, 0.0]
    assert r2["embedding"] == [0.0, 1.0]

    # Semantic search automatically triggers embed_fn
    res = store.search("apple query")
    assert res[0]["id"] == id1

    store.close()
