import time

import pytest

from tokenslim.config import Config
from tokenslim.store import (
    CCRStore,
    InMemoryCCRStore,
    SQLiteCCRStore,
    get_store,
)


def _backends(tmp_path):
    return [
        InMemoryCCRStore(),
        SQLiteCCRStore(str(tmp_path / "ccr.sqlite3")),
    ]


def test_protocol_runtime_check():
    assert isinstance(InMemoryCCRStore(), CCRStore)


@pytest.mark.parametrize("make", [InMemoryCCRStore, None])
def test_put_get_roundtrip(make, tmp_path):
    store = make() if make else SQLiteCCRStore(str(tmp_path / "x.db"))
    h = store.put("the original payload")
    assert isinstance(h, str) and len(h) >= 8
    assert store.get(h) == "the original payload"


def test_put_is_content_addressed_and_idempotent(tmp_path):
    for store in _backends(tmp_path):
        h1 = store.put("same bytes")
        h2 = store.put("same bytes")
        assert h1 == h2
        assert len(store) == 1


def test_get_unknown_hash_returns_none(tmp_path):
    for store in _backends(tmp_path):
        assert store.get("deadbeef") is None


def test_distinct_payloads_distinct_hashes(tmp_path):
    for store in _backends(tmp_path):
        assert store.put("a") != store.put("b")
        assert len(store) == 2


def test_inmemory_ttl_eviction():
    store = InMemoryCCRStore(ttl=0)
    h = store.put("ephemeral")
    time.sleep(0.01)
    assert store.get(h) is None


def test_sqlite_ttl_eviction(tmp_path):
    store = SQLiteCCRStore(str(tmp_path / "ttl.db"), ttl=0)
    h = store.put("ephemeral")
    time.sleep(0.01)
    assert store.get(h) is None


def test_sqlite_persists_across_instances(tmp_path):
    path = str(tmp_path / "persist.db")
    h = SQLiteCCRStore(path).put("durable")
    # A brand new instance pointed at the same file still finds it.
    assert SQLiteCCRStore(path).get(h) == "durable"


def test_get_store_factory(tmp_path):
    assert isinstance(get_store(Config(ccr_backend="memory")), InMemoryCCRStore)
    sqlite = get_store(Config(ccr_backend="sqlite", ccr_path=str(tmp_path / "f.db")))
    assert isinstance(sqlite, SQLiteCCRStore)


def test_get_store_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown CCR backend"):
        get_store(Config(ccr_backend="redis"))
