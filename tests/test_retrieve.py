import json

from tokenslim import compress
from tokenslim.ccr import find_markers
from tokenslim.config import Config
from tokenslim.retrieve import CCRContext, retrieve
from tokenslim.store import InMemoryCCRStore


def _crushed_message(n=120):
    rows = [{"id": i, "status": "ok"} for i in range(n)]
    rows[60] = {"id": 60, "status": "error", "detail": "payment declined"}
    return [{"role": "tool", "tool_call_id": "t1", "content": json.dumps(rows)}], rows


# --- direct retrieve ------------------------------------------------------


def test_retrieve_returns_exact_dropped_rows():
    messages, rows = _crushed_message()
    out, stats = compress(messages, options=Config(min_bytes=0))
    content = out[0]["content"]

    # The dropped rows are gone from the visible output...
    marker = find_markers(content)[0]
    assert marker.count > 0
    assert content.count('"id":30') == 0  # a middle row is not in the output

    # ...but retrievable verbatim from the store.
    restored = json.loads(retrieve(marker.hash, store=stats.store))
    assert len(restored) == marker.count
    # The restored block is exactly the contiguous dropped middle.
    assert restored[0]["id"] == 5  # first dropped (head keep = 5)
    assert all(r["status"] != "error" for r in restored)  # error was kept, not dropped


def test_retrieve_unknown_hash_is_none():
    store = InMemoryCCRStore()
    assert retrieve("nope", store=store) is None


def test_retrieve_via_config_rebuilds_backend(tmp_path):
    cfg = Config(min_bytes=0, ccr_backend="sqlite", ccr_path=str(tmp_path / "r.db"))
    messages, _ = _crushed_message()
    out, _ = compress(messages, options=cfg)
    h = find_markers(out[0]["content"])[0].hash
    # No explicit store — retrieve rebuilds the sqlite backend from config.
    assert retrieve(h, config=cfg) is not None


# --- context tracker ------------------------------------------------------


def test_context_tracks_live_markers_and_scopes_retrieval():
    messages, _ = _crushed_message()
    out, stats = compress(messages, options=Config(min_bytes=0))

    ctx = CCRContext(store=stats.store)
    n = ctx.track(out)
    assert n == 1
    live = ctx.live_hashes()
    assert len(live) == 1

    h = next(iter(live))
    assert ctx.retrieve(h) is not None
    # A hash that was never seen in the conversation is refused even though it
    # may exist in the store.
    assert ctx.retrieve("never-seen-hash") is None


def test_context_handles_anthropic_blocks():
    payload = json.dumps([{"id": i} for i in range(80)])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "data follows"},
                {"type": "tool_result", "tool_use_id": "t1", "content": payload},
            ],
        }
    ]
    out, stats = compress(messages, options=Config(min_bytes=0))
    ctx = CCRContext(store=stats.store)
    assert ctx.track(out) == 1


def test_ccr_disabled_no_store_no_markers():
    messages, _ = _crushed_message()
    out, stats = compress(messages, options=Config(min_bytes=0, ccr=False))
    # CCR off: no store is created and no markers/sentinels are emitted.
    assert stats.store is None
    assert "<<ccr:" not in out[0]["content"]
    assert "__tokenslim_ccr__" not in out[0]["content"]
