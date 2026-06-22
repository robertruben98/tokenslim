"""Tests for SharedContext compressed inter-agent handoffs (issue #40)."""

from __future__ import annotations

import time

import pytest

from tokenslim.memory import SharedContext


def _big_json_blob() -> str:
    # A JSON-ish payload large enough that the core pipeline compresses it.
    items = ",".join(f'{{"i": {i}, "v": "row-{i}"}}' for i in range(60))
    return '{"rows": [' + items + "]}"


def test_put_get_compressed_and_full_roundtrip() -> None:
    ctx = SharedContext()
    original = _big_json_blob()
    handle = ctx.put("handoff-1", original)

    # Default get returns the (possibly) compressed form.
    compressed = ctx.get("handoff-1")
    assert compressed is not None

    # full=True reconstructs the exact original that was put in.
    assert ctx.get("handoff-1", full=True) == original

    assert handle.key == "handoff-1"
    assert handle.orig_tokens >= handle.new_tokens
    assert 0.0 <= handle.ratio <= 1.0


def test_get_unknown_key_returns_none() -> None:
    ctx = SharedContext()
    assert ctx.get("nope") is None
    assert ctx.get("nope", full=True) is None
    assert ctx.handle("nope") is None


def test_put_validation() -> None:
    ctx = SharedContext()
    with pytest.raises(ValueError):
        ctx.put("", "content")
    with pytest.raises(ValueError):
        ctx.put("k", "")


def test_dedup_superset_evicts_contained_entry() -> None:
    ctx = SharedContext()
    ctx.put("part", "alpha beta")
    # A superset put should drop the contained "part" entry.
    ctx.put("whole", "alpha beta gamma delta")
    keys = ctx.keys()
    assert "whole" in keys
    assert "part" not in keys


def test_ttl_expiry() -> None:
    ctx = SharedContext(ttl=0.05)
    ctx.put("ephemeral", "this should expire soon")
    assert ctx.get("ephemeral", full=True) == "this should expire soon"
    time.sleep(0.08)
    assert ctx.get("ephemeral") is None
    assert len(ctx) == 0


def test_size_cap_evicts_oldest() -> None:
    ctx = SharedContext(max_entries=2)
    ctx.put("a", "first entry content")
    ctx.put("b", "second entry content")
    ctx.put("c", "third entry content")
    keys = ctx.keys()
    assert len(keys) == 2
    assert "a" not in keys  # oldest evicted
    assert "c" in keys


def test_handle_without_payload() -> None:
    ctx = SharedContext()
    ctx.put("k", _big_json_blob())
    h = ctx.handle("k")
    assert h is not None
    assert h.key == "k"
    assert h.orig_tokens > 0


def test_delete() -> None:
    ctx = SharedContext()
    ctx.put("k", "some content")
    assert ctx.delete("k") is True
    assert ctx.delete("k") is False
    assert ctx.get("k") is None


def test_overwrite_same_key() -> None:
    ctx = SharedContext()
    ctx.put("k", "first version of the content")
    ctx.put("k", "second version of the content")
    assert ctx.get("k", full=True) == "second version of the content"
    assert len(ctx) == 1
