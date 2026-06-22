"""Integration: the M1 compressors are wired into the router/compress() path."""

import json

from tokenslim import compress
from tokenslim.config import Config
from tokenslim.detector import ContentType
from tokenslim.router import ContentRouter, build_registry


def test_build_registry_uses_real_compressors():
    reg = build_registry(Config())
    assert reg[ContentType.JSON][0] == "smartcrusher"
    assert reg[ContentType.LOG][0] == "log-compressor"
    assert reg[ContentType.SEARCH][0] == "search-compressor"


def test_router_routes_json_to_smartcrusher():
    data = [{"id": i, "status": "ok"} for i in range(100)]
    router = ContentRouter(config=Config(min_bytes=0))
    result = router.route(json.dumps(data))
    assert result.compressor == "smartcrusher"
    assert result.changed is True
    assert "__tokenslim_ccr__" in result.text


def test_router_routes_log_to_log_compressor():
    log = "\n".join([f"INFO step {i}" for i in range(40)] + ["ERROR boom", "1 failed in 1s"])
    router = ContentRouter(config=Config(min_bytes=0))
    result = router.route(log)
    assert result.content_type is ContentType.LOG
    assert result.compressor == "log-compressor"
    assert result.changed is True


def test_compress_end_to_end_crushes_tool_output():
    payload = json.dumps([{"id": i, "status": "ok"} for i in range(200)])
    messages = [{"role": "tool", "tool_call_id": "t1", "content": payload}]
    out, stats = compress(messages, options=Config(min_bytes=0))
    assert stats.saved_tokens > 0
    assert stats.ratio > 0.5  # big homogeneous array -> large win
    assert "__tokenslim_ccr__" in out[0]["content"]


def test_compress_preserves_errors_end_to_end():
    rows = [{"id": i, "status": "ok"} for i in range(200)]
    rows[120] = {"id": 120, "status": "error", "detail": "payment declined"}
    messages = [{"role": "tool", "tool_call_id": "t1", "content": json.dumps(rows)}]
    out, _ = compress(messages, options=Config(min_bytes=0))
    assert "payment declined" in out[0]["content"]
