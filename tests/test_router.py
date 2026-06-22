import json

from tokenslim.config import Config
from tokenslim.detector import ContentType
from tokenslim.router import ContentRouter, minify_json


def test_minify_json_strips_whitespace():
    src = '{\n  "a": 1,\n  "b": [1, 2, 3]\n}'
    out = minify_json(src, ContentType.JSON)
    assert out == '{"a":1,"b":[1,2,3]}'
    assert json.loads(out) == json.loads(src)


def test_minify_json_is_safe_on_garbage():
    assert minify_json("{not json", ContentType.JSON) == "{not json"


def test_router_skips_tiny_payloads():
    router = ContentRouter(config=Config(min_bytes=10_000))
    result = router.route('{"a": 1, "b": 2}')
    assert result.skipped is True
    assert result.changed is False


def test_router_compresses_large_json():
    big = json.dumps({"items": [{"id": i, "name": "x" * 5} for i in range(50)]}, indent=2)
    router = ContentRouter(config=Config(min_bytes=0))
    result = router.route(big)
    assert result.content_type is ContentType.JSON
    assert result.compressor == "json-minify"
    assert result.changed is True
    assert len(result.text) < len(big)


def test_router_respects_enabled_compressors_filter():
    big = json.dumps({"items": list(range(100))})
    router = ContentRouter(config=Config(min_bytes=0, enabled_compressors=("passthrough",)))
    result = router.route(big)
    # json-minify is filtered out, so the block is skipped unchanged.
    assert result.skipped is True
    assert result.text == big


def test_router_passthrough_for_text():
    text = "Just some prose that is definitely longer than the byte threshold here, ok."
    router = ContentRouter(config=Config(min_bytes=0))
    result = router.route(text)
    assert result.compressor == "passthrough"
    assert result.text == text
    assert result.changed is False
