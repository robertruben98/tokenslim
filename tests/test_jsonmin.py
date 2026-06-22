import json

from tokenslim.compressors.jsonmin import JsonMinifier, minify
from tokenslim.config import Config
from tokenslim.detector import ContentType


def test_minify_strips_whitespace_losslessly():
    src = '{\n  "a": 1,\n  "b": [1, 2, 3],\n  "c": {"d": true}\n}'
    out = minify(src)
    assert len(out) < len(src)
    # Byte-lossless on the *value*: re-parses to an equal object.
    assert json.loads(out) == json.loads(src)


def test_minify_keeps_original_when_not_shorter():
    already = '{"a":1}'
    assert minify(already) == already


def test_minify_safe_on_non_json():
    assert minify("not json at all") == "not json at all"
    assert minify("{broken") == "{broken"


def test_minify_preserves_unicode():
    src = '{"name": "café ☕"}'
    out = minify(src)
    assert json.loads(out)["name"] == "café ☕"


def test_compressor_callable():
    comp = JsonMinifier(Config())
    src = '{"x":   [1,   2]}'
    assert json.loads(comp(src, ContentType.JSON)) == {"x": [1, 2]}


def test_minify_handles_arrays_and_scalars():
    assert minify("[1,   2,   3]") == "[1,2,3]"
    assert minify("  42  ") == "42"
