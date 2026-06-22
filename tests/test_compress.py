import json

from tokenslim import compress
from tokenslim.config import Config
from tokenslim.detector import ContentType


def _big_json_message():
    payload = json.dumps(
        {"items": [{"id": i, "name": "item", "tags": ["a", "b"]} for i in range(60)]},
        indent=2,
    )
    return {"role": "tool", "tool_call_id": "t1", "content": payload}


def test_compress_returns_messages_and_stats():
    messages = [_big_json_message()]
    out, stats = compress(messages, options=Config(min_bytes=0))
    assert isinstance(out, list)
    assert stats.orig_tokens > 0
    assert stats.new_tokens <= stats.orig_tokens


def test_compress_reduces_tokens_on_pretty_json():
    messages = [_big_json_message()]
    out, stats = compress(messages, options=Config(min_bytes=0))
    assert stats.new_tokens < stats.orig_tokens
    assert 0.0 < stats.ratio <= 1.0
    assert stats.saved_tokens > 0
    # Output is still valid JSON (SmartCrusher elides the array middle and
    # leaves a CCR sentinel rather than corrupting the structure).
    parsed = json.loads(out[0]["content"])
    assert isinstance(parsed, dict)
    assert "__tokenslim_ccr__" in out[0]["content"]


def test_compress_does_not_mutate_input():
    messages = [_big_json_message()]
    before = json.loads(messages[0]["content"])
    compress(messages, options=Config(min_bytes=0))
    assert json.loads(messages[0]["content"]) == before


def test_compress_records_per_block_detail():
    messages = [_big_json_message()]
    _, stats = compress(messages, options=Config(min_bytes=0))
    assert len(stats.blocks) == 1
    block = stats.blocks[0]
    assert block.message_index == 0
    assert block.content_type is ContentType.JSON
    assert block.compressor == "smartcrusher"
    assert block.changed is True


def test_compress_skips_below_threshold():
    messages = [{"role": "user", "content": '{"a":1}'}]
    out, stats = compress(messages, options=Config(min_bytes=10_000))
    assert out[0]["content"] == messages[0]["content"]
    assert stats.blocks[0].skipped is True
    assert stats.ratio == 0.0


def test_compress_disabled_is_passthrough():
    messages = [_big_json_message()]
    out, stats = compress(messages, options=Config(enabled=False))
    assert out[0]["content"] == messages[0]["content"]
    assert stats.orig_tokens == stats.new_tokens


def test_compress_handles_anthropic_content_blocks():
    payload = json.dumps({"rows": list(range(200))}, indent=2)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "here is the data"},
                {"type": "tool_result", "tool_use_id": "t1", "content": payload},
            ],
        }
    ]
    out, stats = compress(messages, options=Config(min_bytes=0))
    result_block = out[0]["content"][1]
    # The tool_result's big number array is crushed; output stays valid JSON.
    parsed = json.loads(result_block["content"])
    assert isinstance(parsed, dict) and "rows" in parsed
    assert stats.saved_tokens > 0


def test_compress_empty_messages():
    out, stats = compress([])
    assert out == []
    assert stats.orig_tokens == 0
    assert stats.ratio == 0.0
