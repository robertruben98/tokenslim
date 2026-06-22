from tokenslim.formats import (
    MessageFormat,
    anthropic_to_openai,
    detect_format,
    openai_to_anthropic,
)


def test_detect_openai_by_system_role():
    msgs = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
    assert detect_format(msgs) is MessageFormat.OPENAI


def test_detect_openai_by_tool_role():
    msgs = [{"role": "tool", "tool_call_id": "x", "content": "result"}]
    assert detect_format(msgs) is MessageFormat.OPENAI


def test_detect_anthropic_by_tool_result_block():
    msgs = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "out"}],
        }
    ]
    assert detect_format(msgs) is MessageFormat.ANTHROPIC


def test_detect_empty_is_unknown():
    assert detect_format([]) is MessageFormat.UNKNOWN


def test_openai_to_anthropic_drops_system_and_converts_tool():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "abc", "content": "tool out"},
    ]
    out = openai_to_anthropic(msgs)
    assert all(m["role"] != "system" for m in out)
    tool_msg = out[-1]
    assert tool_msg["role"] == "user"
    block = tool_msg["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "abc"
    assert block["content"] == "tool out"


def test_anthropic_to_openai_lifts_tool_results():
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "out"}],
        },
    ]
    out = anthropic_to_openai(msgs)
    tool_msgs = [m for m in out if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "t1"
    assert tool_msgs[0]["content"] == "out"


def test_roundtrip_preserves_user_text():
    msgs = [{"role": "user", "content": "hello world"}]
    back = anthropic_to_openai(openai_to_anthropic(msgs))
    assert back == msgs
