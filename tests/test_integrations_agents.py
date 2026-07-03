import json
import sys
import types
from typing import Any

import pytest

from tokenslim.integrations_agents import (
    TokenSlimStrandsHooks,
    compress_tool_output,
    tokenslim_agno_tool_hook,
    wrap_agno_model,
)

try:
    import strands  # noqa: F401

    HAS_STRANDS = True
except ImportError:
    HAS_STRANDS = False


def _big_json_text() -> str:
    return json.dumps([{"id": i, "value": float(i), "label": f"item-{i}"} for i in range(100)])


# ---------------------------------------------------------------------------
# compress_tool_output — lowest-common-denominator API
# ---------------------------------------------------------------------------


def test_compress_tool_output_big_json() -> None:
    original = _big_json_text()
    compressed = compress_tool_output(original)
    assert len(compressed) < len(original)
    assert "__tokenslim_ccr__" in compressed


def test_compress_tool_output_small_text_passthrough() -> None:
    assert compress_tool_output("hello") == "hello"


def test_compress_tool_output_never_raises() -> None:
    # None-ish / binary-ish / wrong-typed garbage must come back unchanged.
    garbage: list[Any] = [
        None,
        b"\x00\xff\xfe binary bytes",
        123,
        4.5,
        {"not": "a string"},
        ["nor", "this"],
        "",
        "\x00�\udc80 weird text" * 50,
        object(),
    ]
    for item in garbage:
        result = compress_tool_output(item)  # type: ignore[arg-type]
        if isinstance(item, str) and item:
            assert isinstance(result, str)
        else:
            assert result is item, f"non-compressible input mutated: {item!r}"


# ---------------------------------------------------------------------------
# Agno — tool hook
# ---------------------------------------------------------------------------


def test_agno_tool_hook_compresses_big_json_result() -> None:
    original = _big_json_text()
    calls: list[dict[str, Any]] = []

    def fake_tool(query: str) -> str:
        calls.append({"query": query})
        return original

    result = tokenslim_agno_tool_hook("fake_tool", fake_tool, {"query": "orders"})
    assert calls == [{"query": "orders"}]
    assert isinstance(result, str)
    assert len(result) < len(original)
    assert "__tokenslim_ccr__" in result


def test_agno_tool_hook_dict_result_compressed_when_smaller() -> None:
    payload = [{"id": i, "value": float(i), "label": f"item-{i}"} for i in range(100)]

    def fake_tool() -> list[dict[str, Any]]:
        return payload

    result = tokenslim_agno_tool_hook("fake_tool", fake_tool, {})
    assert isinstance(result, str)
    assert len(result) < len(json.dumps(payload))
    assert "__tokenslim_ccr__" in result


def test_agno_tool_hook_small_result_passthrough() -> None:
    def fake_tool() -> str:
        return "ok"

    assert tokenslim_agno_tool_hook("fake_tool", fake_tool, None) == "ok"


def test_agno_tool_hook_non_text_result_passthrough() -> None:
    marker = object()

    def fake_tool() -> Any:
        return marker

    assert tokenslim_agno_tool_hook("fake_tool", fake_tool, {}) is marker


def test_agno_tool_hook_tool_errors_propagate() -> None:
    def broken_tool() -> str:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        tokenslim_agno_tool_hook("broken_tool", broken_tool, {})


# ---------------------------------------------------------------------------
# Agno — model wrapping
# ---------------------------------------------------------------------------


class MockAgnoModel:
    def invoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return {"messages": messages, "kwargs": kwargs}

    async def ainvoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return {"messages": messages, "kwargs": kwargs}

    def response(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return {"messages": messages, "kwargs": kwargs}


def test_wrap_agno_model_compresses_messages_kwarg() -> None:
    model = wrap_agno_model(MockAgnoModel())
    original = _big_json_text()

    res = model.invoke(messages=[{"role": "user", "content": original}])
    content = res["messages"][0]["content"]
    assert len(content) < len(original)
    assert "__tokenslim_ccr__" in content

    res = model.response(messages=[{"role": "user", "content": original}])
    content = res["messages"][0]["content"]
    assert len(content) < len(original)


@pytest.mark.asyncio
async def test_wrap_agno_model_async() -> None:
    model = wrap_agno_model(MockAgnoModel())
    original = _big_json_text()

    res = await model.ainvoke(messages=[{"role": "user", "content": original}])
    content = res["messages"][0]["content"]
    assert len(content) < len(original)
    assert "__tokenslim_ccr__" in content


def test_wrap_agno_model_double_wrap_safe() -> None:
    model = wrap_agno_model(MockAgnoModel())
    first_invoke = model.invoke
    first_response = model.response

    wrap_agno_model(model)
    assert model.invoke is first_invoke
    assert model.response is first_response


def test_wrap_agno_model_positional_messages_pass_through() -> None:
    model = wrap_agno_model(MockAgnoModel())
    original = _big_json_text()

    # Positional messages bypass compression but must still work.
    res = model.invoke([{"role": "user", "content": original}])
    assert res["messages"][0]["content"] == original


def test_wrap_agno_model_object_messages_fall_back_silently() -> None:
    class FakeMessage:
        role = "user"
        content = "x" * 5000

    model = wrap_agno_model(MockAgnoModel())
    msg = FakeMessage()
    res = model.invoke(messages=[msg])
    assert res["messages"][0] is msg


def test_wrap_agno_model_without_matching_methods() -> None:
    class Bare:
        pass

    bare = Bare()
    assert wrap_agno_model(bare) is bare


# ---------------------------------------------------------------------------
# Strands — hook provider
# ---------------------------------------------------------------------------


class MockHookRegistry:
    def __init__(self) -> None:
        self.callbacks: list[tuple[Any, Any]] = []

    def add_callback(self, event_cls: Any, callback: Any) -> None:
        self.callbacks.append((event_cls, callback))


class MockEvent:
    def __init__(self, request: Any) -> None:
        self.request = request


@pytest.mark.skipif(HAS_STRANDS, reason="strands-agents is installed")
def test_strands_register_raises_clean_import_error() -> None:
    hooks = TokenSlimStrandsHooks()
    with pytest.raises(ImportError, match="strands-agents"):
        hooks.register(MockHookRegistry())


def test_strands_register_with_fake_strands_module(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real strands >=1.8 event name.
    class FakeBeforeModelCallEvent:
        pass

    fake_hooks = types.ModuleType("strands.hooks")
    fake_hooks.BeforeModelCallEvent = FakeBeforeModelCallEvent  # type: ignore[attr-defined]
    fake_strands = types.ModuleType("strands")
    fake_strands.hooks = fake_hooks  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "strands", fake_strands)
    monkeypatch.setitem(sys.modules, "strands.hooks", fake_hooks)

    registry = MockHookRegistry()
    TokenSlimStrandsHooks().register(registry)
    assert len(registry.callbacks) == 1
    event_cls, callback = registry.callbacks[0]
    assert event_cls is FakeBeforeModelCallEvent
    assert callable(callback)

    # register_hooks (strands' HookProvider protocol name) is the same entry.
    TokenSlimStrandsHooks().register_hooks(registry)
    assert len(registry.callbacks) == 2


def test_strands_register_with_legacy_event_name(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pre-1.8 strands releases exposed BeforeModelInvocationEvent instead.
    class FakeBeforeModelInvocationEvent:
        pass

    fake_hooks = types.ModuleType("strands.hooks")
    fake_hooks.BeforeModelInvocationEvent = FakeBeforeModelInvocationEvent  # type: ignore[attr-defined]
    fake_strands = types.ModuleType("strands")
    fake_strands.hooks = fake_hooks  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "strands", fake_strands)
    monkeypatch.setitem(sys.modules, "strands.hooks", fake_hooks)

    registry = MockHookRegistry()
    TokenSlimStrandsHooks().register(registry)
    assert len(registry.callbacks) == 1
    assert registry.callbacks[0][0] is FakeBeforeModelInvocationEvent


def test_strands_callback_compresses_request_list_in_place() -> None:
    hooks = TokenSlimStrandsHooks()
    original = _big_json_text()
    messages = [{"role": "user", "content": original}]
    event = MockEvent(request=messages)

    hooks._before_model_invocation(event)
    content = event.request[0]["content"]
    assert len(content) < len(original)
    assert "__tokenslim_ccr__" in content


def test_strands_callback_compresses_request_messages_attr() -> None:
    class FakeRequest:
        def __init__(self, messages: list[dict[str, Any]]) -> None:
            self.messages = messages

    hooks = TokenSlimStrandsHooks()
    original = _big_json_text()
    request = FakeRequest([{"role": "user", "content": original}])

    hooks._before_model_invocation(MockEvent(request=request))
    content = request.messages[0]["content"]
    assert len(content) < len(original)
    assert "__tokenslim_ccr__" in content


class MockAgent:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages


class MockModelCallEvent:
    """Shape of the real strands BeforeModelCallEvent: conversation at .agent.messages."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent


def test_strands_callback_compresses_agent_messages_strands_shape() -> None:
    # Real strands message shape: typeless {'text': str} content blocks.
    hooks = TokenSlimStrandsHooks()
    original = _big_json_text()
    messages: list[Any] = [{"role": "user", "content": [{"text": original}]}]

    hooks._before_model_invocation(MockModelCallEvent(MockAgent(messages)))
    text = messages[0]["content"][0]["text"]
    assert len(text) < len(original)
    assert "__tokenslim_ccr__" in text


def test_strands_callback_compresses_tool_result_blocks() -> None:
    original = _big_json_text()
    messages: list[Any] = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "t1",
                        "status": "success",
                        "content": [{"text": original}],
                    }
                }
            ],
        }
    ]

    TokenSlimStrandsHooks()._before_model_invocation(MockModelCallEvent(MockAgent(messages)))
    text = messages[0]["content"][0]["toolResult"]["content"][0]["text"]
    assert len(text) < len(original)
    assert "__tokenslim_ccr__" in text


def test_strands_callback_small_typeless_blocks_untouched() -> None:
    hooks = TokenSlimStrandsHooks()
    messages: list[Any] = [{"role": "user", "content": [{"text": "hi"}]}]
    hooks._before_model_invocation(MockModelCallEvent(MockAgent(messages)))
    assert messages[0]["content"][0]["text"] == "hi"


@pytest.mark.skipif(not HAS_STRANDS, reason="strands-agents not installed")
def test_strands_real_add_hook_registers_and_compresses() -> None:
    from strands.hooks import BeforeModelCallEvent, HookRegistry

    registry = HookRegistry()
    registry.add_hook(TokenSlimStrandsHooks())

    original = _big_json_text()
    agent = types.SimpleNamespace(messages=[{"role": "user", "content": [{"text": original}]}])
    event = BeforeModelCallEvent(agent=agent)
    assert registry.has_callbacks(), "add_hook registered nothing against real strands"

    registry.invoke_callbacks(event)
    text = agent.messages[0]["content"][0]["text"]
    assert len(text) < len(original)
    assert "__tokenslim_ccr__" in text


def test_strands_callback_never_raises_on_garbage_events() -> None:
    hooks = TokenSlimStrandsHooks()
    hooks._before_model_invocation(object())  # no .agent / .request at all
    hooks._before_model_invocation(MockEvent(request=None))
    hooks._before_model_invocation(MockEvent(request="not messages"))
    hooks._before_model_invocation(MockEvent(request=[]))
    hooks._before_model_invocation(MockEvent(request=[object()]))
    hooks._before_model_invocation(MockModelCallEvent(agent=None))
    hooks._before_model_invocation(MockModelCallEvent(MockAgent(messages=[object()])))
