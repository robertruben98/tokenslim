import json
from typing import Any

import pytest

from tokenslim.integrations import TokenSlimLiteLLMCallback, with_tokenslim


class MockOpenAICompletions:
    def create(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return {"model": model, "messages": messages, "kwargs": kwargs}


class MockOpenAIClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": MockOpenAICompletions()})()


class MockAnthropicMessages:
    def create(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return {"model": model, "messages": messages, "kwargs": kwargs}


class MockAnthropicClient:
    def __init__(self) -> None:
        self.messages = MockAnthropicMessages()


class MockAsyncCompletions:
    async def create(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return {"model": model, "messages": messages, "kwargs": kwargs}


class MockAsyncOpenAIClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": MockAsyncCompletions()})()


def test_wrap_openai() -> None:
    client = MockOpenAIClient()
    wrapped = with_tokenslim(client)

    long_numbers = [float(x) for x in range(100)]
    msg = {"role": "user", "content": json.dumps(long_numbers)}

    res = wrapped.chat.completions.create(model="gpt-4", messages=[msg])
    compressed_content = res["messages"][0]["content"]
    assert len(compressed_content) < len(json.dumps(long_numbers))
    assert "__tokenslim_ccr__" in compressed_content


@pytest.mark.asyncio
async def test_wrap_async_openai() -> None:
    client = MockAsyncOpenAIClient()
    wrapped = with_tokenslim(client)

    long_numbers = [float(x) for x in range(100)]
    msg = {"role": "user", "content": json.dumps(long_numbers)}

    res = await wrapped.chat.completions.create(model="gpt-4", messages=[msg])
    compressed_content = res["messages"][0]["content"]
    assert len(compressed_content) < len(json.dumps(long_numbers))
    assert "__tokenslim_ccr__" in compressed_content


def test_litellm_callback() -> None:
    callback = TokenSlimLiteLLMCallback()
    long_numbers = [float(x) for x in range(100)]
    messages = [{"role": "user", "content": json.dumps(long_numbers)}]

    kwargs: dict[str, Any] = {"messages": messages}
    callback.pre_call_hook(
        user_api_key_dict=None,
        cache_dict=None,
        messages=None,
        model="gpt-4",
        call_type="completion",
        literal_params=None,
        kwargs=kwargs,
    )

    compressed_content = kwargs["messages"][0]["content"]
    assert len(compressed_content) < len(json.dumps(long_numbers))
    assert "__tokenslim_ccr__" in compressed_content
