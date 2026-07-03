import json
from typing import Any

import pytest

from tokenslim.config import Config
from tokenslim.integrations_langchain import (
    _lc_to_openai,
    compress_documents,
    compress_messages,
    wrap_chat_model,
)
from tokenslim.tokenizer import count_tokens


class FakeMessage:
    """Duck-typed LangChain BaseMessage: just ``.type`` + ``.content``."""

    def __init__(self, type_: str, content: Any) -> None:
        self.type = type_
        self.content = content


class FakePydanticMessage(FakeMessage):
    """Adds a pydantic-v2-style ``model_copy`` to exercise that clone path."""

    def __init__(self, type_: str, content: Any) -> None:
        super().__init__(type_, content)
        self.copied_via_model_copy = False

    def model_copy(self, update: dict[str, Any] | None = None) -> "FakePydanticMessage":
        clone = FakePydanticMessage(self.type, self.content)
        for key, value in (update or {}).items():
            setattr(clone, key, value)
        clone.copied_via_model_copy = True
        return clone


class FakeChatModel:
    """Duck-typed chat model recording what invoke/ainvoke/stream received."""

    def __init__(self) -> None:
        self.last_input: Any = None

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> str:
        self.last_input = input
        return "ok"

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> str:
        self.last_input = input
        return "ok"

    def stream(self, input: Any, config: Any = None, **kwargs: Any):
        self.last_input = input
        yield "chunk"


class FakeDocument:
    """Duck-typed LangChain Document: ``.page_content`` + ``.metadata``."""

    def __init__(self, page_content: Any, metadata: dict[str, Any] | None = None) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}


def _big_json() -> str:
    return json.dumps([float(x) for x in range(100)])


def _big_records(marker_index: int = 42, n: int = 60) -> str:
    """JSON records where only one row (deep in the middle) mentions 'zanzibar'."""
    records = []
    for i in range(n):
        note = f"routine maintenance entry number {i}"
        if i == marker_index:
            note = f"zanzibar checkpoint reached during maintenance sweep {i}"
        records.append({"id": i, "note": note})
    return json.dumps(records)


# --- message conversion ---------------------------------------------------


def test_lc_to_openai_role_mapping() -> None:
    msgs = [
        FakeMessage("system", "be terse"),
        FakeMessage("human", "hi"),
        FakeMessage("ai", "hello"),
        FakeMessage("tool", "result"),
        FakeMessage("weird-type", "fallback"),
    ]
    dicts = _lc_to_openai(msgs)
    assert [d["role"] for d in dicts] == ["system", "user", "assistant", "tool", "user"]
    assert dicts[1]["content"] == "hi"


def test_compress_messages_returns_copies() -> None:
    original_text = _big_json()
    msg = FakeMessage("human", original_text)
    out = compress_messages([msg], Config(min_bytes=0))

    assert out[0] is not msg
    assert "__tokenslim_ccr__" in out[0].content
    assert len(out[0].content) < len(original_text)
    assert msg.content == original_text, "caller's message must not be mutated"


def test_compress_messages_prefers_model_copy() -> None:
    original_text = _big_json()
    msg = FakePydanticMessage("human", original_text)
    out = compress_messages([msg], Config(min_bytes=0))

    assert out[0].copied_via_model_copy
    assert "__tokenslim_ccr__" in out[0].content
    assert msg.content == original_text
    assert not msg.copied_via_model_copy


def test_compress_messages_block_list_content() -> None:
    blocks = [{"type": "text", "text": _big_json()}]
    msg = FakeMessage("human", blocks)
    out = compress_messages([msg], Config(min_bytes=0))

    assert "__tokenslim_ccr__" in out[0].content[0]["text"]
    assert msg.content[0]["text"] == _big_json(), "original blocks must stay untouched"


# --- wrap_chat_model --------------------------------------------------------


def test_wrap_invoke_compresses_and_does_not_mutate() -> None:
    model = FakeChatModel()
    wrapped = wrap_chat_model(model, Config(min_bytes=0))
    assert wrapped is model

    original_text = _big_json()
    msg = FakeMessage("human", original_text)
    assert wrapped.invoke([msg]) == "ok"

    sent = model.last_input
    assert sent[0] is not msg, "model must receive copies, not the caller's objects"
    assert len(sent[0].content) < len(original_text)
    assert "__tokenslim_ccr__" in sent[0].content
    assert msg.content == original_text, "caller's message must not be mutated"


@pytest.mark.asyncio
async def test_wrap_ainvoke_compresses() -> None:
    model = FakeChatModel()
    wrap_chat_model(model, Config(min_bytes=0))

    original_text = _big_json()
    msg = FakeMessage("human", original_text)
    assert await model.ainvoke([msg]) == "ok"

    assert "__tokenslim_ccr__" in model.last_input[0].content
    assert msg.content == original_text


def test_wrap_stream_compresses() -> None:
    model = FakeChatModel()
    wrap_chat_model(model, Config(min_bytes=0))

    msg = FakeMessage("human", _big_json())
    assert list(model.stream([msg])) == ["chunk"]
    assert "__tokenslim_ccr__" in model.last_input[0].content


def test_wrap_input_keyword_compresses() -> None:
    model = FakeChatModel()
    wrap_chat_model(model, Config(min_bytes=0))

    msg = FakeMessage("human", _big_json())
    model.invoke(input=[msg])
    assert "__tokenslim_ccr__" in model.last_input[0].content


def test_double_wrap_is_idempotent() -> None:
    model = FakeChatModel()
    wrap_chat_model(model, Config(min_bytes=0))
    first_invoke = model.invoke
    first_ainvoke = model.ainvoke
    first_stream = model.stream

    wrap_chat_model(model, Config(min_bytes=0))
    assert model.invoke is first_invoke
    assert model.ainvoke is first_ainvoke
    assert model.stream is first_stream

    # Still functional (and compressed exactly once) after the no-op re-wrap.
    msg = FakeMessage("human", _big_json())
    model.invoke([msg])
    assert "__tokenslim_ccr__" in model.last_input[0].content


def test_wrap_passthrough_on_odd_inputs() -> None:
    model = FakeChatModel()
    wrap_chat_model(model, Config(min_bytes=0))

    model.invoke("plain string prompt")
    assert model.last_input == "plain string prompt"

    dict_payload = [{"role": "user", "content": "dicts are not BaseMessages"}]
    model.invoke(dict_payload)
    assert model.last_input is dict_payload, "unknown list shapes must pass through untouched"

    model.invoke([])
    assert model.last_input == []

    mixed = [FakeMessage("human", "hi"), "not a message"]
    model.invoke(mixed)
    assert model.last_input is mixed


def test_wrap_object_without_methods_is_noop() -> None:
    class Empty:
        pass

    obj = Empty()
    assert wrap_chat_model(obj) is obj


# --- compress_documents ------------------------------------------------------


def test_compress_documents_reduces_tokens() -> None:
    text = _big_records()
    doc = FakeDocument(text, metadata={"source": "unit"})
    out = compress_documents([doc], Config(min_bytes=0))

    assert len(out) == 1
    assert out[0] is not doc, "must return new document objects"
    assert count_tokens(out[0].page_content, None) < count_tokens(text, None)
    assert "__tokenslim_ccr__" in out[0].page_content
    assert doc.page_content == text, "original document must not be mutated"
    assert out[0].metadata == {"source": "unit"}


def test_compress_documents_respects_query() -> None:
    text = _big_records()
    without = compress_documents([FakeDocument(text)], Config(min_bytes=0))
    with_query = compress_documents([FakeDocument(text)], Config(min_bytes=0), query="zanzibar")

    assert "zanzibar" not in without[0].page_content, "middle row should be crushed away"
    assert "zanzibar" in with_query[0].page_content, "query-relevant row must survive"


def test_compress_documents_passthrough_odd_docs() -> None:
    class Weird:
        pass

    no_text = FakeDocument(None)
    weird = Weird()
    out = compress_documents([no_text, weird], Config(min_bytes=0))

    assert out[0] is no_text
    assert out[1] is weird
