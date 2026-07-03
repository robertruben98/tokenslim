"""LangChain integration — chat models, memory payloads and retriever documents.

Duck-typed like :mod:`tokenslim.integrations`: this module NEVER imports
langchain. A ``BaseMessage`` is anything with ``.type`` ("human"/"ai"/
"system"/"tool") and ``.content`` (str or block list); a ``Document`` is
anything with ``.page_content``. The optional ``langchain`` extra only pins
``langchain-core`` for users who want the real classes — nothing here needs it.

Usage::

    from tokenslim.integrations_langchain import wrap_chat_model

    wrap_chat_model(chat_model)          # invoke/ainvoke/stream now compress
    docs = compress_documents(retriever_docs, query="user question")
"""

from __future__ import annotations

import contextlib
import copy
import functools
import inspect
from typing import TYPE_CHECKING, Any

from .compress import compress

if TYPE_CHECKING:
    from .config import Config

__all__ = ["wrap_chat_model", "compress_documents", "compress_messages"]

# Marker attribute set on patched methods so double-wrapping is a no-op.
_WRAPPED_ATTR = "__tokenslim_wrapped__"

# LangChain BaseMessage.type -> OpenAI chat role.
_TYPE_TO_ROLE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "function",
    "developer": "developer",
}


def _is_lc_message(obj: Any) -> bool:
    """True when ``obj`` quacks like a LangChain BaseMessage."""
    return hasattr(obj, "type") and hasattr(obj, "content")


def _lc_to_openai(messages: Any) -> list[dict[str, Any]]:
    """Convert LangChain-style message objects to OpenAI chat dicts.

    ``content`` is carried over as-is (str or block list) — :func:`compress`
    deep-copies its input, so the original objects are never mutated through
    it. Unknown ``.type`` values fall back to a ``.role`` attribute
    (ChatMessage) and finally to ``"user"``.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        mtype = str(getattr(msg, "type", "") or "")
        role = _TYPE_TO_ROLE.get(mtype) or str(getattr(msg, "role", "") or "") or "user"
        out.append({"role": role, "content": getattr(msg, "content", "")})
    return out


def _clone_with(obj: Any, attr: str, value: Any) -> Any:
    """Copy ``obj`` with ``attr`` set to ``value`` — never mutates ``obj``.

    Prefers pydantic-v2 ``model_copy(update=...)`` (real LangChain objects),
    falling back to ``copy.copy`` + setattr for plain classes.
    """
    model_copy = getattr(obj, "model_copy", None)
    if callable(model_copy):
        with contextlib.suppress(Exception):
            return model_copy(update={attr: value})
    clone = copy.copy(obj)
    setattr(clone, attr, value)
    return clone


def _restore(originals: Any, compressed: list[dict[str, Any]]) -> list[Any]:
    """Write compressed content back onto copies of the original messages.

    A message whose copy cannot be produced is passed through unchanged (and
    unmutated) rather than raising.
    """
    out: list[Any] = []
    for orig, comp in zip(originals, compressed):
        restored = orig
        with contextlib.suppress(Exception):
            restored = _clone_with(orig, "content", comp.get("content"))
        out.append(restored)
    return out


def compress_messages(messages: list[Any], config: Config | None = None) -> list[Any]:
    """Compress a list of LangChain messages, returning new message objects.

    The inputs are never mutated; each output message is a copy of the
    original with its ``content`` replaced by the compressed version. Useful
    directly on memory payloads (chat histories) and used internally by
    :func:`wrap_chat_model`.
    """
    dicts = _lc_to_openai(messages)
    compressed, _stats = compress(dicts, config)
    return _restore(messages, compressed)


def _compress_chat_input(value: Any, config: Config | None) -> Any:
    """Compress a chat-model input when it is a list of BaseMessage-likes.

    Anything else (str prompts, PromptValues, dict payloads, empty or mixed
    lists) passes through untouched; failures silently fall back to the
    original input.
    """
    if not isinstance(value, list) or not value:
        return value
    if not all(_is_lc_message(m) for m in value):
        return value
    with contextlib.suppress(Exception):
        return compress_messages(value, config)
    return value


def _rewrite_call_args(
    args: tuple[Any, ...], kwargs: dict[str, Any], config: Config | None
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Compress the chat input whether passed positionally or as ``input=``."""
    if args:
        return (_compress_chat_input(args[0], config), *args[1:]), kwargs
    if "input" in kwargs:
        kwargs = dict(kwargs)
        kwargs["input"] = _compress_chat_input(kwargs["input"], config)
    return args, kwargs


def _wrap_method(original: Any, config: Config | None) -> Any:
    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = _rewrite_call_args(args, kwargs, config)
            return await original(*args, **kwargs)

        wrapper: Any = async_wrapper
    else:

        @functools.wraps(original)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = _rewrite_call_args(args, kwargs, config)
            return original(*args, **kwargs)

        wrapper = sync_wrapper

    setattr(wrapper, _WRAPPED_ATTR, True)
    return wrapper


def wrap_chat_model(model: Any, config: Config | None = None) -> Any:
    """Patch a LangChain chat model in place to compress message inputs.

    Duck-typed: ``invoke``/``ainvoke`` (and ``stream`` when present) are
    replaced on the instance so list-of-BaseMessage inputs are compressed
    before delegating to the original method. Non-list/unknown inputs pass
    through untouched, and any failure silently falls back to the original
    input. Wrapping is idempotent — already-patched methods are left alone.
    Returns ``model``.
    """
    for name in ("invoke", "ainvoke", "stream"):
        original = getattr(model, name, None)
        if not callable(original) or getattr(original, _WRAPPED_ATTR, False):
            continue
        wrapper = _wrap_method(original, config)
        try:
            setattr(model, name, wrapper)
        except Exception:
            # Pydantic models reject unknown attributes in __setattr__; bypass
            # it — an instance attribute legitimately shadows the method here.
            with contextlib.suppress(Exception):
                object.__setattr__(model, name, wrapper)
    return model


def compress_documents(
    docs: list[Any], config: Config | None = None, query: str | None = None
) -> list[Any]:
    """Compress LangChain documents (duck-typed on ``.page_content``).

    ``query`` feeds :attr:`Config.query` for relevance-aware compression —
    rows matching the query survive aggressive crushing. Returns new document
    objects (``model_copy``/``copy.copy``); the inputs are never mutated.
    Items without a text ``page_content``, or whose compression fails, are
    returned as-is. Works for retriever results and memory payloads alike.
    """
    out: list[Any] = []
    for doc in docs:
        new_doc = doc
        with contextlib.suppress(Exception):
            text = doc.page_content
            if isinstance(text, str) and text:
                compressed, _stats = compress(
                    [{"role": "user", "content": text}], config, query=query
                )
                candidate = compressed[0].get("content")
                new_text = candidate if isinstance(candidate, str) else text
                new_doc = _clone_with(doc, "page_content", new_text)
        out.append(new_doc)
    return out
