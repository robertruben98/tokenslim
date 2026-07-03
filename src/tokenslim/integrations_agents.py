"""Integrations with agent frameworks (Agno, Strands).

Same zero-dependency philosophy as :mod:`tokenslim.integrations`: no framework
SDK is ever imported at module level — everything is duck-typed via ``hasattr``
/ ``getattr``, and the only lazy import (``strands``) happens inside
:meth:`TokenSlimStrandsHooks.register`.

The lowest-common-denominator API is :func:`compress_tool_output` — a
bulletproof ``str -> str`` helper any framework hook can call directly.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
from typing import TYPE_CHECKING, Any

from .compress import compress

if TYPE_CHECKING:
    from .config import Config

__all__ = [
    "compress_tool_output",
    "tokenslim_agno_tool_hook",
    "wrap_agno_model",
    "TokenSlimStrandsHooks",
]


def compress_tool_output(text: str, config: Config | None = None) -> str:
    """Compress a single tool-output string through the tokenslim pipeline.

    Framework-agnostic convenience: wraps ``text`` as a tool message, runs it
    through :func:`tokenslim.compress.compress`, and returns the compressed
    string.

    Bulletproof by contract — this function **never raises** and returns the
    input unchanged on any problem: non-string input, empty input, compression
    failure, or a result that is not strictly smaller than the original.
    """
    if not isinstance(text, str) or not text:
        return text
    with contextlib.suppress(Exception):
        compressed_messages, _ = compress([{"role": "tool", "content": text}], config)
        result = compressed_messages[0].get("content")
        if isinstance(result, str) and 0 < len(result) < len(text):
            return result
    return text


def _compress_result(result: Any, config: Config | None = None) -> Any:
    """Compress a tool result (string or JSON-like); return it unchanged otherwise.

    dict/list results are only replaced by their compressed JSON serialization
    when that actually saves bytes — otherwise the original object is returned
    untouched so downstream type expectations keep holding.
    """
    if isinstance(result, str):
        return compress_tool_output(result, config)
    if isinstance(result, (dict, list)):
        with contextlib.suppress(Exception):
            serialized = json.dumps(result, default=str)
            compressed = compress_tool_output(serialized, config)
            if len(compressed) < len(serialized):
                return compressed
    return result


def tokenslim_agno_tool_hook(
    function_name: str,
    function_call: Any,
    arguments: dict[str, Any] | None,
) -> Any:
    """Agno tool hook: execute the tool, then compress its output.

    Matches agno's tool-hook signature ``(function_name, function_call,
    arguments)`` where the hook is responsible for invoking the tool::

        agent = Agent(tools=[...], tool_hooks=[tokenslim_agno_tool_hook])

    Errors raised by the tool itself propagate (agno owns tool-error
    handling); only the compression step is failure-proof.
    """
    result = function_call(**(arguments or {}))
    return _compress_result(result)


def _wrap_messages_method(original: Any, config: Config | None = None) -> Any:
    """Wrap a sync/async method so its ``messages=`` kwarg is compressed.

    Compression failures (e.g. non-dict message objects) fall back silently
    to the original, untouched kwargs.
    """

    def _rewrite(kwargs: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            messages = kwargs.get("messages")
            if messages:
                compressed, _ = compress(messages, config)
                kwargs["messages"] = compressed

    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            _rewrite(kwargs)
            return await original(*args, **kwargs)

        wrapper = async_wrapper
    else:

        @functools.wraps(original)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            _rewrite(kwargs)
            return original(*args, **kwargs)

        wrapper = sync_wrapper

    wrapper.__tokenslim_wrapped__ = True  # type: ignore[attr-defined]
    return wrapper


def wrap_agno_model(model: Any, config: Config | None = None) -> Any:
    """Duck-patch an agno model so message payloads are compressed pre-call.

    Patches whichever of ``invoke`` / ``ainvoke`` / ``response`` /
    ``aresponse`` exist on the instance (``hasattr`` checks only — no agno
    import). Wrapping is idempotent: already-wrapped methods are skipped, so
    double-wrapping never double-compresses. Any error while patching or
    compressing falls back silently to the original behavior.
    """
    for attr in ("invoke", "ainvoke", "response", "aresponse"):
        with contextlib.suppress(Exception):
            method = getattr(model, attr, None)
            if method is None or not callable(method):
                continue
            if getattr(method, "__tokenslim_wrapped__", False):
                continue
            setattr(model, attr, _wrap_messages_method(method, config))
    return model


class TokenSlimStrandsHooks:
    """Strands (strands-agents) hook provider for transparent compression.

    Register on an agent's hook registry; before every model invocation the
    request messages are compressed in place::

        agent = Agent(model=..., hooks=[TokenSlimStrandsHooks()])

    ``strands`` is imported lazily inside :meth:`register` only — the class
    itself has zero dependencies.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config

    def register(self, registry: Any) -> None:
        """Duck-call ``registry.add_callback`` for the before-model event.

        Raises a clear :class:`ImportError` when the strands-agents package
        is absent. When strands is importable but this version does not
        expose the expected hook API, registration is silently skipped.
        """
        try:
            import strands.hooks as strands_hooks
        except ImportError as exc:
            raise ImportError(
                "TokenSlimStrandsHooks requires the strands-agents package. "
                "Install it with: pip install strands-agents"
            ) from exc

        # strands >=1.8 names the event BeforeModelCallEvent; earlier 1.x
        # releases exposed it as BeforeModelInvocationEvent. Accept either.
        event_cls = getattr(strands_hooks, "BeforeModelCallEvent", None) or getattr(
            strands_hooks, "BeforeModelInvocationEvent", None
        )
        if event_cls is None or not hasattr(registry, "add_callback"):
            return
        registry.add_callback(event_cls, self._before_model_invocation)

    # Strands' HookProvider protocol looks this method up by name.
    register_hooks = register

    def _before_model_invocation(self, event: Any) -> None:
        """Compress the pending conversation in place; failures stay silent.

        The real strands ``BeforeModelCallEvent`` carries the conversation at
        ``event.agent.messages``; request-shaped events (``event.request`` as
        a message list, or with a ``.messages`` attribute) are also handled.
        """
        with contextlib.suppress(Exception):
            messages = _event_messages(event)
            if not messages:
                return
            compressed, _ = compress(messages, self.config)
            messages[:] = compressed
            _compress_strands_blocks(messages, self.config)


def _event_messages(event: Any) -> list[Any] | None:
    """Locate the mutable message list on a before-model event, if any."""
    messages = getattr(getattr(event, "agent", None), "messages", None)
    if isinstance(messages, list):
        return messages
    request = getattr(event, "request", None)
    if isinstance(request, list):
        return request
    messages = getattr(request, "messages", None)
    if isinstance(messages, list):
        return messages
    return None


def _compress_strands_blocks(messages: list[Any], config: Config | None = None) -> None:
    """Compress strands-shaped (typeless) content blocks in place.

    Strands content blocks are keyed by kind (``{"text": ...}``,
    ``{"toolResult": ...}``) and carry no ``"type"`` field, so
    :func:`tokenslim.compress.compress` passes them through untouched. This
    pass routes their text through :func:`compress_tool_output` block by
    block; blocks with a ``"type"`` key were already handled by ``compress``.
    """
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or "type" in block:
                continue
            if isinstance(block.get("text"), str):
                block["text"] = compress_tool_output(block["text"], config)
            tool_result = block.get("toolResult")
            tr_content = tool_result.get("content") if isinstance(tool_result, dict) else None
            if isinstance(tr_content, list):
                for sub in tr_content:
                    if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                        sub["text"] = compress_tool_output(sub["text"], config)
