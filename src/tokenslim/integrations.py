"""Integrations with third-party LLM frameworks (OpenAI, Anthropic, LiteLLM)."""

from __future__ import annotations

import contextlib
import functools
import inspect
from typing import TYPE_CHECKING, Any

from .compress import compress

if TYPE_CHECKING:
    from .config import Config

__all__ = ["with_tokenslim", "TokenSlimLiteLLMCallback"]


def _maybe_reduce_output(messages: Any, config: Config | None, model: Any) -> Any:
    """Apply output reduction (issue #63) when configured; failures stay silent."""
    with contextlib.suppress(Exception):
        from .config import load_config
        from .outputs import apply_output_reduction

        cfg = config if config is not None else load_config()
        if cfg.output_reduction and cfg.output_reduction != "off":
            reduced, _ = apply_output_reduction(
                messages,
                level=cfg.output_reduction,
                model=model if isinstance(model, str) else cfg.model,
            )
            return reduced
    return messages


def _maybe_reduce_output_anthropic(kwargs: dict[str, Any], config: Config | None) -> None:
    """Anthropic dialect: brevity instructions go into the top-level ``system``
    kwarg — the Messages API rejects ``role="system"`` inside ``messages``.

    Mutates ``kwargs`` in place (never the caller's ``system`` list); failures
    stay silent. Unrecognized ``system`` shapes skip reduction entirely.
    """
    with contextlib.suppress(Exception):
        from .config import load_config
        from .outputs import OUTPUT_REDUCTION_LEVELS

        cfg = config if config is not None else load_config()
        instructions = OUTPUT_REDUCTION_LEVELS.get(cfg.output_reduction, "")
        if not instructions:
            return
        system = kwargs.get("system")
        if system is None:
            kwargs["system"] = instructions
        elif isinstance(system, str):
            kwargs["system"] = f"{system}\n\n{instructions}" if system else instructions
        elif isinstance(system, list):
            kwargs["system"] = [*system, {"type": "text", "text": instructions}]


def _wrap_create(
    original_create: Any, config: Config | None = None, anthropic: bool = False
) -> Any:
    def _rewrite_kwargs(kwargs: dict[str, Any]) -> None:
        messages = kwargs.get("messages")
        if messages is None:
            return
        compressed, _ = compress(messages, config)
        if anthropic:
            kwargs["messages"] = compressed
            _maybe_reduce_output_anthropic(kwargs, config)
        else:
            kwargs["messages"] = _maybe_reduce_output(compressed, config, kwargs.get("model"))

    if inspect.iscoroutinefunction(original_create):

        @functools.wraps(original_create)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            _rewrite_kwargs(kwargs)
            return await original_create(*args, **kwargs)

        return async_wrapper
    else:

        @functools.wraps(original_create)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            _rewrite_kwargs(kwargs)
            return original_create(*args, **kwargs)

        return sync_wrapper


def with_tokenslim(client: Any, config: Config | None = None) -> Any:
    """Wrap an OpenAI or Anthropic client to automatically compress messages before sending.

    Works transparently for both sync and async clients. When ``config`` (or
    the ``TOKENSLIM_OUTPUT_REDUCTION`` env var) sets ``output_reduction`` to a
    level other than "off", output-brevity instructions are also appended
    after compression (see :func:`tokenslim.outputs.apply_output_reduction`).
    """
    if (
        hasattr(client, "chat")
        and hasattr(client.chat, "completions")
        and hasattr(client.chat.completions, "create")
    ):
        client.chat.completions.create = _wrap_create(client.chat.completions.create, config)

    # Anthropic client — reduction targets the top-level ``system`` kwarg
    if hasattr(client, "messages") and hasattr(client.messages, "create"):
        client.messages.create = _wrap_create(client.messages.create, config, anthropic=True)

    return client


class TokenSlimLiteLLMCallback:
    """LiteLLM callback class for transparent context compression."""

    def pre_call_hook(
        self,
        user_api_key_dict: dict[str, Any] | None,
        cache_dict: dict[str, Any] | None,
        messages: list[dict[str, Any]] | None,
        model: str | None,
        call_type: str | None,
        literal_params: dict[str, Any] | None,
        kwargs: dict[str, Any] | None,
    ) -> None:
        if kwargs and "messages" in kwargs:
            compressed, _ = compress(kwargs["messages"])
            kwargs["messages"] = compressed
        elif messages:
            compressed, _ = compress(messages)
            with contextlib.suppress(Exception):
                messages[:] = compressed

    async def async_pre_call_hook(
        self,
        user_api_key_dict: dict[str, Any] | None,
        cache_dict: dict[str, Any] | None,
        messages: list[dict[str, Any]] | None,
        model: str | None,
        call_type: str | None,
        literal_params: dict[str, Any] | None,
        kwargs: dict[str, Any] | None,
    ) -> None:
        self.pre_call_hook(
            user_api_key_dict,
            cache_dict,
            messages,
            model,
            call_type,
            literal_params,
            kwargs,
        )
