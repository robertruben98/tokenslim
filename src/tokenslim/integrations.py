"""Integrations with third-party LLM frameworks (OpenAI, Anthropic, LiteLLM)."""

from __future__ import annotations

import contextlib
import functools
import inspect
from typing import Any

from .compress import compress

__all__ = ["with_tokenslim", "TokenSlimLiteLLMCallback"]


def _wrap_create(original_create: Any) -> Any:
    if inspect.iscoroutinefunction(original_create):

        @functools.wraps(original_create)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            messages = kwargs.get("messages")
            if messages is not None:
                compressed, _ = compress(messages)
                kwargs["messages"] = compressed
            return await original_create(*args, **kwargs)

        return async_wrapper
    else:

        @functools.wraps(original_create)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            messages = kwargs.get("messages")
            if messages is not None:
                compressed, _ = compress(messages)
                kwargs["messages"] = compressed
            return original_create(*args, **kwargs)

        return sync_wrapper


def with_tokenslim(client: Any) -> Any:
    """Wrap an OpenAI or Anthropic client to automatically compress messages before sending.

    Works transparently for both sync and async clients.
    """
    if (
        hasattr(client, "chat")
        and hasattr(client.chat, "completions")
        and hasattr(client.chat.completions, "create")
    ):
        client.chat.completions.create = _wrap_create(client.chat.completions.create)

    # Anthropic client
    if hasattr(client, "messages") and hasattr(client.messages, "create"):
        client.messages.create = _wrap_create(client.messages.create)

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
