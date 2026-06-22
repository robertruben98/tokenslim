"""Retrieve API + context tracker.

The flip side of compression: given a CCR ``hash`` (from a marker the model
saw), return the original material that was dropped. :func:`retrieve` is the
direct lookup; :class:`CCRContext` scopes retrieval to a conversation by
tracking which markers are currently "live" in the messages, so a tool can only
fetch what the model was actually shown.
"""

from __future__ import annotations

from typing import Any

from .ccr import CCRMarker, find_markers
from .config import Config
from .store import CCRStore, get_store

__all__ = ["retrieve", "CCRContext"]


def retrieve(hash: str, store: CCRStore | None = None, config: Config | None = None) -> str | None:
    """Return the original material stored under ``hash``.

    Provide an explicit ``store`` (the one used during compression), or a
    ``config`` to rebuild the configured backend. Returns ``None`` if the hash
    is unknown or its record has expired.
    """
    if store is None:
        store = get_store(config or Config())
    return store.get(hash)


class CCRContext:
    """Tracks CCR markers live in a conversation and scopes retrieval to them.

    Feed it the (compressed) messages with :meth:`track`; it scans every text
    block for markers and remembers their hashes. :meth:`retrieve` then only
    returns originals for markers the model has actually seen — a guard against
    a tool fetching arbitrary store contents.
    """

    def __init__(self, store: CCRStore | None = None, config: Config | None = None) -> None:
        self._config = config or Config()
        self._store = store if store is not None else get_store(self._config)
        # hash -> marker metadata seen in the conversation.
        self._live: dict[str, CCRMarker] = {}

    @property
    def store(self) -> CCRStore:
        return self._store

    def track(self, messages: list[dict[str, Any]]) -> int:
        """Scan messages for CCR markers and register them as live.

        Returns the number of newly-registered markers.
        """
        before = len(self._live)
        for msg in messages:
            for text in _iter_text(msg.get("content")):
                for marker in find_markers(text):
                    self._live[marker.hash] = marker
        return len(self._live) - before

    def live_hashes(self) -> frozenset[str]:
        """Hashes of all markers currently tracked as live."""
        return frozenset(self._live)

    def retrieve(self, hash: str) -> str | None:
        """Return the original for ``hash`` only if it is a live marker."""
        if hash not in self._live:
            return None
        return self._store.get(hash)


def _iter_text(content: Any):
    """Yield every text string inside an OpenAI/Anthropic content value."""
    if isinstance(content, str):
        if content:
            yield content
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                yield block["text"]
            elif block.get("type") == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, str) and inner:
                    yield inner
            # Array sentinels live as JSON inside tool-output strings, which are
            # already covered by the string branches above.
