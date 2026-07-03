"""Opt-in local session capture for offline analysis.

Records agent-session events (compression runs, tool calls, outcomes) as
JSON Lines on the local filesystem so ``tokenslim learn`` (#42) can mine them
later. Capture is OFF by default and strictly local: nothing is uploaded, and
raw message content is only written when ``Config.capture_content`` is
explicitly enabled.

On-disk format (the contract consumed by :func:`read_sessions` / #42):

- ``Config.capture_path`` (default ``~/.tokenslim/sessions``) is a directory
  holding one ``<session_id>.jsonl`` file per :class:`SessionCapture`.
- Each line is one JSON event object::

      {"ts": 1712345678.9, "session_id": "<uuid4 hex>", "kind": "compress",
       "payload": {...}}

  ``ts`` is ``time.time()`` at record time; ``kind`` is an open vocabulary —
  tokenslim itself emits ``compress``, ``tool_call`` and ``outcome``.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

from .config import Config, load_config

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["SessionCapture", "get_capture", "read_sessions"]

# Default directory for captured session files (used when capture_path is None).
DEFAULT_CAPTURE_DIR = os.path.join("~", ".tokenslim", "sessions")


def _resolve_dir(config: Config) -> str:
    """Return the absolute capture directory for ``config``."""
    return os.path.expanduser(config.capture_path or DEFAULT_CAPTURE_DIR)


class SessionCapture:
    """Append session events to a local JSONL file.

    One instance is one session: a fresh ``session_id`` (uuid4 hex) is minted
    at construction and every event is appended to
    ``<capture_dir>/<session_id>.jsonl``. :meth:`record` never raises —
    capture failures (unwritable disk, bad payloads) must not break the host
    agent.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config if config is not None else load_config()
        self.session_id = uuid.uuid4().hex
        self.directory = _resolve_dir(self.config)
        self.path = os.path.join(self.directory, f"{self.session_id}.jsonl")
        self._lock = threading.Lock()

    def record(self, kind: str, payload: dict[str, Any]) -> None:
        """Append one ``{ts, session_id, kind, payload}`` line. Never raises."""
        event = {
            "ts": time.time(),
            "session_id": self.session_id,
            "kind": kind,
            "payload": payload,
        }
        with contextlib.suppress(Exception):
            # default=str keeps odd payload values (Enums, Paths, ...) loggable.
            line = json.dumps(event, ensure_ascii=False, default=str)
            with self._lock:
                os.makedirs(self.directory, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")

    def record_tool_call(
        self,
        tool: str,
        arguments: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        """Record a ``tool_call`` event (helper for agent integrations)."""
        payload: dict[str, Any] = {"tool": tool, "arguments": arguments or {}}
        payload.update(extra)
        self.record("tool_call", payload)

    def record_outcome(
        self,
        status: str,
        detail: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        """Record an ``outcome`` event (helper for agent integrations)."""
        payload: dict[str, Any] = {"status": status}
        if detail is not None:
            payload["detail"] = detail
        payload.update(extra)
        self.record("outcome", payload)


_capture_lock = threading.Lock()
_capture: SessionCapture | None = None


def get_capture(config: Config | None = None) -> SessionCapture | None:
    """Return the process-wide :class:`SessionCapture`, or ``None`` when off.

    Singleton-ish: one shared instance per capture directory, so all callers
    in a process append to the same session file. A config pointing at a
    different ``capture_path`` swaps in a fresh instance (fresh session_id).
    """
    cfg = config if config is not None else load_config()
    if not cfg.capture:
        return None
    global _capture
    with _capture_lock:
        if _capture is None or _capture.directory != _resolve_dir(cfg):
            _capture = SessionCapture(cfg)
        return _capture


def read_sessions(path: str | os.PathLike[str] | None = None) -> Iterator[dict[str, Any]]:
    """Iterate captured events from ``path`` (a session dir or one JSONL file).

    Yields event dicts (``{ts, session_id, kind, payload}``) in file order,
    walking ``*.jsonl`` files in sorted-name order when ``path`` is a
    directory. ``path`` defaults to the default capture directory. Missing
    paths yield nothing and malformed lines are skipped — the reader is
    deliberately tolerant so ``tokenslim learn`` (#42) survives partially
    written or corrupted capture files.
    """
    target = os.path.expanduser(path if path is not None else DEFAULT_CAPTURE_DIR)
    try:
        if os.path.isdir(target):
            files = sorted(
                os.path.join(target, name) for name in os.listdir(target) if name.endswith(".jsonl")
            )
        elif os.path.isfile(target):
            files = [target]
        else:
            return
    except OSError:
        return
    for file_path in files:
        try:
            # errors="replace" keeps invalid UTF-8 (e.g. a record truncated
            # mid-multibyte character by a crash) from raising: only the
            # corrupt line then fails json.loads and is skipped below, so the
            # rest of the file — and the walk over later files — survives.
            with open(file_path, encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event
