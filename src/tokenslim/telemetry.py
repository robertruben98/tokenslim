"""Anonymous telemetry for usage and savings analysis.

Privacy model — **config wins, the environment can only *additionally* disable**:

* ``Config.telemetry`` is the master switch and takes precedence. It defaults to
  ``False``, so telemetry is OFF unless a caller explicitly opts in with
  ``compress(config=Config(telemetry=True))``.
* ``TOKENSLIM_TELEMETRY`` can only turn telemetry *off* on top of that
  (``off`` / ``false`` / ``0`` / ``no``). It can never turn it *on* against a
  ``Config(telemetry=False)``.

Resulting matrix::

    Config.telemetry   TOKENSLIM_TELEMETRY   sends?
    False              (any / unset)         no
    True               unset / on / 1 / yes  yes
    True               off / false / 0 / no  no

Events are queued on a single bounded background worker; a full queue drops
events silently. ``compress()`` is never blocked and never has an exception
raised into it.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import urllib.request

from . import __version__

__all__ = ["send_telemetry_event"]

TELEMETRY_ENDPOINT = "https://telemetry.tokenslim.dev/v1/event"

# Bounded queue: if the network can't keep up, events are dropped rather than
# growing memory without limit or blocking the compression path.
_MAX_QUEUE = 256

_queue: queue.Queue[dict | None] | None = None
_worker: threading.Thread | None = None
_lock = threading.Lock()


def _env_opted_out() -> bool:
    """True when TOKENSLIM_TELEMETRY explicitly opts out."""
    return os.environ.get("TOKENSLIM_TELEMETRY", "").strip().lower() in {
        "off",
        "false",
        "0",
        "no",
    }


def _post(payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        TELEMETRY_ENDPOINT,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": f"TokenSlim/{__version__}",
        },
        method="POST",
    )
    # Short timeout to avoid a slow endpoint tying up the worker for long.
    with urllib.request.urlopen(req, timeout=1.0) as resp:
        resp.read()


def _worker_loop(q: queue.Queue[dict | None]) -> None:
    """Drain the queue forever, POSTing each event; ``None`` is a stop sentinel."""
    while True:
        payload = q.get()
        try:
            if payload is None:
                return
            _post(payload)
        except Exception:
            # Best-effort: never let a network/parse error escape the worker.
            pass
        finally:
            q.task_done()


def _ensure_worker() -> queue.Queue[dict | None]:
    """Lazily create the shared queue and its single daemon worker."""
    global _queue, _worker
    with _lock:
        if _queue is None:
            _queue = queue.Queue(maxsize=_MAX_QUEUE)
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(
                target=_worker_loop,
                args=(_queue,),
                daemon=True,
                name="tokenslim-telemetry",
            )
            _worker.start()
        return _queue


def send_telemetry_event(
    orig_tokens: int,
    new_tokens: int,
    model: str | None = None,
    content_types: list[str] | None = None,
    *,
    enabled: bool = True,
) -> None:
    """Queue an anonymous usage event on the shared background worker.

    ``enabled`` carries the caller's ``Config.telemetry`` decision and takes
    precedence: when ``False`` nothing is ever queued or sent. When ``True`` the
    event is still suppressed if ``TOKENSLIM_TELEMETRY`` opts out in the
    environment. The call is non-blocking; if the bounded queue is full the
    event is dropped silently.
    """
    if not enabled:
        return
    if _env_opted_out():
        return

    payload = {
        "version": __version__,
        "orig_tokens": orig_tokens,
        "new_tokens": new_tokens,
        "saved_tokens": orig_tokens - new_tokens,
        "ratio": new_tokens / orig_tokens if orig_tokens > 0 else 1.0,
        "model": model,
        "content_types": content_types or [],
    }

    q = _ensure_worker()
    # Telemetry is best-effort: drop rather than block compress() or grow
    # memory without bound when the queue is full.
    with contextlib.suppress(queue.Full):
        q.put_nowait(payload)


def _drain_for_tests(timeout: float = 2.0) -> bool:
    """Block until the worker has processed every queued event (test-only).

    Returns ``True`` if the queue drained within ``timeout`` seconds.
    """
    import time

    q = _queue
    if q is None:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if q.unfinished_tasks == 0:
            return True
        time.sleep(0.01)
    return q.unfinished_tasks == 0
