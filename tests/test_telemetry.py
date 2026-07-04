"""Telemetry respects Config(telemetry=...) and uses a single bounded worker.

Regression tests for issue #115: telemetry used to ignore ``Config.telemetry``
(it only checked the env var) and spawned one thread per event.
"""

from __future__ import annotations

import http.server
import threading

import pytest

from tokenslim import telemetry


class _CountingHandler(http.server.BaseHTTPRequestHandler):
    """Counts POSTs on the handler subclass so each server is isolated."""

    count = 0

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        type(self).count += 1
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:  # silence test noise
        pass


def _fake_server(monkeypatch):
    """Start a throwaway HTTP server, point telemetry at it, return the handler."""
    handler = type("Handler", (_CountingHandler,), {"count": 0})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1/event"
    monkeypatch.setattr(telemetry, "TELEMETRY_ENDPOINT", url)
    return server, handler


def test_config_disabled_sends_nothing(monkeypatch):
    """Config(telemetry=False) => 0 requests, no matter the env."""
    server, handler = _fake_server(monkeypatch)
    monkeypatch.delenv("TOKENSLIM_TELEMETRY", raising=False)
    try:
        for _ in range(5):
            telemetry.send_telemetry_event(100, 50, enabled=False)
        telemetry._drain_for_tests(timeout=1.0)
        assert handler.count == 0
    finally:
        server.shutdown()


def test_config_enabled_sends(monkeypatch):
    """enabled=True with no env opt-out => one request per event."""
    server, handler = _fake_server(monkeypatch)
    monkeypatch.delenv("TOKENSLIM_TELEMETRY", raising=False)
    try:
        n = 3
        for _ in range(n):
            telemetry.send_telemetry_event(100, 50, model="gpt-4o", enabled=True)
        assert telemetry._drain_for_tests(timeout=3.0)
        assert handler.count == n
    finally:
        server.shutdown()


def test_env_opt_out_overrides_enabled(monkeypatch):
    """Env can *additionally* disable even when the caller opted in."""
    server, handler = _fake_server(monkeypatch)
    monkeypatch.setenv("TOKENSLIM_TELEMETRY", "off")
    try:
        telemetry.send_telemetry_event(100, 50, enabled=True)
        telemetry._drain_for_tests(timeout=1.0)
        assert handler.count == 0
    finally:
        server.shutdown()


def test_env_cannot_enable_against_config(monkeypatch):
    """Env=on cannot turn telemetry on against Config(telemetry=False)."""
    server, handler = _fake_server(monkeypatch)
    monkeypatch.setenv("TOKENSLIM_TELEMETRY", "on")
    try:
        telemetry.send_telemetry_event(100, 50, enabled=False)
        telemetry._drain_for_tests(timeout=1.0)
        assert handler.count == 0
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    ("enabled", "env", "expected"),
    [
        (False, None, 0),
        (False, "on", 0),
        (False, "off", 0),
        (True, None, 1),
        (True, "on", 1),
        (True, "1", 1),
        (True, "yes", 1),
        (True, "off", 0),
        (True, "false", 0),
        (True, "0", 0),
        (True, "no", 0),
    ],
)
def test_env_config_matrix(monkeypatch, enabled, env, expected):
    server, handler = _fake_server(monkeypatch)
    if env is None:
        monkeypatch.delenv("TOKENSLIM_TELEMETRY", raising=False)
    else:
        monkeypatch.setenv("TOKENSLIM_TELEMETRY", env)
    try:
        telemetry.send_telemetry_event(100, 50, enabled=enabled)
        telemetry._drain_for_tests(timeout=3.0)
        assert handler.count == expected
    finally:
        server.shutdown()


def test_single_worker_not_thread_per_event(monkeypatch):
    """M events must not create M threads — at most one telemetry worker."""
    server, handler = _fake_server(monkeypatch)
    monkeypatch.delenv("TOKENSLIM_TELEMETRY", raising=False)
    try:
        for _ in range(20):
            telemetry.send_telemetry_event(100, 50, enabled=True)
        telemetry._drain_for_tests(timeout=3.0)
        workers = [t for t in threading.enumerate() if t.name == "tokenslim-telemetry"]
        assert len(workers) <= 1
        assert handler.count == 20
    finally:
        server.shutdown()
