"""Tests for the compressing reverse proxy (tokenslim.proxy).

Spins the real proxy in a thread in front of a fake in-process upstream that
records everything it receives, then drives it with plain HTTP clients.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from click.testing import CliRunner

from tokenslim import __version__
from tokenslim.ccr import find_markers
from tokenslim.cli import main
from tokenslim.config import Config, load_config
from tokenslim.proxy import RETRIEVE_PATH, make_proxy_server


class _FakeUpstreamHandler(BaseHTTPRequestHandler):
    """Records each request on the server; replies per-path."""

    # HTTP/1.0: closing the connection delimits streamed bodies, so the SSE
    # branch below needs no chunked framing of its own.
    protocol_version = "HTTP/1.0"

    def log_message(self, format, *args):
        pass

    def _record(self, body: bytes) -> None:
        self.server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "headers": dict(self.headers),
                "body": body,
            }
        )

    def do_GET(self):
        self._record(b"")
        data = b'{"object": "list", "data": []}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        self._record(body)

        if self.path == "/v1/error":
            data = json.dumps({"error": {"message": "bad key"}}).encode("utf-8")
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path == "/v1/truncate":
            # Promise 1000 bytes, deliver a handful, then drop the connection —
            # the classic "upstream cut short (length/network)" failure.
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b'{"partial": true}')
            self.wfile.flush()
            return

        try:
            wants_stream = bool(json.loads(body).get("stream"))
        except Exception:
            wants_stream = False

        if wants_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()  # no Content-Length: proxy must re-frame chunked
            self.wfile.write(b'data: {"choice": 1}\n\n')
            self.wfile.flush()
            # Hold the stream open until the client has SEEN the first event;
            # a proxy that buffers the whole body would deadlock here (and the
            # client's socket timeout turns that into a test failure).
            self.server.gate.wait(timeout=10)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        data = json.dumps({"id": "chatcmpl-1", "object": "chat.completion"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def proxy_env():
    """Yield ``(proxy_url, upstream_server)`` with both servers running."""
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstreamHandler)
    upstream.requests = []
    upstream.gate = threading.Event()
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}"

    proxy = make_proxy_server(port=0, upstream=upstream_url, config=Config(), host="127.0.0.1")
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    proxy_url = f"http://127.0.0.1:{proxy.server_address[1]}"

    yield proxy_url, upstream

    proxy.shutdown()
    proxy.server_close()
    upstream.shutdown()
    upstream.server_close()


def _post(url: str, body: bytes, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    all_headers = {"Content-Type": "application/json"}
    all_headers.update(headers or {})
    req = urllib.request.Request(url, data=body, headers=all_headers, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, resp.read()


def _get_json(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, json.loads(resp.read())


@contextlib.contextmanager
def _proxy_stack(config: Config):
    """Yield ``(proxy_url, upstream)`` for a proxy+fake-upstream on ``config``."""
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstreamHandler)
    upstream.requests = []
    upstream.gate = threading.Event()
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}"

    proxy = make_proxy_server(port=0, upstream=upstream_url, config=config, host="127.0.0.1")
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    proxy_url = f"http://127.0.0.1:{proxy.server_address[1]}"
    try:
        yield proxy_url, upstream
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()


def _big_json_message() -> str:
    rows = [{"id": i, "name": f"row-{i}", "status": "ok", "score": i * 0.5} for i in range(60)]
    return json.dumps(rows)


def test_health_endpoint(proxy_env):
    proxy_url, upstream = proxy_env
    with urllib.request.urlopen(f"{proxy_url}/health", timeout=10) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read())
    assert payload == {"status": "ok", "version": __version__}
    assert upstream.requests == [], "health must be answered locally, not forwarded"


def test_chat_completions_messages_compressed(proxy_env):
    proxy_url, upstream = proxy_env
    original = _big_json_message()
    body = json.dumps(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": original}]}
    ).encode("utf-8")

    status, response = _post(f"{proxy_url}/v1/chat/completions", body)

    assert status == 200
    assert json.loads(response)["id"] == "chatcmpl-1"
    assert len(upstream.requests) == 1
    received = json.loads(upstream.requests[0]["body"])
    assert upstream.requests[0]["path"] == "/v1/chat/completions"
    assert received["model"] == "gpt-4o", "non-message fields must be preserved"
    content = received["messages"][0]["content"]
    assert received["messages"][0]["role"] == "user"
    assert len(content) < len(original), "messages must arrive compressed at the upstream"
    assert "__tokenslim_ccr__" in content, "compressed JSON should carry the CCR sentinel"


def test_authorization_header_forwarded(proxy_env):
    proxy_url, upstream = proxy_env
    body = json.dumps({"messages": [{"role": "user", "content": "hello"}]}).encode("utf-8")

    status, _ = _post(
        f"{proxy_url}/v1/chat/completions",
        body,
        headers={"Authorization": "Bearer sk-test-123", "X-Custom": "kept"},
    )

    assert status == 200
    headers = upstream.requests[0]["headers"]
    assert headers.get("Authorization") == "Bearer sk-test-123"
    assert headers.get("X-Custom") == "kept"
    assert headers.get("Content-Type") == "application/json"


def test_malformed_json_passes_through(proxy_env):
    proxy_url, upstream = proxy_env
    body = b'{"messages": [oops this is not json'

    status, response = _post(f"{proxy_url}/v1/chat/completions", body)

    assert status == 200, "malformed input must never break the request"
    assert json.loads(response)["id"] == "chatcmpl-1"
    assert upstream.requests[0]["body"] == body, "malformed body must pass through unchanged"


def test_body_without_messages_passes_through(proxy_env):
    proxy_url, upstream = proxy_env
    body = json.dumps({"input": "hello", "model": "gpt-4o"}).encode("utf-8")

    status, _ = _post(f"{proxy_url}/v1/chat/completions", body)

    assert status == 200
    assert upstream.requests[0]["body"] == body


def test_responses_endpoint_is_pure_passthrough(proxy_env):
    proxy_url, upstream = proxy_env
    body = json.dumps(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": _big_json_message()}]}
    ).encode("utf-8")

    status, _ = _post(f"{proxy_url}/v1/responses", body)

    assert status == 200
    assert upstream.requests[0]["path"] == "/v1/responses"
    assert upstream.requests[0]["body"] == body, "/v1/responses must not be compressed"


def test_get_requests_forwarded(proxy_env):
    proxy_url, upstream = proxy_env
    with urllib.request.urlopen(f"{proxy_url}/v1/models", timeout=10) as resp:
        assert resp.status == 200
        assert json.loads(resp.read())["object"] == "list"
    assert upstream.requests[0] == {
        "method": "GET",
        "path": "/v1/models",
        "headers": upstream.requests[0]["headers"],
        "body": b"",
    }


def test_upstream_error_relayed(proxy_env):
    proxy_url, _ = proxy_env
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(f"{proxy_url}/v1/error", b"{}")
    assert excinfo.value.code == 401
    assert b"bad key" in excinfo.value.read()


def test_sse_stream_chunks_flow_through_in_order(proxy_env):
    proxy_url, upstream = proxy_env
    host, port = proxy_url.removeprefix("http://").split(":")
    body = json.dumps({"stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode(
        "utf-8"
    )

    conn = http.client.HTTPConnection(host, int(port), timeout=10)
    try:
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type") == "text/event-stream"

        # The upstream holds the stream open (gate) until we have read the
        # first event — succeeding here proves the proxy does not buffer.
        first = b""
        while b'data: {"choice": 1}\n\n' not in first:
            piece = resp.read1(64)
            assert piece, "stream ended before the first SSE event arrived"
            first += piece
        upstream.gate.set()

        rest = b""
        while piece := resp.read1(4096):
            rest += piece
    finally:
        upstream.gate.set()  # never leave the upstream handler blocked
        conn.close()

    full = first + rest
    assert full.index(b'data: {"choice": 1}') < full.index(b"data: [DONE]")


def test_proxy_config_env_knobs():
    cfg = load_config(
        env={"TOKENSLIM_PROXY_PORT": "9999", "TOKENSLIM_UPSTREAM": "http://example.com"}
    )
    assert cfg.proxy_port == 9999
    assert cfg.upstream == "http://example.com"
    defaults = Config()
    assert defaults.proxy_port == 8787
    assert defaults.upstream == "https://api.openai.com"


def test_cli_proxy_help_lists_flags():
    runner = CliRunner()
    result = runner.invoke(main, ["proxy", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output
    assert "--upstream" in result.output
    assert "reverse proxy" in result.output


def test_proxy_upgrades_memory_backend_to_persistent_sqlite(tmp_path):
    from tokenslim.store import SQLiteCCRStore

    cfg = Config(ccr_backend="memory", ccr_path=str(tmp_path / "ccr.sqlite3"))
    with _proxy_stack(cfg) as (_proxy_url, _upstream):
        pass
    # The default in-memory store is upgraded to a persistent SQLite file so
    # reversibility survives the request that fills it (issue #119).
    proxy = make_proxy_server(port=0, upstream="http://x", config=cfg, host="127.0.0.1")
    try:
        assert isinstance(proxy.store, SQLiteCCRStore)
    finally:
        proxy.server_close()


def test_proxy_ccr_reversible_across_requests_and_restart(tmp_path):
    ccr_path = str(tmp_path / "ccr.sqlite3")
    cfg = Config(ccr_backend="sqlite", ccr_path=ccr_path)
    original = _big_json_message()
    body = json.dumps(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": original}]}
    ).encode("utf-8")

    with _proxy_stack(cfg) as (proxy_url, upstream):
        status, _ = _post(f"{proxy_url}/v1/chat/completions", body)
        assert status == 200
        sent = json.loads(upstream.requests[0]["body"])["messages"][0]["content"]
        markers = find_markers(sent)
        assert markers, "the compressed request must carry a retrievable CCR marker"
        hash_ = markers[0].hash

        # After the request has fully completed, the marker still expands: the
        # store is not the ephemeral per-request dict that broke reversibility.
        status, payload = _get_json(f"{proxy_url}{RETRIEVE_PATH}?hash={hash_}")
        assert status == 200
        assert payload["hash"] == hash_
        assert "row-30" in payload["original"], "dropped middle rows must be recoverable"

    # After a restart (fresh proxy on the same SQLite file), the marker the
    # first process minted is still expandable — within TTL, across restarts.
    with _proxy_stack(cfg) as (proxy_url, _upstream):
        marker_token = f"<<ccr:{hash_} {markers[0].count} {markers[0].reason}>>"
        url = f"{proxy_url}{RETRIEVE_PATH}?marker={urllib.parse.quote(marker_token)}"
        status, payload = _get_json(url)
        assert status == 200
        assert "row-30" in payload["original"]


def test_retrieve_endpoint_reports_misses(tmp_path):
    cfg = Config(ccr_backend="sqlite", ccr_path=str(tmp_path / "ccr.sqlite3"))
    with _proxy_stack(cfg) as (proxy_url, _upstream):
        with pytest.raises(urllib.error.HTTPError) as miss:
            _get_json(f"{proxy_url}{RETRIEVE_PATH}?hash=deadbeef")
        assert miss.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as bad:
            _get_json(f"{proxy_url}{RETRIEVE_PATH}")
        assert bad.value.code == 400


def test_truncated_upstream_not_relayed_as_complete_200(proxy_env):
    proxy_url, _upstream = proxy_env
    # The upstream promises 1000 bytes but delivers ~17 then drops. The proxy
    # must surface that as a truncated read, never a silent, "complete" 200.
    with pytest.raises(http.client.IncompleteRead):
        _post(f"{proxy_url}/v1/truncate", b"{}")


def test_cli_retrieve_expands_marker(tmp_path, monkeypatch):
    from tokenslim.ccr import make_marker
    from tokenslim.store import SQLiteCCRStore

    path = str(tmp_path / "ccr.sqlite3")
    store = SQLiteCCRStore(path)
    hash_ = store.put("the full original payload")
    store.close()

    monkeypatch.setenv("TOKENSLIM_CCR_BACKEND", "sqlite")
    monkeypatch.setenv("TOKENSLIM_CCR_PATH", path)
    runner = CliRunner()

    hit = runner.invoke(main, ["retrieve", make_marker(hash_, 3)])
    assert hit.exit_code == 0
    assert "the full original payload" in hit.output

    miss = runner.invoke(main, ["retrieve", "deadbeef"])
    assert miss.exit_code == 1
    assert "No CCR record" in miss.output
