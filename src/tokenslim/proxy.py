"""OpenAI-compatible compressing reverse proxy (``tokenslim proxy``).

Stdlib-only: :mod:`http.server` for the listener and :mod:`urllib.request`
for the upstream leg — no framework dependencies.

POST bodies to ``/v1/chat/completions`` have their ``messages`` array run
through :func:`tokenslim.compress` (config resolved from ``TOKENSLIM_*`` env
vars) before being forwarded to the upstream base URL
(``TOKENSLIM_UPSTREAM``, default ``https://api.openai.com``). Every other
path — including ``/v1/responses`` — is passed through untouched.
``Authorization`` and other request headers are preserved.

Responses are relayed in chunks as they arrive, so SSE streams
(``stream=true``) flow through without being buffered whole. Any error while
parsing or compressing a body falls back to forwarding the original bytes:
the proxy must never break a request.

``GET /health`` answers locally with ``{"status": "ok", "version": ...}``.
One line per proxied request is logged to stderr with the path, the
``orig -> new`` token counts, and the savings ratio.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .compress import compress
from .config import Config, load_config

__all__ = [
    "COMPRESS_PATHS",
    "DEFAULT_PORT",
    "DEFAULT_UPSTREAM",
    "TokenSlimProxyServer",
    "make_proxy_server",
    "run_proxy",
]

DEFAULT_PORT = 8787
DEFAULT_UPSTREAM = "https://api.openai.com"

# Paths whose JSON body's `messages` array is compressed before forwarding.
# Everything else (e.g. /v1/responses, /v1/embeddings) is pure passthrough.
COMPRESS_PATHS = frozenset({"/v1/chat/completions"})

_CHUNK_SIZE = 8192

# Hop-by-hop headers (RFC 9110 §7.6.1) plus headers the proxy recomputes.
_SKIP_REQUEST_HEADERS = frozenset(
    {
        "accept-encoding",  # forced to identity so relayed bytes stay 1:1
        "connection",
        "content-length",  # recomputed from the (possibly compressed) body
        "host",  # urllib sets the upstream host
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
_SKIP_RESPONSE_HEADERS = frozenset(
    {
        "connection",
        "content-length",  # re-sent explicitly when the upstream provides it
        "date",  # BaseHTTPRequestHandler emits its own
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "server",  # BaseHTTPRequestHandler emits its own
        "te",
        "trailers",
        "transfer-encoding",  # re-framed by the relay loop
        "upgrade",
    }
)


def _version() -> str:
    """Package version, imported lazily to avoid a circular package import."""
    from tokenslim import __version__

    return __version__


class TokenSlimProxyServer(ThreadingHTTPServer):
    """Threaded HTTP server carrying the upstream URL and resolved config."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        upstream: str = DEFAULT_UPSTREAM,
        config: Config | None = None,
    ) -> None:
        self.upstream = upstream.rstrip("/")
        self.config = config if config is not None else load_config()
        super().__init__(address, _ProxyHandler)


class _ProxyHandler(BaseHTTPRequestHandler):
    """One connection; ``self.server`` is the :class:`TokenSlimProxyServer`."""

    protocol_version = "HTTP/1.1"
    server: TokenSlimProxyServer

    # -- request entry points ------------------------------------------------

    def do_GET(self) -> None:
        try:
            if self.path.split("?", 1)[0] == "/health":
                self._send_json(200, {"status": "ok", "version": _version()})
                self._log_line(200, "health")
                return
            self._forward(b"", "passthrough")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        try:
            body = self._read_body()
            if self.path.split("?", 1)[0] in COMPRESS_PATHS:
                body, note = self._compress_body(body)
            else:
                note = "passthrough"
            self._forward(body, note)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # -- compression ----------------------------------------------------------

    def _compress_body(self, body: bytes) -> tuple[bytes, str]:
        """Compress ``messages`` inside a JSON body; pass through on any error."""
        try:
            payload = json.loads(body)
            messages = payload.get("messages") if isinstance(payload, dict) else None
            if not isinstance(messages, list):
                return body, "passthrough (no messages array)"
            new_messages, stats = compress(messages, options=self.server.config)
            payload["messages"] = new_messages
            note = f"{stats.orig_tokens} -> {stats.new_tokens} tokens ratio={stats.ratio:.2f}"
            return json.dumps(payload).encode("utf-8"), note
        except Exception:
            # Never break a request: forward the original bytes untouched.
            return body, "passthrough (compress error)"

    # -- upstream leg -----------------------------------------------------------

    def _forward(self, body: bytes, note: str) -> None:
        """Send the request upstream and relay the response back in chunks."""
        url = self.server.upstream + self.path
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in _SKIP_REQUEST_HEADERS
        }
        headers["Accept-Encoding"] = "identity"
        request = urllib.request.Request(
            url, data=body or None, headers=headers, method=self.command
        )
        try:
            with urllib.request.urlopen(request) as resp:
                status = self._relay(resp)
        except urllib.error.HTTPError as err:
            # 4xx/5xx from upstream still carry headers and a body — relay them.
            with err:
                status = self._relay(err)
        except Exception as err:
            status = 502
            self._send_json(
                502,
                {"error": {"message": f"tokenslim proxy: upstream error: {err}"}},
            )
        self._log_line(status, note)

    def _relay(self, resp: Any) -> int:
        """Stream an upstream response object back to the client, unbuffered."""
        status = int(getattr(resp, "status", None) or getattr(resp, "code", 502))
        resp_headers = getattr(resp, "headers", None) or {}
        self.send_response(status)
        for name, value in resp_headers.items():
            if name.lower() not in _SKIP_RESPONSE_HEADERS:
                self.send_header(name, value)

        # read1 returns as soon as bytes are available (and decodes chunked
        # transfer-encoding), which keeps SSE events flowing; plain read(n)
        # would block until n bytes arrive.
        read1 = getattr(resp, "read1", None)
        read = read1 if callable(read1) else resp.read

        content_length = resp_headers.get("Content-Length")
        if content_length is not None:
            self.send_header("Content-Length", content_length)
            self.end_headers()
            while chunk := read(_CHUNK_SIZE):
                self.wfile.write(chunk)
                self.wfile.flush()
        else:
            # No length (SSE / chunked upstream): re-frame as chunked and
            # flush each piece through as it arrives.
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while chunk := read(_CHUNK_SIZE):
                self.wfile.write(b"%x\r\n" % len(chunk))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        return status

    # -- plumbing -------------------------------------------------------------

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        return self.rfile.read(length) if length > 0 else b""

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _log_line(self, status: int, note: str) -> None:
        print(
            f"[tokenslim.proxy] {self.command} {self.path} {status} {note}",
            file=sys.stderr,
            flush=True,
        )

    def log_message(self, format: str, *args: Any) -> None:
        """Silence the default per-request log; ``_log_line`` replaces it."""


def make_proxy_server(
    port: int | None = None,
    upstream: str | None = None,
    config: Config | None = None,
    host: str = "",
) -> TokenSlimProxyServer:
    """Build (but do not start) a proxy server; ``port=0`` picks a free port."""
    cfg = config if config is not None else load_config()
    resolved_port = port if port is not None else cfg.proxy_port
    resolved_upstream = upstream if upstream else cfg.upstream
    return TokenSlimProxyServer((host, resolved_port), upstream=resolved_upstream, config=cfg)


def run_proxy(
    port: int | None = None,
    upstream: str | None = None,
    config: Config | None = None,
    host: str = "",
) -> None:
    """Run the compressing reverse proxy until interrupted (blocking)."""
    server = make_proxy_server(port=port, upstream=upstream, config=config, host=host)
    bound_host, bound_port = server.server_address[0], server.server_address[1]
    print(
        f"[tokenslim.proxy] listening on {bound_host or '0.0.0.0'}:{bound_port}"
        f" -> {server.upstream}",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
