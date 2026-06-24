"""Anonymous telemetry for usage and savings analysis.

ON by default, can be disabled by setting the environment variable
TOKENSLIM_TELEMETRY=off or TOKENSLIM_TELEMETRY=false.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request

from . import __version__

__all__ = ["send_telemetry_event"]

TELEMETRY_ENDPOINT = "https://telemetry.tokenslim.dev/v1/event"


def send_telemetry_event(
    orig_tokens: int,
    new_tokens: int,
    model: str | None = None,
    content_types: list[str] | None = None,
) -> None:
    """Send anonymous usage telemetry asynchronously if not opted out."""
    # Check opt-out environment variables
    opt_out_env = os.environ.get("TOKENSLIM_TELEMETRY", "").lower()
    if opt_out_env in ("off", "false", "0", "no"):
        return

    # Document exactly what is sent in the telemetry event:
    # - version: Installed TokenSlim version
    # - orig_tokens: Number of original input tokens
    # - new_tokens: Number of compressed output tokens
    # - saved_tokens: Number of tokens saved by compression
    # - ratio: Ratio of compression savings
    # - model: (Optional) Model name specified for routing/pricing
    # - content_types: (Optional) Content types processed during this run
    payload = {
        "version": __version__,
        "orig_tokens": orig_tokens,
        "new_tokens": new_tokens,
        "saved_tokens": orig_tokens - new_tokens,
        "ratio": new_tokens / orig_tokens if orig_tokens > 0 else 1.0,
        "model": model,
        "content_types": content_types or [],
    }

    def _send() -> None:
        try:
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
            # Short timeout to avoid blocking or hanging the main process
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                resp.read()
        except Exception:
            # Catch all network/parse exceptions silently to never disrupt execution
            pass

    # Fire and forget: run in a background daemon thread
    t = threading.Thread(target=_send, daemon=True)
    t.start()
