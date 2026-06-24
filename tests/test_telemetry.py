import urllib.request
from unittest.mock import MagicMock

from tokenslim.telemetry import send_telemetry_event


def test_telemetry_sends_event(monkeypatch):
    mock_urlopen = MagicMock()
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    # Make sure env is not opted out
    monkeypatch.setenv("TOKENSLIM_TELEMETRY", "on")

    send_telemetry_event(100, 50, model="gpt-4o", content_types=["code"])

    # Wait briefly since it runs in a background thread
    import time

    time.sleep(0.1)

    assert mock_urlopen.call_count == 1
    req = mock_urlopen.call_args[0][0]
    assert req.method == "POST"
    assert req.full_url == "https://telemetry.tokenslim.dev/v1/event"


def test_telemetry_opt_out(monkeypatch):
    mock_urlopen = MagicMock()
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    # Set opt-out
    monkeypatch.setenv("TOKENSLIM_TELEMETRY", "off")

    send_telemetry_event(100, 50)

    import time

    time.sleep(0.1)

    assert mock_urlopen.call_count == 0
