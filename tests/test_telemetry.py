import pytest

from tokenslim.compress import compress
from tokenslim.config import Config, load_config
from tokenslim.telemetry import (
    NullTelemetry,
    TelemetryEvent,
    get_sink,
    record_event,
    set_sink,
)


@pytest.fixture
def fresh_sink():
    prev = get_sink()
    sink = NullTelemetry()
    set_sink(sink)
    yield sink
    set_sink(prev)


def test_default_sink_buffers_locally(fresh_sink):
    record_event("json", 100, 20, "smartcrusher")
    assert len(fresh_sink.events) == 1
    e = fresh_sink.events[0]
    assert isinstance(e, TelemetryEvent)
    assert e.content_type == "json"
    assert e.ratio == pytest.approx(0.8)


def test_disabled_emits_nothing(fresh_sink):
    record_event("json", 100, 20, "smartcrusher", enabled=False)
    assert fresh_sink.events == []


def test_telemetry_on_by_default():
    # Issue #43: telemetry is ON by default.
    assert Config().telemetry is True


def test_env_off_disables():
    assert load_config(env={"TOKENSLIM_TELEMETRY": "off"}).telemetry is False
    assert load_config(env={"TOKENSLIM_TELEMETRY": "false"}).telemetry is False
    assert load_config(env={"TOKENSLIM_TELEMETRY": "on"}).telemetry is True


def test_compress_emits_telemetry_when_enabled(fresh_sink):
    payload = "[" + ",".join(f'{{"id":{i}}}' for i in range(200)) + "]"
    compress(
        [{"role": "tool", "tool_call_id": "t", "content": payload}],
        options=Config(min_bytes=0, telemetry=True),
    )
    assert len(fresh_sink.events) >= 1
    assert fresh_sink.events[0].compressor == "smartcrusher"


def test_compress_silent_when_telemetry_off(fresh_sink):
    payload = "[" + ",".join(f'{{"id":{i}}}' for i in range(200)) + "]"
    compress(
        [{"role": "tool", "tool_call_id": "t", "content": payload}],
        options=Config(min_bytes=0, telemetry=False),
    )
    assert fresh_sink.events == []


def test_event_carries_no_payload(fresh_sink):
    # The event schema has only aggregate fields — assert it cannot leak content.
    record_event("log", 500, 50, "log-compressor")
    e = fresh_sink.events[0]
    assert set(vars(e)) == {"content_type", "orig_tokens", "new_tokens", "compressor"}
