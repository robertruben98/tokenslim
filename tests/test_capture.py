import json
import os

from tokenslim import Config, SessionCapture, compress, get_capture, load_config, read_sessions


def _cfg(tmp_path, **kw):
    return Config(capture=True, capture_path=str(tmp_path), **kw)


def test_capture_off_by_default(tmp_path):
    cfg = Config(capture_path=str(tmp_path))
    assert cfg.capture is False
    assert get_capture(cfg) is None
    compress([{"role": "user", "content": "hello world"}], options=cfg, min_bytes=0)
    assert list(tmp_path.iterdir()) == [], "capture OFF must write nothing"


def test_record_writes_valid_jsonl(tmp_path):
    cap = SessionCapture(_cfg(tmp_path))
    cap.record("tool_call", {"tool": "grep", "arguments": {"pattern": "x"}})
    cap.record("outcome", {"status": "ok"})
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    assert files[0].name == f"{cap.session_id}.jsonl"
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        event = json.loads(line)
        assert set(event) == {"ts", "session_id", "kind", "payload"}
        assert event["session_id"] == cap.session_id
        assert isinstance(event["ts"], float)


def test_read_sessions_round_trip(tmp_path):
    cap = SessionCapture(_cfg(tmp_path))
    payloads = [{"n": i} for i in range(3)]
    for p in payloads:
        cap.record("step", p)
    events = list(read_sessions(str(tmp_path)))
    assert [e["payload"] for e in events] == payloads
    assert all(e["kind"] == "step" for e in events)
    assert all(e["session_id"] == cap.session_id for e in events)
    # A single file path works too.
    assert list(read_sessions(cap.path)) == events


def test_read_sessions_missing_and_malformed(tmp_path):
    assert list(read_sessions(str(tmp_path / "nope"))) == []
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"ok": 1}\nnot json\n\n[1, 2]\n', encoding="utf-8")
    assert list(read_sessions(str(tmp_path))) == [{"ok": 1}]


def test_read_sessions_survives_invalid_utf8(tmp_path):
    # A record truncated mid-multibyte character (kill -9 during record()).
    corrupt = tmp_path / "aaaa.jsonl"
    corrupt.write_bytes(b'{"ts": 1.0, "kind": "before"}\n{"t": "\xe6\x97')
    # Sorts after the corrupt file: must still be reached and yielded.
    (tmp_path / "zzzz.jsonl").write_text('{"ts": 2.0, "kind": "after"}\n', encoding="utf-8")
    events = list(read_sessions(str(tmp_path)))
    assert [e["kind"] for e in events] == ["before", "after"]


def test_compress_emits_compress_event(tmp_path):
    cfg = Config(capture=True, capture_path=str(tmp_path), min_bytes=0)
    messages = [{"role": "user", "content": "line one\nline two\nline three"}]
    compress(messages, options=cfg)
    events = list(read_sessions(str(tmp_path)))
    assert len(events) == 1
    event = events[0]
    assert event["kind"] == "compress"
    payload = event["payload"]
    assert payload["orig_tokens"] > 0
    assert payload["new_tokens"] > 0
    assert 0.0 <= payload["ratio"] <= 1.0
    assert isinstance(payload["content_types"], list) and payload["content_types"]
    assert "messages" not in payload, "raw content must stay private by default"


def test_capture_content_knob_includes_messages(tmp_path):
    cfg = Config(capture=True, capture_path=str(tmp_path), capture_content=True, min_bytes=0)
    messages = [{"role": "user", "content": "some payload text"}]
    compress(messages, options=cfg)
    events = list(read_sessions(str(tmp_path)))
    assert events and events[0]["payload"]["messages"] == messages


def test_env_opt_in_and_defaults():
    cfg = load_config(env={"TOKENSLIM_CAPTURE": "1", "TOKENSLIM_CAPTURE_PATH": "/tmp/sess"})
    assert cfg.capture is True
    assert cfg.capture_path == "/tmp/sess"
    assert Config().capture is False
    assert Config().capture_path is None
    assert Config().capture_content is False


def test_default_capture_dir_under_home():
    cap = SessionCapture(Config(capture=True))
    expected = os.path.expanduser(os.path.join("~", ".tokenslim", "sessions"))
    assert cap.directory == expected
    assert cap.path == os.path.join(expected, f"{cap.session_id}.jsonl")


def test_get_capture_singleton_per_directory(tmp_path):
    cfg = _cfg(tmp_path)
    a = get_capture(cfg)
    b = get_capture(cfg)
    assert a is not None and a is b
    other = get_capture(Config(capture=True, capture_path=str(tmp_path / "other")))
    assert other is not None and other is not a
    assert other.session_id != a.session_id


def test_record_never_raises(tmp_path):
    blocked = tmp_path / "file-not-dir"
    blocked.write_text("x", encoding="utf-8")
    cap = SessionCapture(Config(capture=True, capture_path=str(blocked / "sub")))
    cap.record("kind", {"a": 1})  # unwritable directory: swallowed
    ok = SessionCapture(_cfg(tmp_path))
    ok.record("kind", {"weird": object()})  # non-JSON value: coerced via str()
    events = list(read_sessions(ok.path))
    assert len(events) == 1 and "object object" in events[0]["payload"]["weird"]


def test_tool_call_and_outcome_helpers(tmp_path):
    cap = SessionCapture(_cfg(tmp_path))
    cap.record_tool_call("bash", {"command": "ls"}, duration_ms=12)
    cap.record_outcome("success", detail={"exit_code": 0})
    events = list(read_sessions(cap.path))
    assert events[0]["kind"] == "tool_call"
    expected_call = {"tool": "bash", "arguments": {"command": "ls"}, "duration_ms": 12}
    assert events[0]["payload"] == expected_call
    assert events[1]["kind"] == "outcome"
    assert events[1]["payload"] == {"status": "success", "detail": {"exit_code": 0}}
