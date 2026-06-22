from tokenslim.compressors.logs import (
    Level,
    LogCompressor,
    LogFormat,
    classify_line,
    detect_log_format,
)
from tokenslim.config import Config


def _compress(text, **cfg):
    return LogCompressor(Config(**cfg))(text)


# --- format detection -----------------------------------------------------


def test_detect_pytest():
    assert detect_log_format("FAILED tests/test_x.py::test_a") is LogFormat.PYTEST


def test_detect_cargo():
    assert detect_log_format("error[E0382]: borrow of moved value") is LogFormat.CARGO


def test_detect_jest():
    assert detect_log_format(" PASS  src/util.test.ts") is LogFormat.JEST


def test_detect_npm():
    assert detect_log_format("npm ERR! code ELIFECYCLE") is LogFormat.NPM


def test_detect_generic():
    assert detect_log_format("just some output lines") is LogFormat.GENERIC


# --- line classification --------------------------------------------------


def test_classify_levels():
    assert classify_line("ERROR: something broke") is Level.ERROR
    assert classify_line("E   AssertionError") is Level.ERROR
    assert classify_line("WARNING: deprecated") is Level.WARN
    assert classify_line("5 passed, 1 failed in 2.1s") is Level.SUMMARY
    assert classify_line("DEBUG connecting") is Level.DEBUG
    assert classify_line("plain progress line") is Level.INFO


# --- compression ----------------------------------------------------------


def test_compresses_and_keeps_errors_and_summary():
    lines = [f"INFO step {i} done" for i in range(60)]
    lines += [
        "FAILED tests/test_db.py::test_insert",
        "E   sqlite3.OperationalError: no such table",
        "5 passed, 1 failed in 3.2s",
    ]
    text = "\n".join(lines)
    out = _compress(text)
    assert len(out) < len(text)
    assert "FAILED tests/test_db.py::test_insert" in out
    assert "no such table" in out
    assert "1 failed in 3.2s" in out
    # The noisy INFO body is dropped behind a CCR marker.
    assert "[tokenslim:ccr]" in out


def test_keeps_context_window_around_errors():
    lines = [f"line {i}" for i in range(40)]
    lines[20] = "ERROR boom"
    text = "\n".join(lines)
    out = _compress(text, log_context=2)
    # The two lines on each side of the error survive.
    assert "line 18" in out and "line 19" in out
    assert "line 21" in out and "line 22" in out


def test_short_log_is_untouched():
    text = "\n".join(["a", "b", "c"])
    assert _compress(text) == text


def test_conservative_dedup_collapses_pure_duplicates():
    # 30 identical noise lines plus a failure; the duplicates that survive as
    # context/head collapse, but never across distinguishing content.
    lines = ["connecting to pool"] * 30 + ["ERROR connection refused"]
    text = "\n".join(lines)
    out = _compress(text)
    assert "ERROR connection refused" in out
    # Should not reproduce all 30 identical lines verbatim.
    assert out.count("connecting to pool") < 30


def test_dedup_preserves_distinguishing_ids():
    lines = [f"processed request 0x{1000 + i:x}" for i in range(30)]
    lines.append("ERROR batch failed")
    text = "\n".join(lines)
    out = _compress(text)
    # Lines differ by hex id -> must not be collapsed into a single "(xN)".
    assert "(x" not in out
