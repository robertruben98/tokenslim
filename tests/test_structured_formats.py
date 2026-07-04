"""Tests for JSONL and Markdown-table support (#123)."""

from __future__ import annotations

import json

from tokenslim import Config, compress, detect_content_type, retrieve
from tokenslim.ccr import find_markers
from tokenslim.compressors.structured import JsonlCompressor, MarkdownTableCompressor
from tokenslim.detector import ContentType
from tokenslim.evals import run_suite


def _crush(text: str, **overrides):
    cfg = Config(min_bytes=0, **overrides)
    out, stats = compress([{"role": "tool", "tool_call_id": "t", "content": text}], cfg)
    return out[0]["content"], stats


# --- JSONL ---------------------------------------------------------------


def _jsonl(n: int = 400) -> str:
    lines = [json.dumps({"id": i, "level": "info", "msg": "ping"}) for i in range(n)]
    lines[213] = json.dumps({"id": 213, "level": "error", "msg": "shard-7 unreachable"})
    return "\n".join(lines)


def test_jsonl_is_detected_not_code():
    assert detect_content_type(_jsonl()).content_type is ContentType.JSONL


def test_jsonl_compresses_like_json():
    body, stats = _crush(_jsonl(500))
    assert stats.ratio >= 0.70
    # Output is still line-oriented (records or a CCR sentinel per line).
    for line in body.splitlines():
        s = line.strip()
        assert s.startswith(("{", "[", "[tokenslim:ccr]", "<<ccr"))


def test_jsonl_keeps_error_record_and_is_recoverable():
    body, stats = _crush(_jsonl(500))
    assert "shard-7 unreachable" in body  # rare/error record survives visibly
    markers = find_markers(body)
    assert markers
    assert retrieve(markers[0].hash, store=stats.store) is not None


def test_jsonl_never_inflates():
    body, stats = _crush("\n".join(json.dumps({"x": i}) for i in range(3)))
    assert stats.new_tokens <= stats.orig_tokens


def test_jsonl_malformed_is_passthrough():
    bad = '{"a":1}\nthis is not json\n{"b":2}'
    assert JsonlCompressor(Config())(bad) == bad


def test_jsonl_single_record_is_passthrough():
    one = '{"only": 1}'
    assert JsonlCompressor(Config())(one) == one


# --- Markdown tables -----------------------------------------------------


def _md_table(n: int = 200) -> str:
    rows = ["| id | name | code |", "| --- | --- | --- |"]
    rows += [f"| {i} | item-{i} | 200 |" for i in range(n - 1)]
    rows.append("| LAST | teapot | 418 |")
    return "\n".join(rows)


def test_md_table_is_detected():
    assert detect_content_type(_md_table()).content_type is ContentType.MD_TABLE


def test_md_table_compresses_and_stays_valid():
    body, stats = _crush(_md_table(200))
    assert stats.ratio > 0.40
    lines = body.splitlines()
    assert lines[0] == "| id | name | code |"  # header verbatim
    assert set(lines[1].strip()) <= set("|-: ")  # separator row intact


def test_md_table_keeps_tail_and_recovers_dropped():
    body, stats = _crush(_md_table(200))
    assert "teapot" in body  # tail row kept
    markers = find_markers(body)
    assert markers
    assert retrieve(markers[0].hash, store=stats.store) is not None


def test_md_table_small_is_passthrough():
    small = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
    assert MarkdownTableCompressor(Config())(small) == small


def test_md_table_preserves_trailing_prose():
    table = _md_table(60)
    text = table + "\n\nEsto es una nota después de la tabla."
    body, _ = _crush(text)
    assert body.rstrip().endswith("Esto es una nota después de la tabla.")


# --- no regressions ------------------------------------------------------


def test_plain_json_array_still_json():
    assert detect_content_type('[{"a":1},{"a":2}]').content_type is ContentType.JSON


def test_prose_still_text():
    assert detect_content_type("Hello, this is a normal sentence.").content_type is ContentType.TEXT


def test_eval_suite_includes_new_fixtures_and_is_faithful():
    results = {r.name: r for r in run_suite()}
    assert "jsonl-events" in results
    assert "md-table" in results
    assert results["jsonl-events"].faithful
    assert results["jsonl-events"].ratio >= 0.70
    assert results["md-table"].faithful
    assert results["md-table"].ratio > 0.40
