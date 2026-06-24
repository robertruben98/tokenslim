import pytest

from tokenslim.compress import compress
from tokenslim.config import Config
from tokenslim.metrics import MetricsCollector


def test_record_and_aggregate():
    m = MetricsCollector(model="gpt-4o")
    m.record(1000, 200, label="a")
    m.record(2000, 500, label="b")
    assert m.total_orig_tokens == 3000
    assert m.total_new_tokens == 700
    assert m.total_saved_tokens == 2300
    assert m.overall_ratio == pytest.approx(1 - 700 / 3000)


def test_saved_usd_uses_input_rate():
    m = MetricsCollector(model="gpt-4o")
    m.record(1_000_000, 0)  # saved 1M input tokens at $2.5/1M
    assert m.saved_usd() == pytest.approx(2.5)


def test_per_run_model_overrides_collector_default():
    m = MetricsCollector(model="gpt-4o-mini")
    m.record(1_000_000, 0, model="gpt-4o")  # this run priced as gpt-4o
    assert m.saved_usd() == pytest.approx(2.5)


def test_record_from_stats():
    payload = "[" + ",".join(f'{{"id":{i},"status":"ok"}}' for i in range(200)) + "]"
    _, stats = compress(
        [{"role": "tool", "tool_call_id": "t", "content": payload}],
        options=Config(min_bytes=0),
    )
    m = MetricsCollector(model="gpt-4o")
    rec = m.record_stats(stats, label="orders")
    assert rec.saved_tokens > 0
    assert m.saved_usd() > 0


def test_report_markdown_structure():
    m = MetricsCollector(model="gpt-4o")
    m.record(1000, 200, label="run-a")
    report = m.generate_report()
    assert report.startswith("# tokenslim savings report")
    assert "Saved tokens" in report
    assert "Estimated cost saved" in report
    assert "run-a" in report
    assert "| Run | Model |" in report


def test_empty_report():
    assert "No runs recorded" in MetricsCollector().generate_report()
