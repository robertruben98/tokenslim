import json
import urllib.request

import pytest
from click.testing import CliRunner

from tokenslim.audit import parse_requests, render_audit_report, run_audit
from tokenslim.cli import main
from tokenslim.config import Config
from tokenslim.evals.fixtures import all_fixtures


def _requests():
    """Build audit requests from the bundled eval fixtures (no data files)."""
    return [
        {
            "id": f.name,
            "messages": [
                {"role": "user", "content": f.question or "Summarize this."},
                {"role": "tool", "tool_call_id": "t", "content": f.content},
            ],
        }
        for f in all_fixtures()
    ]


# --- token-only audit -------------------------------------------------------


def test_run_audit_saves_tokens_on_compressible_payloads():
    report = run_audit(_requests(), config=Config(min_bytes=0))
    assert len(report.rows) == len(all_fixtures())
    assert report.saved_tokens > 0, "compressible fixtures must show savings"
    assert 0.0 < report.ratio <= 1.0
    by_id = {row.id: row for row in report.rows}
    assert by_id["json-orders"].saved_tokens > 0
    for row in report.rows:
        assert row.baseline_tokens > 0
        assert row.optimized_tokens <= row.baseline_tokens
        assert row.saved_tokens == row.baseline_tokens - row.optimized_tokens


def test_run_audit_aggregates_match_row_sums():
    report = run_audit(_requests(), config=Config(min_bytes=0))
    assert report.baseline_tokens == sum(r.baseline_tokens for r in report.rows)
    assert report.optimized_tokens == sum(r.optimized_tokens for r in report.rows)
    assert report.saved_tokens == sum(r.saved_tokens for r in report.rows)
    assert report.saved_tokens == report.baseline_tokens - report.optimized_tokens
    for row in report.rows:
        # Per-content-type breakdown sums back to the row totals.
        assert sum(b["baseline_tokens"] for b in row.by_content_type.values()) == (
            row.baseline_tokens
        )
        assert sum(b["optimized_tokens"] for b in row.by_content_type.values()) == (
            row.optimized_tokens
        )


def test_run_audit_accepts_bare_messages_list():
    report = run_audit([[{"role": "user", "content": "hello world"}]], config=Config(min_bytes=0))
    assert len(report.rows) == 1
    assert report.rows[0].id == "req-0"
    assert report.rows[0].baseline_tokens > 0


def test_run_audit_cost_delta_with_model():
    report = run_audit(_requests(), config=Config(min_bytes=0), model="gpt-4o")
    assert report.model == "gpt-4o"
    assert report.baseline_cost is not None and report.baseline_cost > 0
    assert report.saved_cost == pytest.approx(report.baseline_cost - report.optimized_cost)
    # Without a model, cost fields stay None.
    no_model = run_audit(_requests()[:1], config=Config(min_bytes=0))
    assert no_model.baseline_cost is None and no_model.saved_cost is None


def test_run_audit_skips_unrecognized_requests_with_warning():
    report = run_audit([42, {"messages": "nope"}, {"foo": 1}], config=Config(min_bytes=0))
    assert report.rows == []
    assert len(report.warnings) == 3
    assert all("skipped" in w for w in report.warnings)


# --- input parsing ----------------------------------------------------------


def test_parse_requests_json_array_of_requests():
    text = json.dumps(
        [
            [{"role": "user", "content": "hi"}],
            {"id": "a", "messages": [{"role": "user", "content": "yo"}]},
        ]
    )
    assert len(parse_requests(text)) == 2


def test_parse_requests_bare_messages_array_is_one_request():
    text = json.dumps(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    requests = parse_requests(text)
    assert len(requests) == 1
    assert requests[0][0]["role"] == "user"


def test_parse_requests_jsonl():
    lines = [
        json.dumps([{"role": "user", "content": "one"}]),
        "",
        json.dumps({"id": "two", "messages": [{"role": "user", "content": "two"}]}),
    ]
    requests = parse_requests("\n".join(lines))
    assert len(requests) == 2


def test_parse_requests_rejects_unreadable_input():
    with pytest.raises(ValueError):
        parse_requests("this is not json at all {")
    with pytest.raises(ValueError):
        parse_requests("   ")
    with pytest.raises(ValueError):
        parse_requests('"just a string"')


# --- answers mode -----------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_run_audit_answers_mode_success(monkeypatch):
    payload = {"choices": [{"message": {"content": "the same answer"}}]}
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(payload))
    report = run_audit(
        _requests()[:1],
        config=Config(min_bytes=0),
        answers=True,
        env={"OPENAI_API_KEY": "test-key"},
    )
    assert report.answers_mode is True
    row = report.rows[0]
    assert row.baseline_answer == "the same answer"
    assert row.optimized_answer == "the same answer"
    assert row.answer_similarity == pytest.approx(1.0)


def test_run_audit_answers_mode_degrades_on_network_error():
    # Unroutable endpoint: connection refused -> warning + token-only rows.
    report = run_audit(
        _requests()[:2],
        config=Config(min_bytes=0),
        answers=True,
        env={"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "http://127.0.0.1:9"},
    )
    assert report.answers_mode is False
    assert len(report.rows) == 2, "token-only rows must still be produced"
    assert all(row.answer_similarity is None for row in report.rows)
    assert all(row.baseline_answer is None for row in report.rows)
    assert any("answers mode" in w and "token-only" in w for w in report.warnings)
    assert report.saved_tokens > 0


def test_run_audit_answers_mode_without_api_key_warns():
    report = run_audit(_requests()[:1], config=Config(min_bytes=0), answers=True, env={})
    assert report.answers_mode is False
    assert any("OPENAI_API_KEY" in w for w in report.warnings)
    assert report.rows[0].answer_similarity is None


# --- rendering --------------------------------------------------------------


def test_render_audit_report_table():
    report = run_audit(_requests(), config=Config(min_bytes=0), model="gpt-4o")
    text = render_audit_report(report)
    assert "# tokenslim audit report" in text
    assert "| Request | Baseline | Optimized | Saved | Ratio | Similarity |" in text
    assert "json-orders" in text
    assert "Estimated cost" in text


# --- CLI --------------------------------------------------------------------


def test_cli_audit_table_output():
    runner = CliRunner()
    result = runner.invoke(main, ["audit"], input=json.dumps(_requests()))
    assert result.exit_code == 0, result.output
    assert "tokenslim audit report" in result.output
    assert "json-orders" in result.output


def test_cli_audit_json_output():
    runner = CliRunner()
    result = runner.invoke(
        main, ["audit", "--json", "--model", "gpt-4o"], input=json.dumps(_requests())
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["requests"] == len(all_fixtures())
    assert data["saved_tokens"] > 0
    assert data["saved_tokens"] == sum(row["saved_tokens"] for row in data["rows"])
    assert data["baseline_cost_usd"] > data["optimized_cost_usd"]
    assert data["answers_mode"] is False


def test_cli_audit_jsonl_input():
    runner = CliRunner()
    lines = "\n".join(json.dumps(r) for r in _requests()[:2])
    result = runner.invoke(main, ["audit"], input=lines)
    assert result.exit_code == 0, result.output
    assert "**Requests:** 2" in result.output


def test_cli_audit_unreadable_input_exits_1():
    runner = CliRunner()
    result = runner.invoke(main, ["audit"], input="not json {")
    assert result.exit_code == 1
    assert "Error reading requests" in result.output


def test_cli_audit_answers_degrades_gracefully():
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["audit", "--answers"],
        input=json.dumps(_requests()[:1]),
        env={"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "http://127.0.0.1:9"},
    )
    assert result.exit_code == 0, result.output
    assert "Warning" in result.output
    assert "token-only" in result.output
    assert "**Answers mode:** off" in result.output
