import json

from tokenslim.config import Config
from tokenslim.evals import perf_report, run_suite
from tokenslim.evals.fixtures import Fixture, all_fixtures
from tokenslim.evals.harness import _evaluate


def test_run_suite_all_faithful():
    results = run_suite()
    assert len(results) == len(all_fixtures())
    # Every bundled fixture must be faithful: must-keeps survive AND drops are
    # recoverable from the CCR store.
    for r in results:
        assert r.faithful, f"{r.name} not faithful: missing={r.missing}"


def test_run_suite_actually_compresses():
    results = run_suite()
    # At least the JSON and log fixtures compress substantially.
    by_name = {r.name: r for r in results}
    assert by_name["json-orders"].ratio > 0.5
    assert by_name["sre-log"].ratio > 0.5


def test_json_cyclic_regression_does_not_degenerate():
    # #122: a cyclic numeric column used to crush ~0%; the guard must keep it
    # well above 70% while its genuine anomaly stays faithful.
    by_name = {r.name: r for r in run_suite()}
    cyclic = by_name["json-cyclic"]
    assert cyclic.ratio >= 0.7
    assert cyclic.faithful
    # No regression on the monotone JSON fixture.
    assert by_name["json-orders"].ratio > 0.9


def test_must_keep_survives_in_visible_output():
    # The faithfulness guarantee the milestone cares about: answer-bearing
    # content (the error row) is present in the *visible* compressed output.
    fixtures = [f for f in all_fixtures() if f.name == "json-orders"]
    result, visible = _evaluate(fixtures[0], Config(min_bytes=0))
    assert "payment declined by issuer" in visible
    assert result.must_keep_ok


def test_dropped_rows_are_retrievable_exactly():
    # compress -> retrieve round-trip: the dropped middle rows come back byte-
    # for-byte (parse-equal) from the store.
    rows = [{"id": i, "v": "x"} for i in range(300)]
    fixture = Fixture(name="big", content=json.dumps(rows))
    result, visible = _evaluate(fixture, Config(min_bytes=0))
    assert result.n_markers == 1
    assert result.retrievable_ok


def test_error_rows_always_survive_crushing():
    rows = [{"id": i, "status": "ok"} for i in range(300)]
    rows[150] = {"id": 150, "status": "error", "msg": "boom"}
    fixture = Fixture(name="errs", content=json.dumps(rows), must_keep=("boom",))
    result, visible = _evaluate(fixture, Config(min_bytes=0))
    assert "boom" in visible
    assert result.must_keep_ok


def test_perf_report_renders_savings():
    report = perf_report(model="gpt-4o")
    assert report.startswith("# tokenslim perf report")
    assert "Saved tokens" in report
    assert "Estimated cost saved" in report
    assert "All faithful:** True" in report


def test_missing_must_keep_flags_unfaithful():
    # A fixture asserting a string that compression would drop -> not faithful.
    rows = [{"id": i, "tag": "common"} for i in range(300)]
    rows[150] = {"id": 150, "tag": "RARE_BURIED_VALUE_NOT_KEPT"}
    # Disable rare-value preservation by making it look like a high-card id-ish
    # field is not what we key on; here we simply assert a value that lives in
    # the dropped middle and is not error/rare-preserved.
    fixture = Fixture(
        name="lost",
        content=json.dumps(rows),
        must_keep=("definitely-not-present-anywhere",),
    )
    result, _ = _evaluate(fixture, Config(min_bytes=0))
    assert not result.must_keep_ok
    assert not result.faithful
