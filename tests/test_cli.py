import pytest

from tokenslim.cli import main


def test_perf_command(capsys):
    rc = main(["perf", "--model", "gpt-4o"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tokenslim perf report" in out
    assert "Estimated cost saved" in out


def test_evals_command_passes(capsys):
    rc = main(["evals"])
    out = capsys.readouterr().out
    # All bundled fixtures are faithful, so the eval command exits 0.
    assert rc == 0
    assert "PASS" in out
    assert "FAIL" not in out


def test_no_command_errors():
    with pytest.raises(SystemExit):
        main([])
