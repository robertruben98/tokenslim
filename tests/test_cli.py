import json
import os

from click.testing import CliRunner

from tokenslim.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Context compression layer" in result.output


def test_cli_doctor():
    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0
    assert "TokenSlim version" in result.output


def test_cli_perf():
    runner = CliRunner()
    messages = [
        {"role": "user", "content": "hello world"},
    ]
    result = runner.invoke(main, ["perf"], input=json.dumps(messages))
    assert result.exit_code == 0
    assert "Savings Report" in result.output


def test_cli_init():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init"], input="y\ny\ngpt-4o\n0.15\n")
        assert result.exit_code == 0
        assert os.path.exists(".env")
        with open(".env", encoding="utf-8") as f:
            content = f.read()
            assert "TOKENSLIM_ENABLED=true" in content
            assert "TOKENSLIM_TARGET_RATIO=0.15" in content
            assert "TOKENSLIM_MODEL=gpt-4o" in content


def test_cli_install_fallback() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["install"])
    assert result.exit_code == 1
    assert "tokenslim-mcp is not installed" in result.output
