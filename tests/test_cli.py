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


def test_cli_install_registers_local_command(tmp_path, monkeypatch) -> None:
    from dataclasses import replace

    from tokenslim import mcp_server

    # Retarget both agents' config files into the tmp dir.
    for key, name in (("claude-code", "claude.json"), ("cursor", "cursor.json")):
        monkeypatch.setitem(
            mcp_server.AGENTS, key, replace(mcp_server.AGENTS[key], path=tmp_path / name)
        )

    runner = CliRunner()
    result = runner.invoke(main, ["install", "--agent", "claude-code"])
    if "SDK is not installed" in result.output:
        # build_server() guard fires when the mcp extra is absent.
        assert result.exit_code == 1
        return
    assert result.exit_code == 0
    assert "tokenslim-mcp" not in result.output
    data = json.loads((tmp_path / "claude.json").read_text(encoding="utf-8"))
    assert data["mcpServers"]["tokenslim"]["args"] == ["-m", "tokenslim", "mcp"]
