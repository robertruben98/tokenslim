"""Tests for the packaged MCP server and agent registration (issue #120)."""

import json
from dataclasses import replace

import pytest

from tokenslim import mcp_server


def _retarget(monkeypatch, tmp_path):
    """Point both agents' config files at tmp_path and return their paths."""
    paths = {}
    for key, name in (("claude-code", "claude.json"), ("cursor", "cursor.json")):
        path = tmp_path / name
        monkeypatch.setitem(mcp_server.AGENTS, key, replace(mcp_server.AGENTS[key], path=path))
        paths[key] = path
    return paths


def test_build_server_entry_uses_current_interpreter():
    entry = mcp_server.build_server_entry()
    assert entry["args"] == ["-m", "tokenslim", "mcp"]
    assert entry["command"]  # some interpreter path


def test_compress_text_tool_returns_stats():
    # A long, repetitive JSON-ish payload should compress; contract holds either
    # way (never inflates), so assert the shape and the never-inflate invariant.
    text = json.dumps([{"id": i, "role": "member"} for i in range(400)])
    result = mcp_server.compress_text(text)
    assert set(result) == {"text", "orig_tokens", "new_tokens", "saved_tokens", "ratio"}
    assert result["new_tokens"] <= result["orig_tokens"]
    assert isinstance(result["text"], str)


def test_install_mcp_configs_roundtrip(tmp_path, monkeypatch):
    paths = _retarget(monkeypatch, tmp_path)

    results = mcp_server.install_mcp_configs(executable="/usr/bin/python3")
    assert {t.key: changed for t, changed in results} == {
        "claude-code": True,
        "cursor": True,
    }
    entry = json.loads(paths["claude-code"].read_text())["mcpServers"]["tokenslim"]
    assert entry == {"command": "/usr/bin/python3", "args": ["-m", "tokenslim", "mcp"]}

    # Re-installing is idempotent and preserves unrelated servers.
    data = json.loads(paths["claude-code"].read_text())
    data["mcpServers"]["other"] = {"command": "x"}
    paths["claude-code"].write_text(json.dumps(data))
    again = mcp_server.install_mcp_configs(["claude-code"], executable="/usr/bin/python3")
    assert again[0][1] is False
    preserved = json.loads(paths["claude-code"].read_text())["mcpServers"]
    assert "other" in preserved and "tokenslim" in preserved


def test_install_mcp_configs_rejects_unknown_agent(tmp_path, monkeypatch):
    _retarget(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        mcp_server.install_mcp_configs(["nope"])


def test_registered_agents_detects_only_installed(tmp_path, monkeypatch):
    _retarget(monkeypatch, tmp_path)
    assert mcp_server.registered_agents() == []
    mcp_server.install_mcp_configs(["claude-code"])
    assert [t.key for t in mcp_server.registered_agents()] == ["claude-code"]


def test_read_config_rejects_non_object(tmp_path, monkeypatch):
    paths = _retarget(monkeypatch, tmp_path)
    paths["claude-code"].write_text("[]")  # valid JSON, wrong shape
    with pytest.raises(ValueError):
        mcp_server.install_mcp_configs(["claude-code"])


# --- Tests that need the optional MCP SDK -----------------------------------

mcp = pytest.importorskip("mcp")


def test_build_server_registers_tool():
    server = mcp_server.build_server()
    assert server is not None


def test_cli_mcp_check_reports_tools():
    from click.testing import CliRunner

    from tokenslim.cli import main

    result = CliRunner().invoke(main, ["mcp", "--check"])
    assert result.exit_code == 0
    assert "compress_text" in result.output
