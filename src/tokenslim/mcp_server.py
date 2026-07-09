"""TokenSlim MCP server and agent registration (issue #120).

Serves the compression layer over the Model Context Protocol so agents
(Claude Code, Cursor, …) can call it directly:

* ``tokenslim mcp`` runs the stdio server.
* ``tokenslim install`` registers *that local command* into the agents'
  ``mcpServers`` config — no separate PyPI package required.
* ``tokenslim doctor`` smoke-tests that the registered server starts.

The MCP SDK is an optional dependency (the ``mcp`` extra); it is imported
lazily inside :func:`build_server` so importing this module — or any core
tokenslim path — never requires it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compress import compress
from .config import load_config

__all__ = [
    "SERVER_NAME",
    "TOOLS",
    "MissingMCPError",
    "AgentTarget",
    "AGENTS",
    "compress_text",
    "build_server",
    "build_server_entry",
    "serve",
    "install_mcp_configs",
    "registered_agents",
    "smoke_test_command",
]

# Name the server registers under in every agent's ``mcpServers`` map.
SERVER_NAME = "tokenslim"

# Tools the server exposes (declared here so ``--check`` can report them
# without spinning up the async MCP runtime).
TOOLS = ("compress_text",)


class MissingMCPError(RuntimeError):
    """Raised when the optional ``mcp`` SDK is not installed."""

    def __init__(self) -> None:
        super().__init__(
            'The MCP SDK is not installed. Install it with: pip install "tokenslim[mcp]"'
        )


@dataclass(frozen=True)
class AgentTarget:
    """An agent whose MCP config we know how to write."""

    key: str  # CLI value, e.g. "claude-code"
    label: str  # Human-readable name
    path: Path  # Config file to edit
    servers_key: str = "mcpServers"  # Top-level key holding the server map


# Registry of supported agents. Both Claude Code and Cursor consume the
# standard ``mcpServers`` schema, differing only in the config file location.
AGENTS: dict[str, AgentTarget] = {
    "claude-code": AgentTarget(
        key="claude-code",
        label="Claude Code",
        path=Path.home() / ".claude.json",
    ),
    "cursor": AgentTarget(
        key="cursor",
        label="Cursor",
        path=Path.home() / ".cursor" / "mcp.json",
    ),
}


def build_server_entry(executable: str | None = None) -> dict[str, Any]:
    """Return the ``mcpServers`` entry that launches this package's server.

    Uses ``<python> -m tokenslim mcp`` so the agent reuses the exact
    interpreter tokenslim is installed in, regardless of PATH.
    """
    return {"command": executable or sys.executable, "args": ["-m", "tokenslim", "mcp"]}


def compress_text(
    text: str,
    target_ratio: float | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Compress a block of text and report the token savings.

    Returns the compressed text plus original/compressed token counts so the
    calling agent can drop it straight back into its own context. Registered as
    the MCP ``compress_text`` tool (defined at module level so the compression
    path is usable and testable without the MCP SDK).
    """
    cfg = load_config()
    messages: list[dict[str, Any]] = [{"role": "user", "content": text}]
    # min_bytes=0 so short snippets are considered too; compress() still only
    # keeps a rewrite when it is a net token win (never inflates).
    new_messages, stats = compress(
        messages, options=cfg, min_bytes=0, target_ratio=target_ratio, model=model
    )
    return {
        "text": new_messages[0]["content"],
        "orig_tokens": stats.orig_tokens,
        "new_tokens": stats.new_tokens,
        "saved_tokens": stats.saved_tokens,
        "ratio": stats.ratio,
    }


def build_server() -> Any:
    """Build the FastMCP server exposing tokenslim's compression tools.

    Imports the MCP SDK lazily; raises :class:`MissingMCPError` if the ``mcp``
    extra is not installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise MissingMCPError() from exc

    server = FastMCP(SERVER_NAME)
    server.tool()(compress_text)
    return server


def serve() -> None:
    """Run the MCP server over stdio (blocks until the client disconnects)."""
    build_server().run()


def _read_config(path: Path) -> dict[str, Any]:
    """Load an agent config file, tolerating a missing or empty file."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def _write_config(path: Path, data: dict[str, Any]) -> None:
    """Write an agent config file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def install_mcp_configs(
    agents: list[str] | None = None,
    *,
    executable: str | None = None,
) -> list[tuple[AgentTarget, bool]]:
    """Register the tokenslim MCP server into each agent's config.

    Args:
        agents: Agent keys to install into (defaults to all known agents).
        executable: Python interpreter for the launch command (defaults to the
            current one).

    Returns:
        ``(target, changed)`` per agent, where ``changed`` is False when the
        config already pointed at the same command (idempotent re-install).
    """
    keys = agents or list(AGENTS)
    unknown = [k for k in keys if k not in AGENTS]
    if unknown:
        raise ValueError(f"Unknown agent(s): {', '.join(unknown)}. Known: {', '.join(AGENTS)}")

    entry = build_server_entry(executable)
    results: list[tuple[AgentTarget, bool]] = []
    for key in keys:
        target = AGENTS[key]
        data = _read_config(target.path)
        servers = data.setdefault(target.servers_key, {})
        changed = servers.get(SERVER_NAME) != entry
        if changed:
            servers[SERVER_NAME] = entry
            _write_config(target.path, data)
        results.append((target, changed))
    return results


def registered_agents() -> list[AgentTarget]:
    """Return the agents whose config currently registers the tokenslim server."""
    found: list[AgentTarget] = []
    for target in AGENTS.values():
        try:
            data = _read_config(target.path)
        except (OSError, ValueError):
            continue
        if (
            isinstance(data.get(target.servers_key), dict)
            and SERVER_NAME in data[target.servers_key]
        ):
            found.append(target)
    return found


def smoke_test_command(command: list[str] | None = None, *, timeout: float = 30.0) -> str:
    """Run the registered MCP command with ``--check`` and return its output.

    Raises ``subprocess.CalledProcessError`` on a non-zero exit and
    ``subprocess.TimeoutExpired`` if it hangs — either means the server the
    agent would launch does not start.
    """
    entry = build_server_entry()
    cmd = command or [entry["command"], *entry["args"], "--check"]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return proc.stdout.strip()
