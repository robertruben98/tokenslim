"""TokenSlim CLI — command-line tool scaffold."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import click

from . import __version__
from .compress import compress
from .config import load_config


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """TokenSlim CLI — Context compression layer for LLM agents."""
    pass


@main.command()
def init() -> None:
    """Interactive project setup wizard."""
    click.echo("Welcome to TokenSlim initialization wizard!")
    enabled = click.confirm("Enable context compression by default?", default=True)
    ccr = click.confirm("Enable CCR (Compress-Cache-Retrieve) reversibility?", default=True)
    model = click.prompt(
        "Default LLM model name (optional, e.g. gpt-4o)", default="", show_default=False
    )
    ratio = click.prompt("Default target compression ratio (0.0 to 1.0)", type=float, default=0.2)

    lines = [
        f"TOKENSLIM_ENABLED={str(enabled).lower()}",
        f"TOKENSLIM_CCR={str(ccr).lower()}",
        f"TOKENSLIM_TARGET_RATIO={ratio}",
    ]
    if model:
        lines.append(f"TOKENSLIM_MODEL={model}")

    env_path = ".env"
    mode = "a" if os.path.exists(env_path) else "w"
    with open(env_path, mode, encoding="utf-8") as f:
        f.write("\n" + "\n".join(lines) + "\n")

    click.echo(f"Saved default configuration to {env_path}!")


@main.command()
def doctor() -> None:
    """Run diagnostics on TokenSlim installation and environment."""
    click.echo(f"TokenSlim version: {__version__}")
    click.echo(f"Python version: {sys.version.split()[0]}")

    try:
        import tiktoken  # noqa: F401

        click.echo("✓ tiktoken: Installed (accurate token counting enabled)")
    except ImportError:
        click.echo("✗ tiktoken: Not installed (using heuristic tokenizer)")

    try:
        import magika  # noqa: F401

        click.echo("✓ magika: Installed (ML-based content-type detection enabled)")
    except ImportError:
        click.echo("✗ magika: Not installed (using rule-based detector)")

    click.echo("\nResolved Configuration:")
    cfg = load_config()
    for key, value in sorted(cfg.__dict__.items()):
        click.echo(f"  TOKENSLIM_{key.upper()}: {value}")


@main.command()
@click.argument("file", type=click.File("r", encoding="utf-8"), default="-")
def perf(file: Any) -> None:
    """Run a performance savings report on a JSON message array."""
    try:
        data = json.load(file)
    except Exception as e:
        click.echo(f"Error parsing input JSON: {e}", err=True)
        sys.exit(1)

    if not isinstance(data, list):
        click.echo("Input must be a JSON array of messages.", err=True)
        sys.exit(1)

    click.echo("Running compression performance analysis...")
    cfg = load_config()
    try:
        _, stats = compress(data, options=cfg)
        click.echo("\n--- TokenSlim Savings Report ---")
        click.echo(f"Original Tokens:  {stats.orig_tokens}")
        click.echo(f"Compressed:       {stats.new_tokens}")
        click.echo(f"Saved Tokens:     {stats.saved_tokens}")
        click.echo(f"Savings Ratio:    {stats.ratio:.1%}")
        click.echo("--------------------------------")
    except Exception as e:
        click.echo(f"Error during compression: {e}", err=True)
        sys.exit(1)


@main.command(context_settings={"ignore_unknown_options": True})
@click.argument("cmd", nargs=-1, type=click.UNPROCESSED)
def wrap(cmd: tuple[str, ...]) -> None:
    """Wrap a command and run it with context compression enabled."""
    if not cmd:
        click.echo("Usage: tokenslim wrap <command> [args...]")
        sys.exit(1)

    env = os.environ.copy()
    env["TOKENSLIM_ENABLED"] = "true"

    try:
        sys.exit(subprocess.call(cmd, env=env))
    except Exception as e:
        click.echo(f"Error running wrapped command: {e}", err=True)
        sys.exit(1)


@main.command()
def proxy() -> None:
    """Start the transparent compression proxy (Coming Soon)."""
    click.echo("TokenSlim HTTP proxy — transparent context compression.")
    click.echo("This subcommand is a placeholder and will be fully wired in M7.")


@main.command()
def learn() -> None:
    """Mine failure logs and generate learning rules (Coming Soon)."""
    click.echo("Mining failures to write agent rules...")
    click.echo("This subcommand is a placeholder and will be fully wired in M5.")


@main.command()
def evals() -> None:
    """Run accuracy-preservation evaluation harness (Coming Soon)."""
    click.echo("Running quality evaluation suite...")
    click.echo("This subcommand is a placeholder and will be fully wired in M4.")


@main.command()
def memory() -> None:
    """Inspect or query persistent agent memory (Coming Soon)."""
    click.echo("Querying SQLite/Vector memory store...")
    click.echo("This subcommand is a placeholder and will be fully wired in M3.")


@main.command()
def install() -> None:
    """Install/Register MCP server across agent platforms (Claude Code, Cursor)."""
    try:
        from tokenslim_mcp.install import install_mcp_configs

        install_mcp_configs()
    except ImportError:
        click.echo("Error: tokenslim-mcp is not installed in the current environment.", err=True)
        click.echo("Please install it first: pip install tokenslim-mcp", err=True)
        sys.exit(1)


@main.command()
def update() -> None:
    """Check for and apply updates to TokenSlim (Coming Soon)."""
    click.echo("Checking for updates...")
    click.echo("This subcommand is a placeholder and will be fully wired in M11.")


if __name__ == "__main__":
    main()
