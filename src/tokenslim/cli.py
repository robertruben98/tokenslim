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
@click.option("--model", default=None, help="LLM model name for cost estimation.")
@click.argument("file", type=click.File("r", encoding="utf-8"), default="-")
def perf(model: str | None, file: Any) -> None:
    """Run a performance savings report on a JSON message array."""
    if file.name == "<stdin>" and sys.stdin.isatty():
        from .evals import perf_report

        click.echo(perf_report(model=model))
        return

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
    model_name = model or cfg.model or "gpt-4o"
    try:
        _, stats = compress(data, options=cfg)
        from .pricing import estimate_cost

        orig_cost = estimate_cost(model_name, stats.orig_tokens)
        new_cost = estimate_cost(model_name, stats.new_tokens)
        saved_cost = orig_cost - new_cost

        click.echo("\n--- TokenSlim Savings Report ---")
        click.echo(f"Model:            {model_name}")
        click.echo(f"Original Tokens:  {stats.orig_tokens} (${orig_cost:.4f})")
        click.echo(f"Compressed:       {stats.new_tokens} (${new_cost:.4f})")
        click.echo(f"Saved Tokens:     {stats.saved_tokens} (${saved_cost:.4f})")
        click.echo(f"Savings Ratio:    {stats.ratio:.1%}")
        click.echo("--------------------------------")
    except Exception as e:
        click.echo(f"Error during compression: {e}", err=True)
        sys.exit(1)


@main.command("refresh-pricing")
@click.option(
    "--url",
    default="https://raw.githubusercontent.com/robertruben98/tokenslim/main/pricing.json",
    help="Pricing JSON URL.",
)
def refresh_pricing_cmd(url: str) -> None:
    """Refresh the local model token pricing cache."""
    click.echo(f"Refreshing pricing cache from {url}...")
    from .pricing import refresh_pricing

    if refresh_pricing(url):
        click.echo("✓ Pricing cache updated successfully!")
    else:
        click.echo("✗ Error: Failed to download or parse pricing data from URL.", err=True)
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
@click.option(
    "--sessions",
    "sessions_path",
    default=None,
    metavar="PATH",
    help="Capture directory or JSONL file to mine (default: ~/.tokenslim/sessions).",
)
@click.option(
    "--target",
    default="CLAUDE.md",
    show_default=True,
    metavar="PATH",
    help="Rules file to update (CLAUDE.md, AGENTS.md, or any path).",
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Write the learned-rules section to the target (default: preview only).",
)
def learn(sessions_path: str | None, target: str, apply_changes: bool) -> None:
    """Mine captured session failures into agent rules (CLAUDE.md/AGENTS.md)."""
    from .capture import read_sessions
    from .learn import analyze_sessions, apply_rules, propose_rules

    events = list(read_sessions(sessions_path))
    if not events:
        click.echo("no sessions found — enable capture with TOKENSLIM_CAPTURE=1 and retry.")
        return

    findings = analyze_sessions(events)
    if not findings:
        click.echo(f"Analyzed {len(events)} events: no recurring failure patterns found.")
        return

    click.echo(f"Found {len(findings)} pattern(s) in {len(events)} events:")
    for finding in findings:
        click.echo(
            f"- [{finding.kind}] x{finding.evidence_count} in {len(finding.sessions)} "
            f"session(s), confidence {finding.confidence:.2f}: {finding.proposed_rule}"
        )

    block = propose_rules(findings)
    try:
        diff = apply_rules(block, target, dry_run=not apply_changes)
    except (OSError, ValueError) as e:
        # ValueError covers malformed managed-section markers and
        # UnicodeDecodeError (its subclass) from non-UTF-8 targets.
        click.echo(f"Error updating {target}: {e}", err=True)
        sys.exit(1)

    if not diff:
        click.echo(f"\n{target} is already up to date.")
        return
    click.echo("\n" + diff.rstrip("\n"))
    if apply_changes:
        click.echo(f"\nApplied learned rules to {target}.")
    else:
        click.echo("\nPreview only — re-run with --apply to write.")


@main.command()
@click.option(
    "--model",
    default=None,
    help="LLM model name for token counting.",
)
def evals(model: str | None) -> None:
    """Run accuracy-preservation evaluation harness (ratio + faithfulness)."""
    click.echo("Running offline quality evaluation suite...")
    from .config import Config
    from .evals import run_suite

    results = run_suite(config=Config(min_bytes=0, model=model))
    all_faithful = all(r.faithful for r in results)

    click.echo("\n--- Accuracy Preservation Report ---")
    for r in results:
        status = "PASS" if r.faithful else "FAIL"
        click.echo(f"[{status}] {r.name}: ratio={r.ratio:.1%} drops={r.n_markers}")
        if r.missing:
            click.echo(f"        missing must-keep: {r.missing}")
    click.echo("------------------------------------")
    if not all_faithful:
        click.echo("⚠️ Warning: Compression caused accuracy loss on some tasks!", err=True)
        sys.exit(1)


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


@main.command()
@click.option("--model", default=None, help="LLM model name for token counting and cost.")
@click.option(
    "--answers",
    is_flag=True,
    help="Also replay both variants against an OpenAI-compatible endpoint "
    "(needs OPENAI_API_KEY; optional OPENAI_BASE_URL) and diff the answers.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit the machine-readable JSON report instead of the table.",
)
@click.argument("file", type=click.File("r", encoding="utf-8"), default="-")
def audit(model: str | None, answers: bool, as_json: bool, file: Any) -> None:
    """Replay requests baseline vs optimized and diff token counts.

    FILE (default: stdin) contains the requests in either shape:

    \b
      * JSON array — each element a bare messages array or an object like
        {"messages": [...], "id": "req-1"}. A top-level array of
        role/content message objects counts as a single request.
      * JSONL — one request per line (same element shapes).
    """
    from .audit import parse_requests, render_audit_report, run_audit

    try:
        requests = parse_requests(file.read())
    except Exception as e:
        click.echo(f"Error reading requests: {e}", err=True)
        sys.exit(1)

    report = run_audit(requests, config=load_config(), model=model, answers=answers)
    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        click.echo(render_audit_report(report))


if __name__ == "__main__":
    main()
