# CLI reference

The `tokenslim` console script is installed with the package. Every command
supports `--help`; `tokenslim --version` prints the package version.

```text
Usage: tokenslim [OPTIONS] COMMAND [ARGS]...

  TokenSlim CLI — Context compression layer for LLM agents.
```

| Command | Purpose |
| --- | --- |
| [`init`](#init) | Interactive `.env` setup wizard |
| [`doctor`](#doctor) | Environment diagnostics + fully resolved configuration |
| [`perf`](#perf) | Token/cost savings report for a message array |
| [`evals`](#evals) | Offline accuracy-preservation suite (CI-friendly exit code) |
| [`audit`](#audit) | Replay requests baseline vs optimized and diff tokens/answers |
| [`proxy`](#proxy) | OpenAI-compatible compressing reverse proxy |
| [`wrap`](#wrap) | Run any command with `TOKENSLIM_ENABLED=true` injected |
| [`learn`](#learn) | Mine captured session failures into agent rules |
| [`refresh-pricing`](#refresh-pricing) | Refresh the local model-pricing cache |
| [`install`](#install) | Register the MCP server across agent platforms |
| [`memory`](#memory) / [`update`](#update) | Placeholders for later milestones |

## init

Interactive wizard that writes `TOKENSLIM_*` defaults to a `.env` file in the
current directory (appending when one exists):

```console
$ tokenslim init
Welcome to TokenSlim initialization wizard!
Enable context compression by default? [Y/n]: y
Enable CCR (Compress-Cache-Retrieve) reversibility? [Y/n]: y
Default LLM model name (optional, e.g. gpt-4o): gpt-4o
Default target compression ratio (0.0 to 1.0) [0.2]:
Saved default configuration to .env!
```

!!! note
    TokenSlim itself does not load `.env` files — export the variables (or use
    your process manager / `direnv`) so they reach the environment.

## doctor

Prints the version, Python version, optional-dependency status, and every
resolved configuration field with its `TOKENSLIM_*` env var name — the
quickest way to see what a compression run will actually use:

```console
$ tokenslim doctor
TokenSlim version: 0.4.0
Python version: 3.12.8
✓ tiktoken: Installed (accurate token counting enabled)
✗ magika: Not installed (using rule-based detector)

Resolved Configuration:
  TOKENSLIM_CCR: True
  TOKENSLIM_CCR_BACKEND: memory
  TOKENSLIM_MIN_BYTES: 200
  ...
```

The install scripts (`install.sh` / `install.ps1`) run `tokenslim doctor` as
their post-install verification step.

## perf

Compresses a JSON message array and reports token and dollar savings.
Reads from a file argument or stdin (default):

```console
$ tokenslim perf --model gpt-4o < messages.json
Running compression performance analysis...

--- TokenSlim Savings Report ---
Model:            gpt-4o
Original Tokens:  5929 ($0.0148)
Compressed:       147 ($0.0004)
Saved Tokens:     5782 ($0.0145)
Savings Ratio:    97.5%
--------------------------------
```

- `--model NAME` — model for token counting and pricing (default:
  `TOKENSLIM_MODEL` or `gpt-4o`).
- `FILE` — path to a JSON array of messages; `-` (default) reads stdin.
- Run it **interactively with no piped input** and it renders the bundled
  demo-workload report instead (`perf_report()`).

## evals

Runs the offline accuracy-preservation suite: each bundled fixture is
compressed, dropped material is fetched back from the CCR store and
**byte-compared**, and must-keep rows (errors) are checked in the visible
output. Exits `1` if any fixture is unfaithful — safe to wire into CI:

```console
$ tokenslim evals
Running offline quality evaluation suite...

--- Accuracy Preservation Report ---
[PASS] json-orders: ratio=93.8% drops=1
[PASS] sre-log: ratio=92.4% drops=2
[PASS] code-search: ratio=0.0% drops=0
------------------------------------
```

`--model NAME` selects the tokenizer used for the ratio numbers.

## audit

Replays a batch of requests **baseline vs optimized** and diffs the token
counts, with a per-content-type breakdown and (when a model is known) an
estimated USD delta:

```console
$ tokenslim audit < requests.json
# tokenslim audit report

- **Requests:** 1
- **Baseline tokens:** 5,929
- **Optimized tokens:** 147
- **Saved tokens:** 5,782 (97.5%)
- **Answers mode:** off

| Request | Baseline | Optimized | Saved | Ratio | Similarity |
| --- | ---: | ---: | ---: | ---: | ---: |
| req-0 | 5,929 | 147 | 5,782 | 97.5% | - |
```

Input (`FILE` argument or stdin) is either a JSON array or JSONL, where each
request is a bare messages array or `{"messages": [...], "id": "req-1"}`; a
top-level array of role/content objects counts as a single request.

- `--model NAME` — tokenizer + pricing model for the cost delta.
- `--answers` — also replay *both* variants against an OpenAI-compatible
  endpoint (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `temperature=0`) and
  report the answer similarity per request (`difflib` ratio, `1.00` =
  identical). Network/auth failures degrade to a token-only audit with a
  warning — the command never crashes mid-report.
- `--json` — emit the machine-readable report instead of the table.

## proxy

Runs the stdlib-only OpenAI-compatible compressing reverse proxy (see
[Integrations](integrations.md#reverse-proxy) for the full behavior and the
Docker image):

```bash
tokenslim proxy --port 8787 --upstream https://api.openai.com
```

Defaults come from `TOKENSLIM_PROXY_PORT` (8787) and `TOKENSLIM_UPSTREAM`
(`https://api.openai.com`).

## wrap

Runs any command with `TOKENSLIM_ENABLED=true` injected into its environment —
useful when the child process itself embeds TokenSlim. Arguments after the
command are passed through untouched, exit code included:

```bash
tokenslim wrap -- my-agent --its-flags
```

## learn

Mines captured session events for recurring failure patterns and turns them
into rules for your agent instructions file (`CLAUDE.md`, `AGENTS.md`, or any
path). Preview by default; `--apply` writes a managed section:

```bash
export TOKENSLIM_CAPTURE=1        # start recording sessions first
tokenslim learn                   # preview proposed rules as a diff
tokenslim learn --apply           # write them to ./CLAUDE.md
tokenslim learn --sessions ~/.tokenslim/sessions --target AGENTS.md --apply
```

- `--sessions PATH` — capture directory or a single JSONL file (default:
  `~/.tokenslim/sessions`).
- `--target PATH` — rules file to update (default: `CLAUDE.md`).
- `--apply` — write the learned-rules section (otherwise preview only).

Capture is opt-in and strictly local (`TOKENSLIM_CAPTURE=1`); raw message
content is only recorded with `TOKENSLIM_CAPTURE_CONTENT=1`.

## refresh-pricing

Refreshes the local model token-pricing cache used by `perf`, `audit`, and
`estimate_cost`:

```bash
tokenslim refresh-pricing
tokenslim refresh-pricing --url https://example.com/pricing.json
```

## install

Registers the TokenSlim MCP server across agent platforms (Claude Code,
Cursor). Requires the separate `tokenslim-mcp` package:

```bash
pip install tokenslim-mcp
tokenslim install
```

!!! note
    This is unrelated to the repository's `install.sh` / `install.ps1`
    scripts, which install the `tokenslim` package itself — see
    [Quickstart](quickstart.md#install).

## memory

Placeholder — inspecting/querying the persistent agent memory store lands in a
later milestone. (The `ProjectMemoryStore` Python API already works.)

## update

Placeholder — self-update lands in a later milestone. Meanwhile:
`pipx upgrade tokenslim-ai` or `pip install -U tokenslim-ai` (re-running
`install.sh` / `install.ps1` does the same).
