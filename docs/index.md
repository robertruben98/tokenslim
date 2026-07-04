# TokenSlim

**Context compression layer for LLM agents** — compress tool outputs, logs, files
& RAG payloads before they hit the model. Reversible, local-first, zero required
dependencies beyond `click` and `numpy`.

## What it does

Agents burn most of their context window (and your budget) on machine-generated
payloads: JSON tool results, build logs, grep output, diffs, HTML, CSV dumps.
TokenSlim walks an OpenAI- or Anthropic-style message array, detects the content
type of every large text block, and routes it to a purpose-built compressor:

```python
from tokenslim import compress

new_messages, stats = compress(messages)
print(f"saved {stats.saved_tokens} tokens ({stats.ratio:.0%})")
```

Everything lossy is **reversible**: dropped material goes into a
content-addressed CCR store and the output carries a
`<<ccr:HASH N reason>>` marker, so the exact original can be
[retrieved on demand](reversibility.md).

## Why it works

Machine-generated content is massively redundant. A 500-item JSON array of
near-identical rows carries the same information as its first five rows, its
last three, its anomalies, and a count. TokenSlim keeps exactly that — and
never drops rows containing errors, outliers, or query matches.

Typical savings are **60–95% of tokens** on compressible payloads, depending on
content type and redundancy. The bundled offline eval suite
(`uv run tokenslim evals`) measures both the savings *and* faithfulness —
dropped material must be recovered byte-exactly from the CCR store, and
must-keep rows (errors) must survive in the visible output:

```text
--- Accuracy Preservation Report ---
[PASS] json-orders: ratio=93.8% drops=1
[PASS] sre-log: ratio=92.4% drops=2
[PASS] code-search: ratio=0.0% drops=0
------------------------------------
```

- `json-orders` — a homogeneous JSON order array: **93.8% of tokens removed**,
  fully recoverable.
- `sre-log` — a noisy SRE incident log: **92.4% removed**, every error line kept.
- `code-search` — an already-compact grep result: passed through unchanged
  (0.0%), because compressing it would not have paid for itself. TokenSlim
  never inflates or mangles content that is already tight.

## Feature map

| Area | What you get | Docs |
| --- | --- | --- |
| Compression | Content-type detection + 8 real compressors (JSON, logs, search, diffs, code, HTML, CSV, prose) | [Compressors](compressors.md) |
| Reversibility | CCR markers + memory/SQLite/Redis stores + `retrieve()` | [Reversibility](reversibility.md) |
| Prefix caching | Message shaping for OpenAI/Google implicit caches, Anthropic `cache_control` injection | [Caching](caching.md) |
| Semantic caching | Embedding-similarity response cache with a calibrated 0.96 threshold + lexical safety guard | [Caching](caching.md) |
| Images | Vision token estimation + downscale/detail-flag planning | [Compressors](compressors.md#images) |
| Output tokens | Output-length prediction and output-brevity reduction | [Python API](api.md#output-prediction-reduction) |
| Integrations | OpenAI, Anthropic, LiteLLM, LangChain, Agno, Strands, reverse proxy, Docker | [Integrations](integrations.md) |
| CLI | `doctor`, `perf`, `evals`, `audit`, `proxy`, `wrap`, `learn`, … | [CLI](cli.md) |

## Install

```bash
pip install tokenslim
```

Or use the install scripts from the repository root (pipx when available,
`pip install --user` otherwise; `--with-extras` adds
`tokenslim[tokenizers,images,semantic]`):

```bash
sh install.sh --with-extras      # Linux / macOS
.\install.ps1 -WithExtras        # Windows PowerShell
```

Head to the [Quickstart](quickstart.md) next.

## Design principles

- **Local-first** — no network calls in the compression path by default.
  Telemetry is **opt-in**: nothing is sent unless you pass
  `Config(telemetry=True)`. Config wins — `TOKENSLIM_TELEMETRY=off` can
  additionally disable an opt-in, but the env var can never turn telemetry on
  against `Config(telemetry=False)`. See [Telemetry &amp; privacy](#telemetry-privacy).
- **Never break a request** — compressors return the original text on parse
  failure or when the output would not be smaller; integration wrappers and the
  proxy swallow their own errors.
- **Reversible by default** — CCR is on unless you turn it off.
- **Zero heavy deps** — tokenizers, tree-sitter, Pillow, and
  sentence-transformers are all optional extras; every integration is
  duck-typed and never imports the framework SDK.

## Telemetry &amp; privacy

Telemetry is **off by default** (`Config.telemetry` defaults to `False`) and
opt-in. When enabled it sends an anonymous, payload-free event (token counts,
compression ratio, version, model name, content types) on a single background
worker — never blocking compression. **Config always wins**; the environment
variable can only *additionally* disable it:

| `Config.telemetry` | `TOKENSLIM_TELEMETRY`      | Sends? |
| ------------------ | ------------------------- | ------ |
| `False` (default)  | anything / unset          | no     |
| `True`             | unset / `on` / `1` / `yes`| yes    |
| `True`             | `off` / `false` / `0` / `no` | no  |

You never need `TOKENSLIM_TELEMETRY=off` to stay private — that is already the
default. The env var exists to force telemetry off in an environment where some
code opts in with `Config(telemetry=True)`.
