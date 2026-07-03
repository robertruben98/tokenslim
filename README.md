# tokenslim

[![CI](https://github.com/robertruben98/tokenslim/actions/workflows/ci.yml/badge.svg)](https://github.com/robertruben98/tokenslim/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)

Context compression layer for LLM agents — compress tool outputs, logs, files & RAG before they hit the model. Reversible, local-first.

## Install

```bash
pip install -e ".[dev]"        # development
pip install tokenslim          # (once published)
```

Optional extras: `tokenizers` (accurate tiktoken counts), `code` (tree-sitter AST-aware code compression), `redis` (distributed CCR backend), `ml` (Magika-based detection).
Python 3.10+.

Or use the install scripts (pipx when available, `pip install --user` otherwise;
both finish by running `tokenslim doctor` to verify):

```bash
sh install.sh --with-extras      # Linux / macOS ("tokenslim[tokenizers,images,semantic]")
.\install.ps1 -WithExtras        # Windows PowerShell
```

**Documentation:** <https://robertruben98.github.io/tokenslim/> — quickstart,
compressor & config reference, CCR reversibility, caching, integrations, CLI
and Python API (built with mkdocs-material from `docs/`).

## Quick start

```python
from tokenslim import compress

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "tool", "tool_call_id": "t1", "content": '{"users": [' + ", ".join('{"id": %d, "role": "member"}' % i for i in range(500)) + "]}"},
]

new_messages, stats = compress(messages)
print(f"saved {stats.saved_tokens} tokens ({stats.ratio:.0%})")
```

`compress()` walks an OpenAI- or Anthropic-style message array, routes each
large text block to a content-type-specific compressor, and returns the
rewritten messages plus a `CompressionStats` (orig/new tokens, ratio, per-block
detail). The input is never mutated.

## How it works

```
messages ─▶ format detect ─▶ per block:
                              detect content-type ─▶ ContentRouter ─▶ compressor
                                                       │
                                                       └─ skip if < min_bytes
```

- **tokenizer** — `count_tokens(text, model)`. Dependency-free heuristic by
  default; uses `tiktoken` automatically when installed.
- **detector** — rule/regex classification into `{json, code, log, diff,
  search, markdown, text}` with a confidence score. Pluggable for an ML
  detector later.
- **router** — registry mapping content-type → compressor. Skips payloads below
  a byte threshold and aggregates stats.
- **compressors** — real algorithms behind the router:
  - **SmartCrusher** (JSON) — crushes homogeneous arrays: keeps first/last N,
    drops the redundant middle, and appends a CCR sentinel with the dropped
    count + content hash. Never drops items containing error keywords; preserves
    rare status values (anomalies survive).
  - **LogCompressor** — detects build/test flavour (pytest/npm/cargo/jest/make/
    generic), classifies each line by level, keeps errors/failures/warnings/
    summaries with a context window, conservatively dedups (keeps lines that
    differ by an id/address distinct).
  - **SearchCompressor** — parses grep/ripgrep `file:line:content` (+ `-C`
    context, Windows paths, hyphen filenames), groups hits by file to kill path
    repetition, scores by relevance, and caps the number of files. Optional
    query-aware BM25 re-ranking when `config.query` is set.
  - **DiffCompressor** — parses unified diffs, caps files by change density,
    keeps the first/last + highest-churn hunks per file, trims context lines,
    and CCRs the rest — committing the compaction only when it shrinks the diff
    below ~0.8 of the original.
  - **JsonMinifier** — lossless parse → compact re-serialise; keeps the original
    if not shorter. Usable as a pre-pass or for non-array JSON.
  - **CodeCompressor** — AST-aware (tree-sitter, Python/JavaScript) body elision:
    keeps signatures, structure and docstrings, CCRs the collapsed function
    bodies. Requires the `code` extra; falls back to a safe no-op when the
    tree-sitter grammars aren't installed.
  - **TextCompressor** — extractive prose/markdown summarisation: scores
    sentences/sections and keeps the highest-signal ones, CCRing the rest.
- **sizer** — `compute_optimal_k(n, target_ratio)`: shared exponential-decay
  budget helper used to size diff/log/search selections (monotonic, no
  k=1 overshoot).
- **relevance** — `BM25Scorer`: zero-dependency query-aware scorer (behind a
  `Scorer` protocol so an embedding scorer can drop in later).
- **CCR** — compress-cache-retrieve: dropped material is cached in a
  content-addressed store (`InMemoryCCRStore` / `SQLiteCCRStore`) and the output
  carries a canonical `<<ccr:HASH N reason>>` marker. `retrieve(hash)` returns
  the exact original, so lossy compression is reversible. `CCRContext` scopes
  retrieval to markers actually seen in a conversation.
- **config** — layered: built-in defaults < `TOKENSLIM_*` env vars < per-call
  overrides.
- **formats** — detect OpenAI vs Anthropic message shapes and convert between
  them.

## Configuration

| Env var | Default | Meaning |
| --- | --- | --- |
| `TOKENSLIM_MIN_BYTES` | `200` | Skip blocks smaller than this |
| `TOKENSLIM_MODEL` | _(none)_ | Model name for token counting |
| `TOKENSLIM_ENABLED` | `true` | Master on/off switch |
| `TOKENSLIM_ENABLED_COMPRESSORS` | _(all)_ | Comma-separated allowlist |
| `TOKENSLIM_CCR` | `true` | Emit CCR sentinels for dropped material |
| `TOKENSLIM_CRUSH_KEEP_HEAD` | `5` | SmartCrusher head items kept |
| `TOKENSLIM_CRUSH_KEEP_TAIL` | `3` | SmartCrusher tail items kept |
| `TOKENSLIM_ERROR_KEYWORDS` | _(builtin)_ | Comma-separated must-keep keywords |
| `TOKENSLIM_SEARCH_MAX_FILES` | `20` | Max files kept by SearchCompressor |
| `TOKENSLIM_DIFF_MAX_FILES` | `10` | Max files kept by DiffCompressor |
| `TOKENSLIM_DIFF_MAX_HUNKS_PER_FILE` | `4` | Max hunks kept per file |
| `TOKENSLIM_DIFF_CONTEXT` | `2` | Context lines kept per hunk edge |
| `TOKENSLIM_TARGET_RATIO` | `0.2` | Adaptive sizer keep fraction |
| `TOKENSLIM_QUERY` | _(none)_ | Query for BM25-aware ranking |
| `TOKENSLIM_CCR_BACKEND` | `memory` | CCR store backend (`memory` / `sqlite` / `redis`) |
| `TOKENSLIM_CCR_PATH` | `tokenslim_ccr.sqlite3` | SQLite file (sqlite backend) |
| `TOKENSLIM_CCR_TTL` | _(none)_ | Seconds before a stored original expires |

Per call: `compress(messages, min_bytes=0, model="gpt-4o")`.

## Reversibility (CCR)

```python
from tokenslim import compress, retrieve, Config

out, stats = compress(messages, options=Config(ccr_backend="sqlite"))
# out carries <<ccr:HASH N reason>> markers; the dropped originals live in
# stats.store, retrievable on demand:
from tokenslim.ccr import find_markers
for marker in find_markers(out[0]["content"]):
    original = retrieve(marker.hash, store=stats.store)
```

## Status

**Compressors complete and router-wired.** JSON (SmartCrusher), build/test logs,
search results, unified diffs, AST-aware code (tree-sitter), and extractive
text/markdown all have real algorithms; a lossless JSON minifier, a shared
adaptive sizer, and a BM25 relevance scorer round out the engine. All lossy drops
are reversible via the CCR store, available in-memory, SQLite, and distributed
Redis backends. Still open: SmartCrusher's statistical-outlier / query-anchor
keep and a dedicated stack-trace state machine.

## License

Apache-2.0.
