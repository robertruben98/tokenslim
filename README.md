# tokenslim

Context compression layer for LLM agents — compress tool outputs, logs, files & RAG before they hit the model. Reversible, local-first.

## Install

```bash
pip install -e ".[dev]"        # development
pip install tokenslim          # (once published)
```

Optional extras: `tokenizers` (accurate tiktoken counts), `ml` (Magika-based detection).
Python 3.10+.

## Quick start

```python
from tokenslim import compress

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "tool", "tool_call_id": "t1", "content": pretty_printed_json_blob},
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
    repetition, scores by relevance, and caps the number of files.
- **CCR** — compressed-content records: tiny sentinels recording what was
  dropped + a content hash, so compression stays auditable.
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

Per call: `compress(messages, min_bytes=0, model="gpt-4o")`.

## Status

**M1 Core compressors.** JSON (SmartCrusher), build/test logs, and search
results have real algorithms. Diff, AST-aware code, and extractive text
compression — plus query-aware relevance scoring and statistical outlier
preservation — are tracked for a later milestone and currently pass through.

## License

Apache-2.0.
