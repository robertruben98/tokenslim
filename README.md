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

Per call: `compress(messages, min_bytes=0, model="gpt-4o")`.

## Status

**M0 Foundation.** For M0 the only real compressor is JSON whitespace
minification; other content types pass through unchanged. Real compression
algorithms (logs, diffs, search results, code) land in M1 behind the same
router/registry.

## License

Apache-2.0.
