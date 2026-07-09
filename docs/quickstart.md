# Quickstart

## Install

```bash
pip install tokenslim-ai
```

Optional extras (all independent — install what you need):

| Extra | Enables |
| --- | --- |
| `tokenizers` | Accurate OpenAI token counts via `tiktoken` (heuristic estimator otherwise) |
| `images` | Actual image resizing in `reduce_image_tokens` via Pillow (planning works without it) |
| `semantic` | `SentenceTransformerEmbedder` for the semantic cache |
| `code` | AST-aware code compression via tree-sitter (Python/JavaScript) |
| `ml` | Magika ML-based content-type detection |
| `langchain` | Pins `langchain-core` (the wrappers are duck-typed and work without it) |
| `agno` / `strands` | The respective agent-framework SDKs |
| `docs` | mkdocs-material, for building this documentation site |

```bash
pip install "tokenslim-ai[tokenizers,images,semantic]"
```

Or run the install script from the repository root — it prefers `pipx` when
installed, falls back to `pip install --user`, and finishes by running
`tokenslim doctor` to verify the install:

```bash
sh install.sh                 # plain install
sh install.sh --with-extras   # adds tokenizers,images,semantic
```

Windows PowerShell equivalent:

```powershell
.\install.ps1              # plain install
.\install.ps1 -WithExtras  # adds tokenizers,images,semantic
```

## Compress in five lines

```python
from tokenslim import compress

new_messages, stats = compress(messages)          # OpenAI/Anthropic-style array
print(f"{stats.orig_tokens} -> {stats.new_tokens} tokens ({stats.ratio:.0%} saved)")
# per-block detail in stats.blocks; dropped originals in stats.store
```

`compress()` never mutates its input. Per-call overrides go straight into the
config: `compress(messages, min_bytes=0, model="gpt-4o", query="user question")`.

A frozen [`Config`](compressors.md#configuration) resolves as
**defaults → `TOKENSLIM_*` env vars → per-call overrides**:

```python
from tokenslim import Config, compress

cfg = Config(ccr_backend="sqlite", crush_keep_head=8)
out, stats = compress(messages, options=cfg)
```

## Wrap a client instead

If you would rather never call `compress()` yourself, wrap your SDK client —
every request's `messages` is compressed transparently
(see [Integrations](integrations.md) for LiteLLM, LangChain, Agno and Strands):

```python
from openai import OpenAI
from tokenslim import with_tokenslim

client = with_tokenslim(OpenAI())
client.chat.completions.create(model="gpt-4o", messages=messages)
```

## CLI tour

The `tokenslim` console script ships with the package
(full reference: [CLI](cli.md)):

```bash
tokenslim doctor                       # environment + resolved configuration
tokenslim perf < messages.json         # savings report for a message array
tokenslim evals                        # offline accuracy-preservation suite
tokenslim audit < requests.json        # baseline-vs-optimized replay + diff
tokenslim proxy --port 8787            # OpenAI-compatible compressing proxy
tokenslim wrap -- my-agent --flag      # run any command with TOKENSLIM_ENABLED=true
tokenslim init                         # interactive .env setup wizard
```

Try the savings report right away:

```bash
python - <<'EOF' | tokenslim perf --model gpt-4o
import json
rows = [{"id": i, "role": "member"} for i in range(500)]
print(json.dumps([{"role": "tool", "content": json.dumps({"users": rows})}]))
EOF
```

## What next

- [Compressors](compressors.md) — what each content type gets and every tuning knob.
- [Reversibility](reversibility.md) — retrieve the dropped originals.
- [Caching](caching.md) — prefix-cache shaping and the semantic response cache.
- [Integrations](integrations.md) — frameworks, the proxy, and Docker.
