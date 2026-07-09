# Caching

TokenSlim ships two independent, opt-in caching layers:

- **Prefix-cache awareness** — shape requests so the *provider's* prompt cache
  (OpenAI, Google, Anthropic) actually engages, cutting the price of repeated
  prefixes.
- **Semantic response cache** — skip the LLM call entirely when a new prompt is
  semantically equivalent to one you already answered.

Neither is wired into `compress()` — you call them explicitly.

## Prefix cache

OpenAI and Google Gemini run *implicit* prefix caches: a request whose leading
tokens are byte-identical to a recent one gets the cached prefix discount
automatically. Anthropic uses explicit `cache_control` breakpoints. Either way
the enemy is the same: volatile bytes (timestamps, UUIDs, request ids) early in
the prompt invalidate everything after them.

### One call does the shaping

```python
from tokenslim import optimize_for_prefix_cache

optimized, report = optimize_for_prefix_cache(messages, provider="openai")
print(report.cacheable, report.stable_prefix_tokens)
for hint in report.hints:
    print("-", hint)
```

`optimize_for_prefix_cache` (providers: `"openai"`, `"google"`, `"anthropic"`)
is non-mutating and applies, in order:

1. `stabilize_message_order` — hoists `system`/`developer` messages to the
   front, preserving conversation order otherwise (the only provably safe move).
2. `normalize_dynamic_content` on the **system prompt only** — volatile
   substrings are rewritten to placeholders so the prefix stays byte-stable:
   UUIDs → `<UUID>`, API keys → `<TOKEN>`, ISO/epoch timestamps →
   `<TIMESTAMP>`, long hex → `<HASH>`. Conversation turns are never rewritten.
3. Counts the stable prefix and compares it with the provider minimum
   (~1024 tokens for all three; Gemini 2.5 Pro and Claude Haiku need 2048) —
   `report.cacheable` tells you whether the cache will engage, and
   `report.hints` says what to move where when it will not.

For Anthropic the same call also injects `cache_control` breakpoints via
`insert_anthropic_cache_control`.

### The pieces individually

```python
from tokenslim import (
    find_volatile_spans,          # report cache-busting substrings, no rewrite
    insert_anthropic_cache_control,  # add {"type": "ephemeral"} breakpoints
    normalize_dynamic_content,    # rewrite volatile substrings to placeholders
    stabilize_message_order,      # hoist system/developer messages
)

spans = find_volatile_spans(system_prompt)   # VolatileSpan(kind, start, end, text)
messages2, system2 = insert_anthropic_cache_control(messages, system=system_prompt)
```

`insert_anthropic_cache_control(messages, system=None, min_bytes=2048,
max_breakpoints=4)` marks the system prompt first, then scans messages
backward, adding `{"cache_control": {"type": "ephemeral"}}` to the last text
block of the largest qualifying items.

!!! note "normalize_dynamic_content changes what the model sees"
    Placeholder substitution is aggressive by design — it will also rewrite
    meaningful dates or git SHAs. `optimize_for_prefix_cache` therefore
    normalizes only the designated-stable segment; apply it to other content
    deliberately, not blindly.

## Semantic cache

`SemanticCache` returns a previously stored response when a new prompt is an
exact match or embedding-cosine-close to a cached one:

```python
from tokenslim import SemanticCache, SentenceTransformerEmbedder

cache = SemanticCache(SentenceTransformerEmbedder())   # pip install "tokenslim-ai[semantic]"

if (hit := cache.get(prompt)) is not None:
    return hit.response          # hit.similarity, hit.key_prompt for auditing
response = call_llm(prompt)
cache.put(prompt, response)
```

It is an LRU cache (`max_entries=1024`), never raises from `get`/`put`
(embedder failures degrade to a miss), and takes any embedding backend that
satisfies the tiny `Embedder` protocol: `embed(texts: list[str]) ->
list[list[float]]`.

### Why the threshold is 0.96 (and why there is a guard)

The defaults come from a calibration experiment on real embeddings
(RTX 5070 Ti, three sentence-transformers models, 120 hand-written EN+ES
prompt pairs):

- The commonly proposed cosine threshold of **0.95 serves a wrong cached
  answer on 5–10% of adversarial near-miss pairs** (all-MiniLM-L6-v2 5%,
  all-mpnet-base-v2 10%, bge-small-en-v1.5 7.5%).
- The surviving false positives are exactly the dangerous class: **date swaps**
  ("June 5th" vs "July 5th") and **polarity flips** ("enable dark mode" vs
  "disable dark mode") score 0.96–0.98 — embeddings barely register them.
- No cosine threshold reaches zero near-miss false positives at useful recall:
  at the 0.98–0.99 needed for zero FPs, recall collapses to 0–5%.

So `SemanticCache` defaults to **threshold 0.96** *plus* a cheap **lexical
guard**: numeric/date/month tokens must match exactly, and antonym/negation
flips reject the candidate. Matching is **bilingual (EN + ES)** on
accent-stripped tokens, so both English polarity words (`enable`/`disable`,
`add`/`remove`, a one-sided `not` — English contractions like `don't` are
normalized to `not`) and inflected Spanish verbs (`crea`/`borra`,
`activa`/`desactiva`, `sube`/`baja`, a one-sided `no`/`nunca`/`sin`, …) are
caught. This closes the known false positive *"crea el usuario admin"* vs
*"borra el usuario admin"* (cosine ≈ 0.969 ≥ 0.96, opposite action). The guard
killed every observed high-similarity false positive in the experiment; its
failure mode is an occasional extra cache miss, never a wrong answer. Disable it
(`guard=False`) only if wrong-but-similar answers are acceptable.

`Config.semantic_cache_threshold` (env: `TOKENSLIM_SEMANTIC_CACHE_THRESHOLD`)
carries the same 0.96 default for your own wiring. Thresholds are **not**
transferable between embedding models — recalibrate if you swap models. The
recommended model is `BAAI/bge-small-en-v1.5`, the experiment's best
safety/recall trade-off at 0.96.

### Remote GPU embeddings with HTTPEmbedder

Embedding locally pulls in sentence-transformers + torch. If you have a GPU
box elsewhere (or want to keep agent processes lean), point `HTTPEmbedder` at
a remote embedding service instead — the contract is a single endpoint:

```text
POST {base_url}/embed
{"texts": ["...", "..."]}
->  {"embeddings": [[...], [...]]}      # one vector per input text
```

```python
from tokenslim import HTTPEmbedder, SemanticCache

cache = SemanticCache(HTTPEmbedder("http://gpu-box:8000", timeout=10.0))
```

A matching server is a few lines of FastAPI on the GPU machine:

```python
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")

@app.post("/embed")
def embed(body: dict):
    return {"embeddings": model.encode(body["texts"]).tolist()}
```

Network or protocol failures raise `OSError` from `HTTPEmbedder.embed`;
inside `SemanticCache` that simply becomes a cache miss / skipped insert, so
the agent keeps working when the GPU box is down.
