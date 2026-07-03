# Python API

Everything documented here is importable from the top-level package:

```python
import tokenslim                      # tokenslim.__version__
from tokenslim import compress, Config, retrieve, SemanticCache, ...
```

The public surface is exactly `tokenslim.__all__` — anything you have to
import from a submodule is internal and may change without notice.

## Core compression

| Export | What it is |
| --- | --- |
| `compress(messages, options=None, **overrides)` | The entry point — returns `(new_messages, stats)`; never mutates its input |
| `CompressionStats` | `orig_tokens`, `new_tokens`, `ratio`, `saved_tokens`, `blocks`, `store` |
| `BlockStat` | Per-block detail: message index, path, content type, confidence, compressor, token delta |
| `Config` | Frozen dataclass of every knob; `Config.merged(**overrides)` returns a copy |
| `load_config(env=None, **overrides)` | Defaults → `TOKENSLIM_*` env → overrides |

See [Quickstart](quickstart.md) and [Compressors — Configuration](compressors.md#configuration).

## Detection & routing

| Export | What it is |
| --- | --- |
| `detect_content_type(text)` | Rule-based classifier → `DetectionResult(content_type, confidence)` |
| `ContentType` | `str` enum: `json`, `html`, `code`, `log`, `diff`, `search`, `csv`, `markdown`, `text` |
| `DetectionResult` | Frozen result of a detection |
| `ContentRouter` | Detect → registry lookup → compress; `router.register(type, name, fn)` for custom compressors |
| `RouteResult` | `text`, `content_type`, `confidence`, `compressor`, `changed` |
| `build_registry(config, store)` | The default `{ContentType: (name, compressor)}` mapping |

## Compressor classes

All follow the same shape — construct with an optional `Config` and `CCRStore`,
call with `(text, content_type) -> str`, never raise (they return the input on
any failure). Full behavior in [Compressors](compressors.md).

| Export | `name` (allowlist key) |
| --- | --- |
| `SmartCrusher` | `smartcrusher` |
| `LogCompressor` | `log-compressor` |
| `SearchCompressor` | `search-compressor` |
| `DiffCompressor` | `diff-compressor` |
| `HtmlExtractor` | `html-extractor` |
| `TabularCompressor` | `tabular` |
| `JsonMinifier` | `json-minify` (lossless) |

## Reversibility (CCR)

Full guide: [Reversibility](reversibility.md).

| Export | What it is |
| --- | --- |
| `retrieve(hash, store=None, config=None)` | Fetch a dropped original back — pass `stats.store` or a persistent-backend config |
| `CCRContext` | Scoped retrieval: only serves hashes for markers `track()`-ed from the conversation |
| `CCRStore` | The store protocol: `put(original) -> hash`, `get(hash) -> str \| None` |
| `InMemoryCCRStore` / `SQLiteCCRStore` | Backends (`redis` is selected via `Config.ccr_backend`) |
| `get_store(config)` | Backend factory dispatched on `Config.ccr_backend` |
| `CCRMarker` | Parsed `<<ccr:HASH N reason>>` marker |
| `make_marker` / `parse_marker` / `find_markers` / `strip_markers` | Marker helpers |

## Prefix caching

Full guide: [Caching — Prefix cache](caching.md#prefix-cache).

| Export | What it is |
| --- | --- |
| `optimize_for_prefix_cache(messages, provider, model=None)` | One-call shaping → `(optimized, PrefixCacheReport)` |
| `stabilize_message_order(messages)` | Hoist `system`/`developer` messages to the front |
| `normalize_dynamic_content(text)` | Rewrite UUIDs/keys/timestamps/hashes to stable placeholders |
| `find_volatile_spans(text)` | Report cache-busting substrings without rewriting |
| `insert_anthropic_cache_control(messages, system=None, min_bytes=2048, max_breakpoints=4)` | Add `{"type": "ephemeral"}` breakpoints → `(messages, system)` |
| `PrefixCacheReport` / `VolatileSpan` | Result dataclasses |

## Semantic cache

Full guide: [Caching — Semantic cache](caching.md#semantic-cache).

| Export | What it is |
| --- | --- |
| `SemanticCache(embedder, threshold=0.96, max_entries=1024, guard=True)` | LRU response cache keyed by embedding similarity + lexical guard |
| `CacheHit` | `response`, `similarity`, `key_prompt` |
| `Embedder` | Protocol: `embed(texts: list[str]) -> list[list[float]]` |
| `SentenceTransformerEmbedder(model_name="BAAI/bge-small-en-v1.5")` | Local embeddings (`semantic` extra) |
| `HTTPEmbedder(base_url, timeout=10.0)` | Remote embedding service (`POST {base_url}/embed`) |

## Images

Full guide: [Compressors — Images](compressors.md#images).

| Export | What it is |
| --- | --- |
| `estimate_image_tokens(width, height, provider, detail="auto")` | Published token formulas for OpenAI/Anthropic/Google |
| `plan_image_reduction(width, height, provider, target_tokens=None, detail="auto")` | Cheapest useful reduction → `ImagePlan` |
| `reduce_image_tokens(messages, provider, options=None, **overrides)` | Apply plans to embedded base64 images → `(messages, ImageStats)` |
| `ImagePlan` / `ImageStats` | Result dataclasses |

## Output prediction & reduction

Predict how long the reply will be (to set `max_tokens`) and ask the model to
be brief (to spend fewer output tokens):

| Export | What it is |
| --- | --- |
| `predict_output_tokens(messages, model=None, *, weights=None)` | Linear estimate → `OutputPrediction` (`tokens`, `low`/`high`, `confidence`) |
| `suggest_max_tokens(prediction, safety=1.5, floor=64, cap=16384)` | Turn a prediction into a `max_tokens` value |
| `extract_output_features(messages, model=None)` | The predictor's feature vector, for your own models |
| `apply_output_reduction(messages, level="balanced", model=None)` | Append brevity instructions to the system message → `(messages, report)` |
| `OUTPUT_REDUCTION_LEVELS` | `{"off", "balanced", "aggressive"}` → instruction blocks |
| `BALANCED_INSTRUCTIONS` / `AGGRESSIVE_INSTRUCTIONS` | The raw instruction text |
| `measure_output_delta(baseline_response, reduced_response, model=None)` | Tokens saved on a reply pair → `OutputDelta` |
| `OutputPrediction` / `OutputReductionReport` / `OutputDelta` | Result dataclasses |

`Config.output_reduction` (`"off"`/`"balanced"`/`"aggressive"`) makes the
[integration wrappers](integrations.md) apply reduction automatically.

## Integrations

Full guide: [Integrations](integrations.md).

| Export | Framework |
| --- | --- |
| `with_tokenslim(client, config=None)` | OpenAI / Anthropic SDK clients (sync + async) |
| `TokenSlimLiteLLMCallback` | LiteLLM proxy callback (`pre_call_hook` / `async_pre_call_hook`) |
| `wrap_chat_model(model, config=None)` | LangChain chat models (`invoke`/`ainvoke`/`stream`) |
| `compress_messages(messages, config=None)` | LangChain message lists (returns new objects) |
| `compress_documents(docs, config=None, query=None)` | LangChain documents, BM25 query-aware |
| `tokenslim_agno_tool_hook` | Agno `tool_hooks` entry |
| `wrap_agno_model(model, config=None)` | Agno model wrapper |
| `TokenSlimStrandsHooks` | Strands hook provider |
| `compress_tool_output(text, config=None)` | Framework-agnostic `str -> str`; never raises |

## Reverse proxy

| Export | What it is |
| --- | --- |
| `run_proxy(port=None, upstream=None)` | Blocking entry point behind `tokenslim proxy` |
| `make_proxy_server(...)` | Build the server without serving (for tests/embedding) |
| `TokenSlimProxyServer` | The `ThreadingHTTPServer` subclass itself |

## Message formats

| Export | What it is |
| --- | --- |
| `detect_format(messages)` | → `MessageFormat` (`openai` / `anthropic` shapes) |
| `MessageFormat` | `str` enum of dialects |
| `openai_to_anthropic(messages)` / `anthropic_to_openai(messages)` | Dialect converters |

## Tokens, pricing & sizing

| Export | What it is |
| --- | --- |
| `count_tokens(text, model=None)` | Heuristic counter; uses `tiktoken` automatically when installed |
| `get_tokenizer(model=None)` | The underlying (cached) tokenizer object |
| `estimate_cost(model, input_tokens, output_tokens=0)` | USD estimate from the local pricing table |
| `refresh_pricing(url)` | Refresh the pricing cache (also: `tokenslim refresh-pricing`) |
| `compute_optimal_k(n, target_ratio)` | Shared exponential-decay keep-budget used by the compressors |
| `BM25Scorer` / `Scorer` | Zero-dependency query relevance scorer + its protocol |

## Evals, audit & sessions

| Export | What it is |
| --- | --- |
| `run_suite(fixtures=None, config=None)` | The [`tokenslim evals`](cli.md#evals) harness, programmatically |
| `perf_report(fixtures=None, config=None, model=None)` | The demo savings report as a string |
| `run_audit(requests, config=None, model=None, answers=False)` | Baseline-vs-optimized replay → `AuditReport` |
| `parse_requests(text)` / `render_audit_report(report)` | Audit input/output helpers |
| `AuditReport` / `AuditRow` | Audit result dataclasses |
| `SessionCapture` / `get_capture(config=None)` / `read_sessions(path=None)` | Opt-in local JSONL session capture |
| `analyze_sessions(events)` / `propose_rules(findings)` / `apply_rules(block, target, dry_run=True)` / `Finding` | The [`tokenslim learn`](cli.md#learn) pipeline, programmatically |

## Memory & shared context

| Export | What it is |
| --- | --- |
| `ProjectMemoryStore` | Project-scoped persistent memory (`.tokenslim/` SQLite + embedding/BM25 search) |
| `SharedContext` | Deduplicating context stash for inter-agent communication |
