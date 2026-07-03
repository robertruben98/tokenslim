# Compressors

Every large text block in a message array is classified by
`detect_content_type()` (rule-based, dependency-free, with a confidence score)
and routed by the `ContentRouter` to the compressor registered for that type.
Blocks smaller than `min_bytes` (default **200** UTF-8 bytes) are passed
through untouched, and every compressor returns the original text whenever
parsing fails or the output would not be smaller.

| Content type | Compressor (`name`) | Strategy |
| --- | --- | --- |
| `json` | `smartcrusher` | Crush homogeneous arrays: keep head/tail + anomalies, CCR the middle |
| `log` | `log-compressor` | Keep errors/warnings/summaries with context, dedup noise |
| `search` | `search-compressor` | Group grep/ripgrep hits by file, rank, cap files |
| `diff` | `diff-compressor` | Cap files and hunks by churn, trim context lines |
| `code` | `code-compressor` | AST-aware body elision (tree-sitter; safe no-op without it) |
| `html` | `html-extractor` | Extract readable text, drop markup |
| `csv` | `tabular` | Keep head/tail rows + statistical outliers |
| `markdown` / `text` | `text-compressor` | Extractive sentence/section summarisation |

A lossless `JsonMinifier` (`json-minify`) is also exported for parse → compact
re-serialisation without any dropping.

## SmartCrusher (JSON)

Crushes homogeneous JSON arrays: keeps the first `crush_keep_head` and last
`crush_keep_tail` items, drops the redundant middle, and appends a CCR sentinel
recording the dropped count and content hash. Items containing any
`error_keywords` substring are **never** dropped; rare status values and
statistical outliers survive; when `Config.query` is set, query-matching items
are kept too.

| Knob | Default | Meaning |
| --- | --- | --- |
| `crush_keep_head` | `5` | Items kept from the head of a crushed array |
| `crush_keep_tail` | `3` | Items kept from the tail |
| `crush_min_items` | `12` | Only crush arrays with at least this many items |
| `max_items_after_crush` | `None` | Optional hard budget for head+tail items |
| `error_keywords` | `error, fail, …` | Case-insensitive must-keep substrings |

## LogCompressor

Detects the build/test flavour (pytest, npm, cargo, jest, make, generic),
classifies each line by level, keeps errors/failures/warnings/summaries with a
context window, and conservatively dedups (lines differing only by an
id/address stay distinct).

| Knob | Default | Meaning |
| --- | --- | --- |
| `log_context` | `1` | Context lines kept around each important line |

## SearchCompressor

Parses grep/ripgrep `file:line:content` output (including `-C` context blocks,
Windows paths, and hyphenated filenames), groups hits by file to eliminate path
repetition, scores files by relevance, and caps the file count. When
`Config.query` is set, hits are re-ranked with the built-in BM25 scorer.

| Knob | Default | Meaning |
| --- | --- | --- |
| `search_max_files` | `20` | Maximum distinct files kept |
| `query` | `None` | Query string for BM25-aware ranking |

## DiffCompressor

Parses unified diffs, keeps at most `diff_max_files` files (most-changed
first), keeps the first/last + highest-churn hunks per file, trims each kept
hunk's context, and CCRs the rest — committing the compaction only when it
actually shrinks the diff.

| Knob | Default | Meaning |
| --- | --- | --- |
| `diff_max_files` | `10` | Max files kept, most-changed first |
| `diff_max_hunks_per_file` | `4` | Max hunks kept per file |
| `diff_context` | `2` | Context lines kept at each hunk edge |

## CodeCompressor

AST-aware (tree-sitter, Python/JavaScript) body elision: keeps signatures,
structure, and docstrings; CCRs the collapsed function bodies. Requires the
`code` extra — without the grammars it is a safe no-op.

## HtmlExtractor

Extracts the readable text from HTML documents and drops the markup.

| Knob | Default | Meaning |
| --- | --- | --- |
| `html_keep_links` | `false` | Keep hyperlink targets as `text (url)` instead of dropping URLs |

## TabularCompressor (CSV)

Keeps the header, the first `csv_keep_head` and last `csv_keep_tail` data rows,
plus up to `csv_max_outliers` statistically anomalous rows (|z| > 2.5 or
min/max holders in numeric columns). Elided rows go to the CCR store.

| Knob | Default | Meaning |
| --- | --- | --- |
| `csv_keep_head` | `5` | Data rows kept from the head |
| `csv_keep_tail` | `3` | Data rows kept from the tail |
| `csv_max_outliers` | `5` | Max outlier rows kept |

## TextCompressor (prose & Markdown)

Extractive summarisation: scores sentences/sections and keeps the
highest-signal ones, CCRing the rest. Sized by the shared adaptive sizer.

| Knob | Default | Meaning |
| --- | --- | --- |
| `target_ratio` | `0.2` | Fraction of items the adaptive sizer keeps at the reference size |

## Images

Vision inputs are billed by image *dimensions*, not payload size, so the wins
come from resizing to each provider's sweet spot or flipping OpenAI's `detail`
flag. Three layers, cheapest first:

```python
from tokenslim import estimate_image_tokens, plan_image_reduction, reduce_image_tokens

estimate_image_tokens(2048, 1536, provider="openai")   # published formulas
plan = plan_image_reduction(2048, 1536, provider="anthropic")
new_messages, stats = reduce_image_tokens(messages, provider="openai")
```

With the `images` extra (Pillow), `reduce_image_tokens` actually resizes and
re-encodes embedded base64 images; without it, dimensions are read from
PNG/JPEG/GIF headers and the plans are only reported in the stats.

| Knob | Default | Meaning |
| --- | --- | --- |
| `image_max_tokens` | `None` | Per-image token budget (`None` = provider sweet spot) |
| `image_detail` | `auto` | OpenAI `detail` flag: `auto`, `low`, or `high` |

## Configuration

`Config` is a frozen dataclass resolved in layers — built-in defaults, then
`TOKENSLIM_*` environment variables, then per-call overrides. Every field maps
to an env var automatically (`min_bytes` → `TOKENSLIM_MIN_BYTES`), and
`tokenslim doctor` prints the fully resolved configuration.

Global knobs that shape all compressors:

| Knob | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Master switch — `false` makes `compress()` a passthrough |
| `min_bytes` | `200` | Skip blocks smaller than this many UTF-8 bytes |
| `model` | `None` | Model name for token counting (selects tokenizer backend) |
| `enabled_compressors` | `None` (all) | Comma-separated allowlist of compressor names |
| `ccr` | `true` | Emit CCR markers for dropped material (see [Reversibility](reversibility.md)) |
| `target_ratio` | `0.2` | Adaptive sizer keep-fraction |
| `query` | `None` | Query string for relevance-aware compression |

```python
from tokenslim import Config, compress

out, stats = compress(
    messages,
    options=Config(enabled_compressors=("smartcrusher", "log-compressor")),
)
```

## Extending the router

Register a custom compressor at runtime without forking:

```python
from tokenslim import ContentRouter, ContentType

router = ContentRouter()
router.register(ContentType.TEXT, "my-compressor", my_callable)
result = router.route(text)   # RouteResult(text, content_type, compressor, changed, ...)
```

A compressor is any callable with the `(text, content_type) -> str` signature.
