"""Layered configuration.

Resolution order (lowest to highest precedence):

1. Built-in defaults (the :class:`Config` field defaults).
2. Environment variables prefixed ``TOKENSLIM_`` (e.g. ``TOKENSLIM_MIN_BYTES``).
3. Per-call overrides passed to :func:`compress`.

A config file layer is reserved for a later milestone; the env layer already
covers the M0 surface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Any

__all__ = ["Config", "load_config", "AUTO_QUERY"]

_ENV_PREFIX = "TOKENSLIM_"

# Sentinel :attr:`Config.query` value meaning "derive the relevance query from
# the last user message" (resolved in :func:`tokenslim.compress.compress`).
# ``query=None`` disables derivation; any other string is used verbatim.
AUTO_QUERY = "auto"


@dataclass(frozen=True)
class Config:
    """Resolved configuration for a compression run."""

    # Payloads smaller than this many bytes are passed through untouched.
    min_bytes: int = 200
    # A compressed block is kept only when it saves MORE than this many tokens
    # net (CCR marker cost included). Otherwise the block is reverted to its
    # original, so compression can never inflate a block (issue #117: a marker
    # can shrink characters while growing tokens).
    min_token_savings: int = 0
    # Model name used for token counting (selects the tokenizer backend).
    model: str | None = None
    # Master switch — when False, compress() is a no-op passthrough.
    enabled: bool = True
    # Compressors allowed to run. None means "all registered compressors".
    enabled_compressors: tuple[str, ...] | None = None
    # Emit CCR (compressed-content-record) sentinels describing what was
    # dropped so a downstream tool can reason about / reverse the compression.
    ccr: bool = True
    telemetry: bool = False

    # --- CCR store (reversibility) ---
    # Backend for storing dropped originals: "memory" (default), "sqlite", or "redis".
    ccr_backend: str = "memory"
    # SQLite database file (only used when ccr_backend == "sqlite").
    ccr_path: str = "tokenslim_ccr.sqlite3"
    # Optional time-to-live (seconds) for stored records; None = keep forever.
    ccr_ttl: int | None = None
    # Redis URL connection string (only used when ccr_backend == "redis").
    redis_url: str = "redis://localhost:6379/0"

    # --- SmartCrusher (JSON arrays) ---
    # Items kept from the head and tail of a crushed array.
    crush_keep_head: int = 5
    crush_keep_tail: int = 3
    # Only crush arrays with at least this many items.
    crush_min_items: int = 12
    # Optional hard budget for number of head+tail items to keep.
    max_items_after_crush: int | None = None
    # Maximum JSON nesting depth SmartCrusher walks; deeper subtrees are passed
    # through untouched so pathological nesting can't blow the stack (issue #116).
    max_json_depth: int = 200
    # Substrings (case-insensitive) that mark an item as must-keep.
    error_keywords: tuple[str, ...] = (
        "error",
        "fail",
        "failed",
        "failure",
        "exception",
        "traceback",
        "fatal",
        "critical",
        "panic",
        "denied",
        "timeout",
        "abort",
    )

    # --- LogCompressor / SearchCompressor ---
    # Context lines kept around a kept (important) log line.
    log_context: int = 1
    # Maximum number of distinct files kept by the search compressor.
    search_max_files: int = 20

    # --- Adaptive sizer ---
    # Fraction of items the sizer keeps at the reference size (0 < r <= 1).
    target_ratio: float = 0.2

    # --- DiffCompressor ---
    # Max files kept (most-changed first); rest are elided to the CCR store.
    diff_max_files: int = 10
    # Max hunks kept per file before extra hunks are elided.
    diff_max_hunks_per_file: int = 4
    # Trim each kept hunk's leading/trailing context lines to this many.
    diff_context: int = 2

    # --- HtmlExtractor ---
    # Keep hyperlink targets as "text (url)" instead of dropping the URL.
    html_keep_links: bool = False

    # --- Relevance (BM25) ---
    # Relevance query used by query-aware compressors (SmartCrusher JSON-match,
    # SearchCompressor BM25, LogCompressor, TextCompressor). Three modes:
    #   * "auto" (default): compress() derives the query from the last user
    #     message in the array (issue #124), truncated to a fixed char budget.
    #   * None: derivation OFF — the exact pre-#124 behavior (no relevance).
    #   * any other string: used verbatim as the query (caller-forced).
    query: str | None = AUTO_QUERY

    # --- images ---
    # Per-image token budget for reduce_image_tokens (None = provider sweet spot).
    image_max_tokens: int | None = None
    # Detail level for OpenAI-style image blocks: "auto", "low", or "high".
    image_detail: str = "auto"
    # --- Session capture (opt-in, local-only) ---
    # Record session events (compress runs, tool calls, outcomes) to local
    # JSONL for offline mining by `tokenslim learn`. OFF by default.
    capture: bool = False
    # Directory for session JSONL files; None means ~/.tokenslim/sessions.
    capture_path: str | None = None
    # Include raw message content in captured 'compress' events. OFF by
    # default for privacy — only token counts and content types are recorded.
    capture_content: bool = False
    # --- TabularCompressor ---
    # Data rows kept from the head and tail of a compressed CSV table.
    csv_keep_head: int = 5
    csv_keep_tail: int = 3
    # Max outlier rows (|z| > 2.5 or min/max holders in numeric columns) kept.
    csv_max_outliers: int = 5
    # --- Semantic cache ---
    # Minimum cosine similarity for a SemanticCache hit. Calibrated on real
    # embeddings (see semcache.py): 0.95 mis-serves 5-10% of near-miss pairs.
    semantic_cache_threshold: float = 0.96

    # --- Proxy (tokenslim proxy) ---
    # TCP port the compressing reverse proxy listens on.
    proxy_port: int = 8787
    # Upstream base URL requests are forwarded to (OpenAI-compatible API).
    upstream: str = "https://api.openai.com"

    # --- Output reduction ---
    # Brevity instructions appended to the system message so the model writes
    # shorter replies: "off" (default), "balanced", or "aggressive". When set,
    # integration wrappers apply it automatically after compress().
    output_reduction: str = "off"

    def merged(self, **overrides: Any) -> Config:
        """Return a copy with ``overrides`` applied, ignoring ``None`` values."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean) if clean else self


def _coerce(name: str, raw: str) -> Any:
    """Coerce an env string to the type of field ``name`` on :class:`Config`."""
    field_types = {f.name: f.type for f in fields(Config)}
    target = field_types.get(name, "str")
    target = str(target)
    if "bool" in target:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if "float" in target:
        return float(raw)
    if "int" in target:
        return int(raw)
    if "tuple" in target:
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    return raw


def load_config(env: dict[str, str] | None = None, **overrides: Any) -> Config:
    """Build a :class:`Config` from defaults, env vars, then per-call overrides."""
    source = os.environ if env is None else env
    env_values: dict[str, Any] = {}
    valid = {f.name for f in fields(Config)}
    for key, value in source.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        name = key[len(_ENV_PREFIX) :].lower()
        if name in valid:
            env_values[name] = _coerce(name, value)
    return Config().merged(**env_values).merged(**overrides)
