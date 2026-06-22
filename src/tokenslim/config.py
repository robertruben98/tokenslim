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

__all__ = ["Config", "load_config"]

_ENV_PREFIX = "TOKENSLIM_"


@dataclass(frozen=True)
class Config:
    """Resolved configuration for a compression run."""

    # Payloads smaller than this many bytes are passed through untouched.
    min_bytes: int = 200
    # Model name used for token counting (selects the tokenizer backend).
    model: str | None = None
    # Master switch — when False, compress() is a no-op passthrough.
    enabled: bool = True
    # Compressors allowed to run. None means "all registered compressors".
    enabled_compressors: tuple[str, ...] | None = None
    # Emit CCR (compressed-content-record) sentinels describing what was
    # dropped so a downstream tool can reason about / reverse the compression.
    ccr: bool = True
    # Anonymous aggregate telemetry, ON by default (TOKENSLIM_TELEMETRY=off to
    # disable). The shipped sink is a local no-op stub — no network egress.
    telemetry: bool = True

    # --- CCR store (reversibility) ---
    # Backend for storing dropped originals: "memory" (default) or "sqlite".
    ccr_backend: str = "memory"
    # SQLite database file (only used when ccr_backend == "sqlite").
    ccr_path: str = "tokenslim_ccr.sqlite3"
    # Optional time-to-live (seconds) for stored records; None = keep forever.
    ccr_ttl: int | None = None

    # --- SmartCrusher (JSON arrays) ---
    # Items kept from the head and tail of a crushed array.
    crush_keep_head: int = 5
    crush_keep_tail: int = 3
    # Only crush arrays with at least this many items.
    crush_min_items: int = 12
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

    # --- Relevance (BM25) ---
    # Optional query string; when set, compressors can rank by relevance to it.
    query: str | None = None

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
