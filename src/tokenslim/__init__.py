"""tokenslim — context compression layer for LLM agents.

Compress tool outputs, logs, files and RAG payloads before they hit the model.
Local-first and reversible.

Quick start::

    from tokenslim import compress

    new_messages, stats = compress(messages)
    print(stats.ratio)
"""

from __future__ import annotations

from .ccr import (
    CCRMarker,
    find_markers,
    make_marker,
    parse_marker,
    strip_markers,
)
from .compress import BlockStat, CompressionStats, compress
from .compressors import (
    DiffCompressor,
    JsonMinifier,
    LogCompressor,
    SearchCompressor,
    SmartCrusher,
)
from .config import Config, load_config
from .detector import ContentType, DetectionResult, detect_content_type
from .formats import (
    MessageFormat,
    anthropic_to_openai,
    detect_format,
    openai_to_anthropic,
)
from .relevance import BM25Scorer, Scorer
from .retrieve import CCRContext, retrieve
from .router import ContentRouter, RouteResult, build_registry
from .sizer import compute_optimal_k
from .store import CCRStore, InMemoryCCRStore, SQLiteCCRStore, get_store
from .tokenizer import count_tokens, get_tokenizer

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "compress",
    "CompressionStats",
    "BlockStat",
    "Config",
    "load_config",
    "ContentType",
    "DetectionResult",
    "detect_content_type",
    "ContentRouter",
    "RouteResult",
    "build_registry",
    "SmartCrusher",
    "LogCompressor",
    "SearchCompressor",
    "DiffCompressor",
    "JsonMinifier",
    "compute_optimal_k",
    "BM25Scorer",
    "Scorer",
    "count_tokens",
    "get_tokenizer",
    "MessageFormat",
    "detect_format",
    "openai_to_anthropic",
    "anthropic_to_openai",
    # CCR / reversibility
    "CCRStore",
    "InMemoryCCRStore",
    "SQLiteCCRStore",
    "get_store",
    "retrieve",
    "CCRContext",
    "CCRMarker",
    "make_marker",
    "parse_marker",
    "find_markers",
    "strip_markers",
]
