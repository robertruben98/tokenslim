"""tokenslim — context compression layer for LLM agents.

Compress tool outputs, logs, files and RAG payloads before they hit the model.
Local-first and reversible.

Quick start::

    from tokenslim import compress

    new_messages, stats = compress(messages)
    print(stats.ratio)
"""

from __future__ import annotations

from .compress import BlockStat, CompressionStats, compress
from .config import Config, load_config
from .detector import ContentType, DetectionResult, detect_content_type
from .formats import (
    MessageFormat,
    anthropic_to_openai,
    detect_format,
    openai_to_anthropic,
)
from .router import ContentRouter, RouteResult
from .tokenizer import count_tokens, get_tokenizer

__version__ = "0.0.1"

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
    "count_tokens",
    "get_tokenizer",
    "MessageFormat",
    "detect_format",
    "openai_to_anthropic",
    "anthropic_to_openai",
]
