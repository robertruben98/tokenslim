"""tokenslim — context compression layer for LLM agents.

Compress tool outputs, logs, files and RAG payloads before they hit the model.
Local-first and reversible.

Quick start::

    from tokenslim import compress

    new_messages, stats = compress(messages)
    print(stats.ratio)
"""

from __future__ import annotations

from .audit import (
    AuditReport,
    AuditRow,
    parse_requests,
    render_audit_report,
    run_audit,
)
from .cache import (
    PrefixCacheReport,
    VolatileSpan,
    find_volatile_spans,
    insert_anthropic_cache_control,
    normalize_dynamic_content,
    optimize_for_prefix_cache,
    stabilize_message_order,
)
from .capture import SessionCapture, get_capture, read_sessions
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
    HtmlExtractor,
    JsonMinifier,
    LogCompressor,
    SearchCompressor,
    SmartCrusher,
    TabularCompressor,
)
from .config import Config, load_config
from .context import SharedContext
from .detector import ContentType, DetectionResult, detect_content_type
from .evals import perf_report, run_suite
from .formats import (
    MessageFormat,
    anthropic_to_openai,
    detect_format,
    openai_to_anthropic,
)
from .images import (
    ImagePlan,
    ImageStats,
    estimate_image_tokens,
    plan_image_reduction,
    reduce_image_tokens,
)
from .integrations import TokenSlimLiteLLMCallback, with_tokenslim
from .integrations_agents import (
    TokenSlimStrandsHooks,
    compress_tool_output,
    tokenslim_agno_tool_hook,
    wrap_agno_model,
)
from .integrations_langchain import compress_documents, compress_messages, wrap_chat_model
from .learn import Finding, analyze_sessions, apply_rules, propose_rules
from .memory import ProjectMemoryStore
from .outputs import (
    AGGRESSIVE_INSTRUCTIONS,
    BALANCED_INSTRUCTIONS,
    OUTPUT_REDUCTION_LEVELS,
    OutputDelta,
    OutputReductionReport,
    apply_output_reduction,
    measure_output_delta,
)
from .predict import (
    OutputPrediction,
    extract_output_features,
    predict_output_tokens,
    suggest_max_tokens,
)
from .pricing import estimate_cost, refresh_pricing
from .proxy import TokenSlimProxyServer, make_proxy_server, run_proxy
from .relevance import BM25Scorer, Scorer
from .retrieve import CCRContext, retrieve
from .router import ContentRouter, RouteResult, build_registry
from .semcache import (
    CacheHit,
    Embedder,
    HTTPEmbedder,
    SemanticCache,
    SentenceTransformerEmbedder,
)
from .sizer import compute_optimal_k
from .store import CCRStore, InMemoryCCRStore, SQLiteCCRStore, get_store
from .tokenizer import count_tokens, get_tokenizer

__version__ = "0.4.0"

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
    "TabularCompressor",
    "HtmlExtractor",
    "compute_optimal_k",
    "BM25Scorer",
    "Scorer",
    "count_tokens",
    "get_tokenizer",
    "MessageFormat",
    "detect_format",
    "openai_to_anthropic",
    "anthropic_to_openai",
    "with_tokenslim",
    "TokenSlimLiteLLMCallback",
    # LangChain (duck-typed; never imports langchain)
    "wrap_chat_model",
    "compress_documents",
    "compress_messages",
    "insert_anthropic_cache_control",
    "normalize_dynamic_content",
    # Images
    "ImagePlan",
    "ImageStats",
    "estimate_image_tokens",
    "plan_image_reduction",
    "reduce_image_tokens",
    "stabilize_message_order",
    "optimize_for_prefix_cache",
    "find_volatile_spans",
    "PrefixCacheReport",
    "VolatileSpan",
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
    "ProjectMemoryStore",
    "estimate_cost",
    "refresh_pricing",
    "run_suite",
    "perf_report",
    "SharedContext",
    # Session capture (opt-in, local-only)
    "SessionCapture",
    "get_capture",
    "read_sessions",
    # Output-length prediction
    "OutputPrediction",
    "predict_output_tokens",
    "extract_output_features",
    "suggest_max_tokens",
    # Output-token reduction
    "OUTPUT_REDUCTION_LEVELS",
    "BALANCED_INSTRUCTIONS",
    "AGGRESSIVE_INSTRUCTIONS",
    "OutputReductionReport",
    "OutputDelta",
    "apply_output_reduction",
    "measure_output_delta",
    # Learn (failure mining -> agent rules)
    "Finding",
    "analyze_sessions",
    "propose_rules",
    "apply_rules",
    # Audit (baseline vs optimized replay)
    "AuditReport",
    "AuditRow",
    "run_audit",
    "parse_requests",
    "render_audit_report",
    # Compressing reverse proxy (tokenslim proxy)
    "TokenSlimProxyServer",
    "make_proxy_server",
    "run_proxy",
    # Semantic cache (opt-in; embeddings via any Embedder implementation)
    "SemanticCache",
    "HTTPEmbedder",
    "CacheHit",
    "Embedder",
    "SentenceTransformerEmbedder",
    # Agent-framework hooks (Agno, Strands)
    "compress_tool_output",
    "tokenslim_agno_tool_hook",
    "wrap_agno_model",
    "TokenSlimStrandsHooks",
]
