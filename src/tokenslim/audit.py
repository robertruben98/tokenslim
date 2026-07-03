"""Audit — replay identical requests baseline vs optimized (issue #69).

Given a batch of OpenAI-style requests, :func:`run_audit` compresses each one
and diffs token counts: ``baseline`` (original messages) vs ``optimized``
(after :func:`tokenslim.compress`). Per-request rows carry a per-content-type
breakdown from :class:`~tokenslim.compress.BlockStat`; aggregate totals include
an estimated USD cost delta when a model is given.

Optionally (``answers=True``) both variants are replayed against an
OpenAI-compatible chat-completions endpoint (``OPENAI_API_KEY`` +
``OPENAI_BASE_URL``, read at call time, stdlib HTTP only) with
``temperature=0`` and the answers are diffed with
:class:`difflib.SequenceMatcher`. Any network/auth error degrades gracefully
to a token-only audit with a warning in the report — :func:`run_audit` never
raises.

Note: ``ratio`` here is the fraction of tokens *saved* (matching
:attr:`CompressionStats.ratio` and :mod:`tokenslim.metrics`), NOT the
retention ratio reported by :mod:`tokenslim.telemetry`.
"""

from __future__ import annotations

import difflib
import json
import os
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .compress import compress
from .config import Config, load_config
from .pricing import estimate_cost

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["AuditRow", "AuditReport", "run_audit", "parse_requests", "render_audit_report"]

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_ANSWER_TIMEOUT = 30.0


@dataclass(frozen=True)
class AuditRow:
    """Baseline-vs-optimized comparison for a single request."""

    id: str
    baseline_tokens: int
    optimized_tokens: int
    saved_tokens: int
    # Fraction of tokens saved (0.0 = no change).
    ratio: float
    # Per-content-type token breakdown, e.g.
    # {"json": {"baseline_tokens": 900, "optimized_tokens": 120, "saved_tokens": 780}}.
    by_content_type: dict[str, dict[str, int]] = field(default_factory=dict)
    # Answers mode (None when answers were not collected for this row).
    baseline_answer: str | None = None
    optimized_answer: str | None = None
    # difflib.SequenceMatcher ratio between the two answers (1.0 = identical).
    answer_similarity: float | None = None


@dataclass
class AuditReport:
    """Aggregate result of an audit run."""

    rows: list[AuditRow] = field(default_factory=list)
    model: str | None = None
    baseline_tokens: int = 0
    optimized_tokens: int = 0
    saved_tokens: int = 0
    # Estimated input-cost USD (None when no model was given).
    baseline_cost: float | None = None
    optimized_cost: float | None = None
    saved_cost: float | None = None
    # True when at least one row carries a live answer diff.
    answers_mode: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        """Overall fraction of tokens saved (0.0 = no change)."""
        if self.baseline_tokens == 0:
            return 0.0
        return 1.0 - (self.optimized_tokens / self.baseline_tokens)

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable report (used by ``tokenslim audit --json``)."""
        return {
            "model": self.model,
            "requests": len(self.rows),
            "baseline_tokens": self.baseline_tokens,
            "optimized_tokens": self.optimized_tokens,
            "saved_tokens": self.saved_tokens,
            "ratio": self.ratio,
            "baseline_cost_usd": self.baseline_cost,
            "optimized_cost_usd": self.optimized_cost,
            "saved_cost_usd": self.saved_cost,
            "answers_mode": self.answers_mode,
            "warnings": list(self.warnings),
            "rows": [asdict(row) for row in self.rows],
        }


def parse_requests(text: str) -> list[Any]:
    """Parse audit requests from a JSON array or JSONL string.

    Accepted shapes:

    * a JSON array whose elements are requests — each a bare messages array or
      an object like ``{"messages": [...], "id": "req-1"}``;
    * a JSON array of ``{"role": ..., "content": ...}`` message objects, which
      is treated as a *single* request;
    * JSONL — one request per line (same element shapes).

    Raises:
        ValueError: when the input is not parseable in any of those shapes.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty input: expected a JSON array or JSONL of requests")

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return _parse_jsonl(stripped)

    if isinstance(data, list):
        if data and all(_looks_like_message(item) for item in data):
            # A bare messages array is one request, not N requests.
            return [data]
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("input must be a JSON array of requests (or JSONL, one request per line)")


def _parse_jsonl(text: str) -> list[Any]:
    requests: list[Any] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            requests.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON on line {lineno}: {e}") from e
    if not requests:
        raise ValueError("empty input: expected a JSON array or JSONL of requests")
    return requests


def _looks_like_message(item: Any) -> bool:
    return isinstance(item, dict) and "role" in item and "messages" not in item


def _normalize_request(raw: Any, index: int) -> tuple[str, list[dict[str, Any]]] | None:
    """Return ``(id, messages)`` for a request, or None when unrecognized."""
    if isinstance(raw, list):
        return f"req-{index}", raw
    if isinstance(raw, dict) and isinstance(raw.get("messages"), list):
        req_id = raw.get("id")
        return (str(req_id) if req_id is not None else f"req-{index}"), raw["messages"]
    return None


def _chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: float = _ANSWER_TIMEOUT,
) -> str:
    """POST one chat completion (temperature=0) and return the answer text."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({"model": model, "messages": messages, "temperature": 0}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "tokenslim-audit",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"] or ""


def run_audit(
    requests: Sequence[Any],
    config: Config | None = None,
    model: str | None = None,
    *,
    answers: bool = False,
    env: Mapping[str, str] | None = None,
) -> AuditReport:
    """Replay ``requests`` with and without compression and diff token counts.

    Args:
        requests: Requests to audit. Each item is either a bare OpenAI-style
            messages array or a dict like ``{"messages": [...], "id": ...}``.
            Unrecognized items are skipped with a warning — never an exception.
        config: A resolved :class:`Config`; loaded from the environment when
            omitted.
        model: Model name for token counting and cost estimation (falls back
            to ``config.model``; cost fields stay ``None`` without a model).
        answers: When True, also send BOTH variants to an OpenAI-compatible
            endpoint (``OPENAI_API_KEY`` + optional ``OPENAI_BASE_URL`` read
            from ``env`` at call time) and record an answer-similarity signal.
            Any failure degrades to token-only mode with a warning.
        env: Environment mapping for answers-mode credentials (defaults to
            ``os.environ``; injectable for tests).
    """
    cfg = (config if config is not None else load_config()).merged(model=model)
    model_name = model or cfg.model
    report = AuditReport(model=model_name)

    environ: Mapping[str, str] = os.environ if env is None else env
    answers_enabled = False
    api_key = ""
    base_url = _DEFAULT_BASE_URL
    if answers:
        api_key = environ.get("OPENAI_API_KEY", "")
        base_url = environ.get("OPENAI_BASE_URL", _DEFAULT_BASE_URL)
        if api_key:
            answers_enabled = True
        else:
            report.warnings.append(
                "answers mode: OPENAI_API_KEY is not set; running token-only audit"
            )

    for index, raw in enumerate(requests):
        normalized = _normalize_request(raw, index)
        if normalized is None:
            report.warnings.append(
                f"request {index}: unrecognized shape, skipped "
                "(expected a messages array or {'messages': [...]})"
            )
            continue
        req_id, messages = normalized

        try:
            optimized, stats = compress(messages, options=cfg)
        except Exception as e:  # noqa: BLE001 — audit must never crash on one request
            report.warnings.append(
                f"request '{req_id}': compression failed ({e.__class__.__name__}: {e}), skipped"
            )
            continue

        by_type: dict[str, dict[str, int]] = {}
        for block in stats.blocks:
            bucket = by_type.setdefault(
                block.content_type.value,
                {"baseline_tokens": 0, "optimized_tokens": 0, "saved_tokens": 0},
            )
            bucket["baseline_tokens"] += block.orig_tokens
            bucket["optimized_tokens"] += block.new_tokens
            bucket["saved_tokens"] += block.orig_tokens - block.new_tokens

        baseline_answer: str | None = None
        optimized_answer: str | None = None
        similarity: float | None = None
        if answers_enabled:
            answer_model = model_name or "gpt-4o"
            try:
                baseline_answer = _chat_completion(base_url, api_key, answer_model, messages)
                optimized_answer = _chat_completion(base_url, api_key, answer_model, optimized)
                similarity = difflib.SequenceMatcher(
                    None, baseline_answer, optimized_answer
                ).ratio()
            except Exception as e:  # noqa: BLE001 — degrade to token-only, never crash
                answers_enabled = False
                baseline_answer = optimized_answer = None
                report.warnings.append(
                    f"answers mode: request '{req_id}' failed "
                    f"({e.__class__.__name__}: {e}); falling back to token-only audit"
                )

        row = AuditRow(
            id=req_id,
            baseline_tokens=stats.orig_tokens,
            optimized_tokens=stats.new_tokens,
            saved_tokens=stats.saved_tokens,
            ratio=stats.ratio,
            by_content_type=by_type,
            baseline_answer=baseline_answer,
            optimized_answer=optimized_answer,
            answer_similarity=similarity,
        )
        report.rows.append(row)
        report.baseline_tokens += row.baseline_tokens
        report.optimized_tokens += row.optimized_tokens
        report.saved_tokens += row.saved_tokens

    if model_name:
        report.baseline_cost = estimate_cost(model_name, report.baseline_tokens)
        report.optimized_cost = estimate_cost(model_name, report.optimized_tokens)
        report.saved_cost = report.baseline_cost - report.optimized_cost

    report.answers_mode = any(row.answer_similarity is not None for row in report.rows)
    return report


def render_audit_report(report: AuditReport) -> str:
    """Render an :class:`AuditReport` as a human-readable table (perf style)."""
    lines = [
        "# tokenslim audit report",
        "",
        f"- **Requests:** {len(report.rows)}",
        f"- **Baseline tokens:** {report.baseline_tokens:,}",
        f"- **Optimized tokens:** {report.optimized_tokens:,}",
        f"- **Saved tokens:** {report.saved_tokens:,} ({report.ratio:.1%})",
    ]
    if report.saved_cost is not None:
        lines.append(
            f"- **Estimated cost:** ${report.baseline_cost:,.6f} -> "
            f"${report.optimized_cost:,.6f} "
            f"(saved ${report.saved_cost:,.6f}, model: {report.model})"
        )
    lines.append(f"- **Answers mode:** {'on' if report.answers_mode else 'off'}")
    for warning in report.warnings:
        lines.append(f"- **Warning:** {warning}")
    lines += [
        "",
        "| Request | Baseline | Optimized | Saved | Ratio | Similarity |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.rows:
        sim = f"{row.answer_similarity:.2f}" if row.answer_similarity is not None else "-"
        lines.append(
            f"| {row.id} | {row.baseline_tokens:,} | {row.optimized_tokens:,} | "
            f"{row.saved_tokens:,} | {row.ratio:.1%} | {sim} |"
        )
    return "\n".join(lines)
