"""Output-token reduction — trim what the model writes back.

Input compression (the rest of tokenslim) shrinks what the model *reads*;
this module shrinks what it *writes*. It appends a provider-neutral,
plain-text instruction block to the system message asking the model to skip
ceremony (preambles, recaps, restated code) and, at the aggressive level, to
obey hard brevity rules.

Two halves:

1. :func:`apply_output_reduction` — rewrite a message array (never mutating
   the input, like :func:`tokenslim.compress.compress`) and report the
   expected effect using :mod:`tokenslim.predict`.
2. :func:`measure_output_delta` — compare a baseline reply against a reduced
   reply and quantify the tokens actually saved.

Usage::

    from tokenslim import apply_output_reduction, measure_output_delta

    reduced, report = apply_output_reduction(messages, level="balanced")
    response = client.chat.completions.create(
        model="gpt-4o", messages=reduced, max_tokens=report.suggested_max_tokens
    )
    delta = measure_output_delta(baseline_text, response_text)

Pure stdlib; token counts funnel through :func:`tokenslim.tokenizer.count_tokens`.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .predict import predict_output_tokens, suggest_max_tokens
from .tokenizer import count_tokens

__all__ = [
    "OUTPUT_REDUCTION_LEVELS",
    "BALANCED_INSTRUCTIONS",
    "AGGRESSIVE_INSTRUCTIONS",
    "OutputReductionReport",
    "OutputDelta",
    "apply_output_reduction",
    "measure_output_delta",
]

# --- instruction blocks (provider-neutral plain text) -----------------------

# Balanced: cut ceremony without constraining substance.
BALANCED_INSTRUCTIONS: str = """\
Response length rules:
- Do not restate code, files, or data that are unchanged; reference them instead.
- Skip preambles, restated plans, and closing recaps; start with the substance.
- Keep prose terse on routine steps; expand only where a decision, trade-off, or caveat matters."""

# Aggressive: everything in balanced, plus hard rules for maximum brevity.
AGGRESSIVE_INSTRUCTIONS: str = (
    BALANCED_INSTRUCTIONS
    + """
Hard brevity rules:
- Answer first; add explanation only when strictly necessary, after the answer.
- Never restate the question, the inputs, or content already shown in the conversation.
- Show code changes as diffs or changed lines only; never re-print whole files.
- Boilerplate (imports, config, scaffolding) at maximum brevity or omitted entirely.
- No filler, no apologies, no offers of further help."""
)

# Level name -> instruction block appended to the system message. "off" maps
# to the empty string: apply_output_reduction() is then an identity rewrite.
OUTPUT_REDUCTION_LEVELS: dict[str, str] = {
    "off": "",
    "balanced": BALANCED_INSTRUCTIONS,
    "aggressive": AGGRESSIVE_INSTRUCTIONS,
}

# max_tokens safety factor per level (see predict.suggest_max_tokens). The
# aggressive instructions actively shorten replies, so the headroom is tighter.
_SAFETY_BY_LEVEL: dict[str, float] = {
    "off": 1.5,
    "balanced": 1.5,
    "aggressive": 1.2,
}


@dataclass(frozen=True)
class OutputReductionReport:
    """What :func:`apply_output_reduction` did and what to expect from it.

    Attributes:
        level: The effective level applied ("off" for unknown level names).
        instruction_tokens: Input-token cost of the appended instruction block
            (0 when level is "off") — the price paid for shorter outputs.
        predicted_tokens_before: Predicted reply length for the *original*
            messages, from :func:`tokenslim.predict.predict_output_tokens`.
        suggested_max_tokens: A ``max_tokens`` value to send with the request,
            from :func:`tokenslim.predict.suggest_max_tokens` (tighter safety
            factor at the aggressive level).
    """

    level: str
    instruction_tokens: int
    predicted_tokens_before: int
    suggested_max_tokens: int


@dataclass(frozen=True)
class OutputDelta:
    """Measured effect of output reduction on one reply.

    Attributes:
        baseline_tokens: Tokens in the reply without reduction.
        reduced_tokens: Tokens in the reply with reduction applied.
        saved_tokens: ``baseline_tokens - reduced_tokens`` (negative when the
            reduced reply is longer).
        ratio: Fraction of output tokens removed, ``1 - reduced/baseline``
            (0.0 when the baseline is empty) — same semantics as
            :attr:`tokenslim.compress.CompressionStats.ratio`.
    """

    baseline_tokens: int
    reduced_tokens: int
    saved_tokens: int
    ratio: float


def _append_instructions(messages: list[Any], instructions: str) -> None:
    """Append ``instructions`` to the first system message (create one if absent).

    Mutates ``messages`` in place — callers pass a deep copy.
    """
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = f"{content}\n\n{instructions}" if content else instructions
            return
        if isinstance(content, list):
            content.append({"type": "text", "text": instructions})
            return
        if content is None:
            msg["content"] = instructions
            return
        break  # unrecognized content shape: fall through to a fresh system message
    messages.insert(0, {"role": "system", "content": instructions})


def apply_output_reduction(
    messages: Sequence[Mapping[str, Any]],
    level: str = "balanced",
    model: str | None = None,
) -> tuple[list[dict[str, Any]], OutputReductionReport]:
    """Append output-brevity instructions to the system message.

    Args:
        messages: OpenAI/Anthropic-style message array. Never mutated — the
            returned array is a deep copy (same contract as ``compress()``).
        level: One of :data:`OUTPUT_REDUCTION_LEVELS` — "off", "balanced", or
            "aggressive". Unknown names are treated as "off" (never raises).
        model: Optional model name, forwarded to the tokenizer and predictor.

    Returns:
        ``(rewritten_messages, report)``. With level "off" the messages are an
        unmodified copy; otherwise the instruction block is appended to the
        first system message, or a new system message is inserted at index 0.
    """
    if level not in OUTPUT_REDUCTION_LEVELS:
        level = "off"
    instructions = OUTPUT_REDUCTION_LEVELS[level]

    prediction = predict_output_tokens(messages, model)
    report = OutputReductionReport(
        level=level,
        instruction_tokens=count_tokens(instructions, model) if instructions else 0,
        predicted_tokens_before=prediction.tokens,
        suggested_max_tokens=suggest_max_tokens(prediction, safety=_SAFETY_BY_LEVEL[level]),
    )

    out: list[dict[str, Any]] = [copy.deepcopy(dict(msg)) for msg in messages]
    if instructions:
        _append_instructions(out, instructions)
    return out, report


def measure_output_delta(
    baseline_response: str,
    reduced_response: str,
    model: str | None = None,
) -> OutputDelta:
    """Quantify the output tokens saved by reduction on one reply pair.

    Args:
        baseline_response: The reply text produced *without* output reduction.
        reduced_response: The reply text produced *with* it.
        model: Optional model name, forwarded to the tokenizer.
    """
    baseline_tokens = count_tokens(baseline_response, model)
    reduced_tokens = count_tokens(reduced_response, model)
    ratio = 1.0 - (reduced_tokens / baseline_tokens) if baseline_tokens else 0.0
    return OutputDelta(
        baseline_tokens=baseline_tokens,
        reduced_tokens=reduced_tokens,
        saved_tokens=baseline_tokens - reduced_tokens,
        ratio=ratio,
    )
