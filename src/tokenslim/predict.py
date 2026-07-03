"""Output-length prediction — estimate how many tokens a reply will take.

Before sending a prompt you often need a ``max_tokens`` value. Guessing high
wastes reserved context (and money on providers that bill reserved output);
guessing low truncates answers. This module predicts the expected output
length from cheap, deterministic prompt features and a **transparent linear
model**: ``tokens = intercept + sum(weight[f] * feature[f])``.

The weights are data, not code. :data:`DEFAULT_WEIGHTS` /
:data:`DEFAULT_INTERCEPT` ship heuristic values; a calibration run on real
traffic can produce a JSON file (``{"intercept": float, "weights": {...},
"meta": {...}}``) that :func:`load_weights` reads and
:func:`predict_output_tokens` consumes — no code change needed to swap in a
fitted model.

Pure stdlib. Token counts funnel through :func:`tokenslim.tokenizer.count_tokens`
like the rest of the library.

Usage::

    from tokenslim import predict_output_tokens, suggest_max_tokens

    pred = predict_output_tokens(messages)
    max_tokens = suggest_max_tokens(pred)
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tokenizer import count_tokens

__all__ = [
    "OutputPrediction",
    "extract_output_features",
    "predict_output_tokens",
    "suggest_max_tokens",
    "load_weights",
    "save_weights",
    "DEFAULT_INTERCEPT",
    "DEFAULT_WEIGHTS",
    "FEATURE_NAMES",
    "MIN_PREDICTED_TOKENS",
    "MAX_PREDICTED_TOKENS",
]

# Predictions are clamped to this range: below ~16 tokens a reply is barely a
# sentence fragment; above 8192 the linear extrapolation is no longer credible.
MIN_PREDICTED_TOKENS = 16
MAX_PREDICTED_TOKENS = 8192

# Canonical feature order. extract_output_features() returns exactly these
# keys; a calibration run should emit weights for (a subset of) these names.
FEATURE_NAMES: tuple[str, ...] = (
    "prompt_tokens",
    "last_user_tokens",
    "n_questions",
    "has_code_fence",
    "asks_for_code",
    "asks_for_list",
    "asks_brevity",
    "asks_detail",
    "yes_no_question",
    "explicit_length_request",
    "json_schema_requested",
    "n_messages",
    "system_tokens",
)

# Hand-tuned heuristic defaults (units: output tokens per unit of feature).
# Replace via a calibrated JSON file (see load_weights) — do not fit by hand.
DEFAULT_INTERCEPT: float = 180.0
DEFAULT_WEIGHTS: dict[str, float] = {
    "prompt_tokens": 0.04,
    "last_user_tokens": 0.35,
    "n_questions": 55.0,
    "has_code_fence": 110.0,
    "asks_for_code": 210.0,
    "asks_for_list": 90.0,
    "asks_brevity": -140.0,
    "asks_detail": 320.0,
    "yes_no_question": -110.0,
    # explicit_length_request is already expressed in estimated output tokens,
    # so its natural coefficient is 1.0 — it dominates when present.
    "explicit_length_request": 1.0,
    "json_schema_requested": 40.0,
    "n_messages": 2.0,
    "system_tokens": 0.01,
}

# --- feature regexes (all case-insensitive, applied to the last user msg) ---

_CODE_VERB_RE = re.compile(
    r"\b(write|implement|create|build|refactor|fix|debug|generate|add|optimi[sz]e)\b",
    re.IGNORECASE,
)
_CODE_NOUN_RE = re.compile(
    r"\b(code|function|class|script|program|method|module|test|bug|api|endpoint"
    r"|component|regex|quer(?:y|ies)|snippet|algorithm|unit\s+tests?)\b",
    re.IGNORECASE,
)
_LIST_RE = re.compile(
    r"\b(enumerate|list|steps?|bullet(?:\s+points?)?|checklist|itemi[sz]e)\b",
    re.IGNORECASE,
)
_BREVITY_RE = re.compile(
    r"\b(brief(?:ly)?|one\s+word|tl;?dr|short(?:ly)?|concise(?:ly)?"
    r"|in\s+a\s+(?:word|sentence)|one[- ]liner)\b",
    re.IGNORECASE,
)
_DETAIL_RE = re.compile(
    r"\b(in\s+detail|detailed|comprehensive(?:ly)?|explain\s+thoroughly|thorough(?:ly)?"
    r"|in[- ]depth|elaborate|extensive(?:ly)?|step[- ]by[- ]step)\b",
    re.IGNORECASE,
)
_YES_NO_RE = re.compile(
    r"^(?:is|are|was|were|am|do|does|did|can|could|will|would|should|shall"
    r"|has|have|had|must)\b",
    re.IGNORECASE,
)
_LENGTH_RE = re.compile(
    r"\b(\d{1,6})\s*(words?|lines?|paragraphs?|sentences?|bullet\s+points?|bullets?|items?)\b",
    re.IGNORECASE,
)
_JSON_RE = re.compile(
    r"\bjson\s+(?:schema|format|object|only|response|output)\b"
    r"|\b(?:as|in|valid|return|output|respond\s+(?:in|with)|format(?:ted)?\s+as)\s+json\b",
    re.IGNORECASE,
)

# Rough output-token cost of one unit of each explicitly requested length unit
# (singular, lowercased). "200 words" ≈ 260 tokens, "3 paragraphs" ≈ 240, etc.
_LENGTH_UNIT_TOKENS: dict[str, float] = {
    "word": 1.3,
    "line": 12.0,
    "paragraph": 80.0,
    "sentence": 20.0,
    "bullet": 15.0,
    "bullet point": 15.0,
    "item": 15.0,
}


@dataclass(frozen=True)
class OutputPrediction:
    """A predicted output length with an uncertainty band.

    Attributes:
        tokens: Point estimate of the reply length in tokens.
        low: Lower bound of the plausible range (shrinks as confidence rises).
        high: Upper bound of the plausible range.
        confidence: Heuristic confidence in [0, 1] — high when the prompt
            carries strong length signals (explicit "N words", yes/no shape).
        features: The extracted feature vector the estimate was computed from.
    """

    tokens: int
    low: int
    high: int
    confidence: float
    features: dict[str, float]


def _message_text(message: Any) -> str:
    """Concatenate every text piece inside one message (str or block list)."""
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    parts: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                if block.get("type") == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, str):
                        parts.append(inner)
    return "\n".join(p for p in parts if p)


def _explicit_length_tokens(text: str) -> float:
    """Parse the first "N words/lines/paragraphs/..." request into ~tokens."""
    match = _LENGTH_RE.search(text)
    if match is None:
        return 0.0
    count = float(match.group(1))
    unit = " ".join(match.group(2).lower().split())
    if unit.endswith("s"):
        unit = unit[:-1]
    return count * _LENGTH_UNIT_TOKENS.get(unit, 1.0)


def extract_output_features(
    messages: Sequence[Mapping[str, Any]],
    model: str | None = None,
) -> dict[str, float]:
    """Extract the deterministic prompt features used for length prediction.

    Returns a dict with exactly the keys in :data:`FEATURE_NAMES`, every value
    a finite ``float``. Boolean features are 0.0/1.0;
    ``explicit_length_request`` is the requested length converted to an
    approximate token count (0.0 when absent).
    """
    all_texts = [_message_text(m) for m in messages]
    system_texts = [
        t
        for m, t in zip(messages, all_texts)
        if isinstance(m, Mapping) and m.get("role") == "system"
    ]
    last_user = ""
    for m, t in zip(messages, all_texts):
        if isinstance(m, Mapping) and m.get("role") == "user":
            last_user = t

    full_prompt = "\n".join(t for t in all_texts if t)
    system_text = "\n".join(system_texts)
    user_and_system = f"{last_user}\n{system_text}"

    asks_for_code = bool(_CODE_VERB_RE.search(last_user)) and bool(_CODE_NOUN_RE.search(last_user))
    return {
        "prompt_tokens": float(count_tokens(full_prompt, model)),
        "last_user_tokens": float(count_tokens(last_user, model)),
        "n_questions": float(last_user.count("?")),
        "has_code_fence": float("```" in full_prompt),
        "asks_for_code": float(asks_for_code),
        "asks_for_list": float(bool(_LIST_RE.search(last_user))),
        "asks_brevity": float(bool(_BREVITY_RE.search(last_user))),
        "asks_detail": float(bool(_DETAIL_RE.search(last_user))),
        "yes_no_question": float(bool(_YES_NO_RE.match(last_user.lstrip()))),
        "explicit_length_request": _explicit_length_tokens(last_user),
        "json_schema_requested": float(bool(_JSON_RE.search(user_and_system))),
        "n_messages": float(len(messages)),
        "system_tokens": float(count_tokens(system_text, model)),
    }


def _confidence(features: Mapping[str, float]) -> float:
    """Heuristic confidence: strong length cues raise it, open-endedness lowers it."""
    conf = 0.35
    if features.get("explicit_length_request", 0.0) > 0:
        conf += 0.5
    if features.get("yes_no_question"):
        conf += 0.15
    if features.get("asks_brevity"):
        conf += 0.1
    if features.get("asks_for_list"):
        conf += 0.05
    if features.get("asks_detail"):
        conf -= 0.1
    return max(0.05, min(0.95, conf))


def predict_output_tokens(
    messages: Sequence[Mapping[str, Any]],
    model: str | None = None,
    *,
    weights: Mapping[str, Any] | None = None,
) -> OutputPrediction:
    """Predict the reply length for ``messages`` with a linear model.

    Args:
        messages: OpenAI/Anthropic-style message array (the prompt).
        model: Optional model name, forwarded to the tokenizer.
        weights: Optional weight blob in the :func:`load_weights` format
            (``{"intercept": float, "weights": {feature: coef}, ...}``).
            Defaults to :data:`DEFAULT_WEIGHTS` / :data:`DEFAULT_INTERCEPT`.

    The point estimate is clamped to
    [:data:`MIN_PREDICTED_TOKENS`, :data:`MAX_PREDICTED_TOKENS`]. The
    ``low``/``high`` band spans ``tokens / spread`` to ``tokens * spread``
    where ``spread = 2 - confidence`` — a full halve/double at zero confidence,
    collapsing toward the point estimate as confidence approaches 1.
    """
    if weights is None:
        intercept = DEFAULT_INTERCEPT
        coefs: Mapping[str, float] = DEFAULT_WEIGHTS
    else:
        intercept = float(weights.get("intercept", DEFAULT_INTERCEPT))
        raw_coefs = weights.get("weights", {})
        coefs = {str(k): float(v) for k, v in dict(raw_coefs).items()}

    features = extract_output_features(messages, model)
    raw = intercept + sum(coefs.get(name, 0.0) * value for name, value in features.items())

    tokens = int(round(min(float(MAX_PREDICTED_TOKENS), max(float(MIN_PREDICTED_TOKENS), raw))))
    confidence = _confidence(features)
    spread = 2.0 - confidence
    low = max(MIN_PREDICTED_TOKENS, int(round(tokens / spread)))
    high = min(MAX_PREDICTED_TOKENS, int(round(tokens * spread)))
    return OutputPrediction(
        tokens=tokens, low=low, high=high, confidence=confidence, features=features
    )


def suggest_max_tokens(
    prediction: OutputPrediction,
    safety: float = 1.5,
    floor: int = 64,
    cap: int = 16384,
) -> int:
    """Turn a prediction into a ``max_tokens`` value to send to the API.

    Takes the larger of ``tokens * safety`` and the prediction's ``high``
    bound (so a wide uncertainty band is never undercut), then clamps to
    ``[floor, cap]``.
    """
    target = max(prediction.tokens * safety, float(prediction.high))
    return int(min(float(cap), max(float(floor), math.ceil(target))))


def load_weights(path: str | Path) -> dict[str, Any]:
    """Load a weight file produced by :func:`save_weights` (or a calibration run).

    Expected JSON shape: ``{"intercept": float, "weights": {feature: coef},
    "meta": {...}}`` (``meta`` optional). Returns the validated blob, ready to
    pass as ``predict_output_tokens(..., weights=blob)``.

    Raises:
        ValueError: If the file is not a JSON object of the expected shape.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"weight file {path}: expected a JSON object, got {type(data).__name__}")
    if "intercept" not in data or "weights" not in data:
        raise ValueError(f"weight file {path}: missing required 'intercept'/'weights' keys")
    raw_weights = data["weights"]
    if not isinstance(raw_weights, dict):
        raise ValueError(f"weight file {path}: 'weights' must be an object of feature: coef")
    try:
        intercept = float(data["intercept"])
        coefs = {str(k): float(v) for k, v in raw_weights.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"weight file {path}: non-numeric intercept or coefficient") from exc
    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError(f"weight file {path}: 'meta' must be an object")
    return {"intercept": intercept, "weights": coefs, "meta": meta}


def save_weights(
    path: str | Path,
    *,
    intercept: float | None = None,
    weights: Mapping[str, float] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write a weight file in the :func:`load_weights` JSON format.

    Defaults to the shipped heuristic model, so ``save_weights(path)`` emits a
    template a calibration experiment can start from. Returns the path written.
    """
    blob = {
        "intercept": float(intercept if intercept is not None else DEFAULT_INTERCEPT),
        "weights": {
            str(k): float(v)
            for k, v in (weights if weights is not None else DEFAULT_WEIGHTS).items()
        },
        "meta": dict(meta) if meta is not None else {},
    }
    out = Path(path)
    out.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
