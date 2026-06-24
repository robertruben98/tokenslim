"""Evaluation harness for testing context compression accuracy preservation.

Defines standard test suites (GSM8K, QA) and runs them across baseline
and compressed prompts to report the accuracy delta and token/cost savings.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TypedDict

from .compress import compress
from .config import Config
from .tokenizer import count_tokens


class Fixture(TypedDict):
    id: str
    suite: str
    description: str
    messages: list[dict[str, Any]]
    validator: Callable[[str], bool]


__all__ = ["run_eval_suite", "EVAL_FIXTURES"]


# Helper validators
def validate_math(ans: str) -> bool:
    return "500" in ans


def validate_logs(ans: str) -> bool:
    return "ZeroDivisionError" in ans and "42" in ans


def validate_code(ans: str) -> bool:
    return "list[float]" in ans and "threshold" in ans


# Evaluation fixtures testing different compressors
EVAL_FIXTURES: list[Fixture] = [
    {
        "id": "math_outliers",
        "suite": "gsm8k",
        "description": "Calculate total value from a transaction list with outlier values.",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful financial assistant.",
            },
            {
                "role": "user",
                "content": "Here is the transaction JSON list:\n"
                + json.dumps(
                    [{"id": i, "val": 1.0} for i in range(50)]
                    + [{"id": 50, "val": 500.0}]
                    + [{"id": i, "val": 1.0} for i in range(51, 100)]
                )
                + "\nWhat is the value of the outlier transaction?",
            },
        ],
        "validator": validate_math,
    },
    {
        "id": "log_traceback",
        "suite": "qa",
        "description": "Identify traceback error type and line from build logs.",
        "messages": [
            {
                "role": "system",
                "content": "You are a systems engineer helper.",
            },
            {
                "role": "user",
                "content": "Analyze these logs:\n"
                + "\n".join([f"INFO line {i}" for i in range(40)])
                + "\nTraceback (most recent call last):\n"
                + '  File "app.py", line 42, in run\n'
                + '    raise ZeroDivisionError("division by zero")\n'
                + "ZeroDivisionError: division by zero\n"
                + "\n".join([f"INFO line {i + 40}" for i in range(40)])
                + "\nWhat error occurred and on what line?",
            },
        ],
        "validator": validate_logs,
    },
    {
        "id": "code_signature",
        "suite": "qa",
        "description": "Find parameter types of a function in a class.",
        "messages": [
            {
                "role": "system",
                "content": "You are a code analyst assistant.",
            },
            {
                "role": "user",
                "content": "Here is the python code:\n"
                + "class Model:\n"
                + "    def predict(self, data: list[float], threshold: float = 0.5) -> bool:\n"
                + '        """Predict output.\n'
                + "        This is a multi-line docstring.\n"
                + '        """\n'
                + "        x = [d * 2 for d in data]\n"
                + "        return sum(x) > threshold\n"
                + "\nWhat are the types of the parameters 'data' and 'threshold' in 'predict'?",
            },
        ],
        "validator": validate_code,
    },
]


def simulate_llm(messages: list[dict[str, Any]]) -> str:
    """Deterministic simulated solver behaving like an LLM reading the context."""
    full_text = "\n".join(msg.get("content", "") for msg in messages)

    # 1. Math Outliers solver
    if "500" in full_text:
        return "The outlier transaction has a value of 500.0."

    # 2. Log Traceback solver
    if "ZeroDivisionError" in full_text and "42" in full_text:
        return "ZeroDivisionError occurred on line 42."

    # 3. Code Signature solver
    if "predict" in full_text and "list[float]" in full_text:
        return "The parameter types are data: list[float] and threshold: float."

    return "Error: Information was elided or could not be found in the context."


def run_eval_suite(
    suite_name: str = "all",
    config: Config | None = None,
) -> dict[str, Any]:
    """Run accuracy-preservation evaluation suite comparing baseline vs compressed."""
    cfg = config or Config()
    fixtures = EVAL_FIXTURES
    if suite_name != "all":
        fixtures = [f for f in fixtures if f["suite"] == suite_name]

    if not fixtures:
        return {
            "total": 0,
            "baseline_correct": 0,
            "compressed_correct": 0,
            "baseline_tokens": 0,
            "compressed_tokens": 0,
            "results": [],
        }

    results = []
    baseline_correct = 0
    compressed_correct = 0
    baseline_tokens = 0
    compressed_tokens = 0

    for fixture in fixtures:
        msgs = fixture["messages"]
        validator = fixture["validator"]

        # 1. Run Baseline
        base_tokens = sum(count_tokens(msg.get("content", ""), cfg.model) for msg in msgs)
        base_ans = simulate_llm(msgs)
        base_ok = validator(base_ans)
        if base_ok:
            baseline_correct += 1
        baseline_tokens += base_tokens

        # 2. Run Compressed
        comp_msgs, stats = compress(msgs, options=cfg)
        comp_tokens = stats.new_tokens
        comp_ans = simulate_llm(comp_msgs)
        comp_ok = validator(comp_ans)
        if comp_ok:
            compressed_correct += 1
        compressed_tokens += comp_tokens

        results.append(
            {
                "id": fixture["id"],
                "description": fixture["description"],
                "baseline_ok": base_ok,
                "compressed_ok": comp_ok,
                "baseline_tokens": base_tokens,
                "compressed_tokens": comp_tokens,
                "saved_tokens": base_tokens - comp_tokens,
                "ratio": (base_tokens - comp_tokens) / base_tokens if base_tokens > 0 else 0.0,
            }
        )

    n = len(fixtures)
    return {
        "total": n,
        "baseline_correct": baseline_correct,
        "compressed_correct": compressed_correct,
        "baseline_accuracy": baseline_correct / n,
        "compressed_accuracy": compressed_correct / n,
        "baseline_tokens": baseline_tokens,
        "compressed_tokens": compressed_tokens,
        "saved_tokens": baseline_tokens - compressed_tokens,
        "ratio": (
            (baseline_tokens - compressed_tokens) / baseline_tokens if baseline_tokens > 0 else 0.0
        ),
        "results": results,
    }
