"""Tests for tokenslim.outputs (issue #63 — output-token reduction)."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from tokenslim.config import Config, load_config
from tokenslim.integrations import with_tokenslim
from tokenslim.outputs import (
    AGGRESSIVE_INSTRUCTIONS,
    BALANCED_INSTRUCTIONS,
    OUTPUT_REDUCTION_LEVELS,
    OutputDelta,
    OutputReductionReport,
    apply_output_reduction,
    measure_output_delta,
)
from tokenslim.tokenizer import count_tokens


def _messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Refactor this function and explain the changes."},
    ]


# --- levels ------------------------------------------------------------------


def test_levels_dict_shape() -> None:
    assert set(OUTPUT_REDUCTION_LEVELS) == {"off", "balanced", "aggressive"}
    assert OUTPUT_REDUCTION_LEVELS["off"] == ""
    assert OUTPUT_REDUCTION_LEVELS["balanced"] == BALANCED_INSTRUCTIONS
    assert OUTPUT_REDUCTION_LEVELS["aggressive"] == AGGRESSIVE_INSTRUCTIONS


def test_aggressive_contains_balanced_core_and_is_stronger() -> None:
    assert BALANCED_INSTRUCTIONS in AGGRESSIVE_INSTRUCTIONS
    assert len(AGGRESSIVE_INSTRUCTIONS) > len(BALANCED_INSTRUCTIONS)
    assert count_tokens(AGGRESSIVE_INSTRUCTIONS) > count_tokens(BALANCED_INSTRUCTIONS)


# --- apply_output_reduction --------------------------------------------------


def test_off_is_identity() -> None:
    messages = _messages()
    out, report = apply_output_reduction(messages, level="off")
    assert out == messages
    assert out is not messages, "must return a copy even when off"
    assert report.level == "off"
    assert report.instruction_tokens == 0
    assert report.predicted_tokens_before > 0
    assert report.suggested_max_tokens > 0


def test_unknown_level_treated_as_off() -> None:
    messages = _messages()
    out, report = apply_output_reduction(messages, level="turbo")
    assert out == messages
    assert report.level == "off"
    assert report.instruction_tokens == 0


def test_input_never_mutated() -> None:
    messages = _messages()
    snapshot = copy.deepcopy(messages)
    out, _ = apply_output_reduction(messages, level="aggressive")
    assert messages == snapshot, "input messages were mutated"
    assert out[0]["content"] != messages[0]["content"]
    assert out[0] is not messages[0], "returned messages share dicts with the input"


def test_appends_to_existing_system_message() -> None:
    out, report = apply_output_reduction(_messages(), level="balanced")
    assert len(out) == 2, "no new message should be added when a system message exists"
    system = out[0]["content"]
    assert system.startswith("You are a helpful assistant.")
    assert BALANCED_INSTRUCTIONS in system
    assert report.level == "balanced"
    assert report.instruction_tokens == count_tokens(BALANCED_INSTRUCTIONS)


def test_creates_system_message_as_first_when_absent() -> None:
    messages = [{"role": "user", "content": "Summarize the report."}]
    out, _ = apply_output_reduction(messages, level="balanced")
    assert len(out) == 2
    assert out[0]["role"] == "system"
    assert out[0]["content"] == BALANCED_INSTRUCTIONS
    assert out[1] == messages[0]


def test_appends_text_block_to_block_list_system() -> None:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Be helpful."}]},
        {"role": "user", "content": "hi"},
    ]
    out, _ = apply_output_reduction(messages, level="aggressive")
    blocks = out[0]["content"]
    assert blocks[0] == {"type": "text", "text": "Be helpful."}
    assert blocks[-1] == {"type": "text", "text": AGGRESSIVE_INSTRUCTIONS}


def test_report_is_frozen() -> None:
    _, report = apply_output_reduction(_messages(), level="balanced")
    assert isinstance(report, OutputReductionReport)
    with pytest.raises(AttributeError):
        report.level = "aggressive"  # type: ignore[misc]


def test_aggressive_suggests_tighter_max_tokens() -> None:
    # Explicit length request -> high confidence -> the safety factor (not the
    # uncertainty band) drives suggest_max_tokens, so aggressive < balanced.
    messages = [{"role": "user", "content": "Explain the design in 200 words."}]
    _, balanced = apply_output_reduction(messages, level="balanced")
    _, aggressive = apply_output_reduction(messages, level="aggressive")
    assert balanced.predicted_tokens_before == aggressive.predicted_tokens_before
    assert aggressive.suggested_max_tokens < balanced.suggested_max_tokens


# --- measure_output_delta ----------------------------------------------------


def test_measure_output_delta_math() -> None:
    baseline = "Sure! Here is a detailed recap of everything you asked, step by step. " * 10
    reduced = "Done: renamed the function and removed the dead branch."
    delta = measure_output_delta(baseline, reduced)
    assert isinstance(delta, OutputDelta)
    assert delta.baseline_tokens == count_tokens(baseline)
    assert delta.reduced_tokens == count_tokens(reduced)
    assert delta.saved_tokens == delta.baseline_tokens - delta.reduced_tokens
    assert delta.ratio == pytest.approx(1.0 - delta.reduced_tokens / delta.baseline_tokens)
    assert 0.0 < delta.ratio < 1.0


def test_measure_output_delta_empty_baseline() -> None:
    delta = measure_output_delta("", "anything")
    assert delta.baseline_tokens == 0
    assert delta.ratio == 0.0
    assert delta.saved_tokens == -delta.reduced_tokens


def test_measure_output_delta_negative_savings() -> None:
    delta = measure_output_delta("short", "a much longer reply than the baseline was")
    assert delta.saved_tokens < 0
    assert delta.ratio < 0.0


# --- config knob -------------------------------------------------------------


def test_config_knob_default_and_env() -> None:
    assert Config().output_reduction == "off"
    cfg = load_config(env={"TOKENSLIM_OUTPUT_REDUCTION": "aggressive"})
    assert cfg.output_reduction == "aggressive"


# --- integration path (mock client pattern from test_integrations.py) --------


class _MockCompletions:
    def create(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return {"model": model, "messages": messages, "kwargs": kwargs}


class _MockOpenAIClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _MockCompletions()})()


class _MockAsyncCompletions:
    async def create(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return {"model": model, "messages": messages, "kwargs": kwargs}


class _MockAsyncOpenAIClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _MockAsyncCompletions()})()


def test_integration_applies_reduction_when_config_set() -> None:
    client = with_tokenslim(_MockOpenAIClient(), config=Config(output_reduction="balanced"))
    res = client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": "hello"}]
    )
    assert res["messages"][0]["role"] == "system"
    assert BALANCED_INSTRUCTIONS in res["messages"][0]["content"]
    assert res["messages"][1] == {"role": "user", "content": "hello"}


def test_integration_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOKENSLIM_OUTPUT_REDUCTION", raising=False)
    client = with_tokenslim(_MockOpenAIClient())
    res = client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": "hello"}]
    )
    assert all(m.get("role") != "system" for m in res["messages"])


def test_integration_reduction_composes_with_compression() -> None:
    client = with_tokenslim(_MockOpenAIClient(), config=Config(output_reduction="aggressive"))
    long_numbers = [float(x) for x in range(100)]
    original = json.dumps(long_numbers)
    res = client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": original}]
    )
    assert res["messages"][0]["role"] == "system"
    assert AGGRESSIVE_INSTRUCTIONS in res["messages"][0]["content"]
    user_content = res["messages"][1]["content"]
    assert len(user_content) < len(original), "compression should still apply"
    assert "__tokenslim_ccr__" in user_content


@pytest.mark.asyncio
async def test_integration_async_applies_reduction() -> None:
    client = with_tokenslim(_MockAsyncOpenAIClient(), config=Config(output_reduction="balanced"))
    res = await client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": "hello"}]
    )
    assert res["messages"][0]["role"] == "system"
    assert BALANCED_INSTRUCTIONS in res["messages"][0]["content"]


# --- Anthropic integration path (system kwarg, never role="system") ----------


class _MockAnthropicMessages:
    def create(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return {"model": model, "messages": messages, "kwargs": kwargs}


class _MockAnthropicClient:
    def __init__(self) -> None:
        self.messages = _MockAnthropicMessages()


class _MockAsyncAnthropicMessages:
    async def create(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return {"model": model, "messages": messages, "kwargs": kwargs}


class _MockAsyncAnthropicClient:
    def __init__(self) -> None:
        self.messages = _MockAsyncAnthropicMessages()


def _assert_no_system_role(messages: list[dict[str, Any]]) -> None:
    roles = {m.get("role") for m in messages}
    assert "system" not in roles, f"role='system' must never enter Anthropic messages: {roles}"
    assert roles <= {"user", "assistant"}


def test_anthropic_reduction_creates_system_kwarg() -> None:
    client = with_tokenslim(_MockAnthropicClient(), config=Config(output_reduction="balanced"))
    res = client.messages.create(
        model="claude-sonnet-4-5", messages=[{"role": "user", "content": "hello"}]
    )
    assert res["kwargs"]["system"] == BALANCED_INSTRUCTIONS
    _assert_no_system_role(res["messages"])


def test_anthropic_reduction_appends_to_str_system() -> None:
    client = with_tokenslim(_MockAnthropicClient(), config=Config(output_reduction="aggressive"))
    res = client.messages.create(
        model="claude-sonnet-4-5",
        system="Be helpful.",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
            {"role": "user", "content": "continue"},
        ],
    )
    system = res["kwargs"]["system"]
    assert system.startswith("Be helpful.")
    assert AGGRESSIVE_INSTRUCTIONS in system
    _assert_no_system_role(res["messages"])


def test_anthropic_reduction_appends_text_block_to_list_system() -> None:
    original_system = [
        {"type": "text", "text": "Be helpful.", "cache_control": {"type": "ephemeral"}}
    ]
    snapshot = copy.deepcopy(original_system)
    client = with_tokenslim(_MockAnthropicClient(), config=Config(output_reduction="balanced"))
    res = client.messages.create(
        model="claude-sonnet-4-5",
        system=original_system,
        messages=[{"role": "user", "content": "hello"}],
    )
    system = res["kwargs"]["system"]
    assert system[0] == snapshot[0], "existing block (incl. cache_control) must be preserved"
    assert system[-1] == {"type": "text", "text": BALANCED_INSTRUCTIONS}
    assert original_system == snapshot, "caller's system list must not be mutated"
    _assert_no_system_role(res["messages"])


def test_anthropic_unrecognized_system_shape_skips_reduction() -> None:
    weird = {"unexpected": "shape"}
    client = with_tokenslim(_MockAnthropicClient(), config=Config(output_reduction="balanced"))
    res = client.messages.create(
        model="claude-sonnet-4-5", system=weird, messages=[{"role": "user", "content": "hello"}]
    )
    assert res["kwargs"]["system"] == {"unexpected": "shape"}
    _assert_no_system_role(res["messages"])


def test_anthropic_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOKENSLIM_OUTPUT_REDUCTION", raising=False)
    client = with_tokenslim(_MockAnthropicClient())
    res = client.messages.create(
        model="claude-sonnet-4-5", messages=[{"role": "user", "content": "hello"}]
    )
    assert "system" not in res["kwargs"]
    _assert_no_system_role(res["messages"])


@pytest.mark.asyncio
async def test_anthropic_async_reduction_creates_system_kwarg() -> None:
    client = with_tokenslim(_MockAsyncAnthropicClient(), config=Config(output_reduction="balanced"))
    res = await client.messages.create(
        model="claude-sonnet-4-5",
        system="Be helpful.",
        messages=[{"role": "user", "content": "hello"}],
    )
    system = res["kwargs"]["system"]
    assert system.startswith("Be helpful.")
    assert BALANCED_INSTRUCTIONS in system
    _assert_no_system_role(res["messages"])
