import copy

import pytest

from tokenslim.cache import (
    PrefixCacheReport,
    find_volatile_spans,
    insert_anthropic_cache_control,
    normalize_dynamic_content,
    optimize_for_prefix_cache,
    stabilize_message_order,
)


def test_normalize_dynamic_content() -> None:
    text = (
        "Requested at 2026-06-24T11:26:29Z by user-uuid "
        "550e8400-e29b-41d4-a716-446655440000 with token "
        "gho_abcdefghijklmnopqrstuvwxyz1234567890ABCD and hash "
        "d41d8cd98f00b204e9800998ecf8427e. Epoch time is 1719221189."
    )
    normalized = normalize_dynamic_content(text)
    assert "<UUID>" in normalized
    assert "<TOKEN>" in normalized
    assert "<TIMESTAMP>" in normalized
    assert "<HASH>" in normalized
    # The static parts are still there
    assert "Requested at" in normalized
    assert "by user-uuid" in normalized


def test_insert_anthropic_cache_control_system_string() -> None:
    system = "A" * 3000
    messages = [
        {"role": "user", "content": "hello"},
    ]
    new_msgs, new_sys = insert_anthropic_cache_control(messages, system=system, min_bytes=2000)
    assert isinstance(new_sys, list)
    assert new_sys[0]["cache_control"] == {"type": "ephemeral"}
    assert new_sys[0]["text"] == system


def test_insert_anthropic_cache_control_system_list() -> None:
    system = [
        {"type": "text", "text": "system header"},
        {"type": "text", "text": "B" * 3000},
    ]
    messages = [
        {"role": "user", "content": "hello"},
    ]
    new_msgs, new_sys = insert_anthropic_cache_control(messages, system=system, min_bytes=2000)
    assert isinstance(new_sys, list)
    assert new_sys[0].get("cache_control") is None
    assert new_sys[1]["cache_control"] == {"type": "ephemeral"}


def test_insert_anthropic_cache_control_messages() -> None:
    messages = [
        {"role": "user", "content": "small content"},
        {"role": "assistant", "content": "C" * 3000},
        {"role": "user", "content": "D" * 3000},
    ]
    new_msgs, new_sys = insert_anthropic_cache_control(messages, min_bytes=2000, max_breakpoints=2)

    # Handled messages backward:
    # 1. messages[2] ("D" * 3000) gets cache_control
    assert isinstance(new_msgs[2]["content"], list)
    assert new_msgs[2]["content"][0]["cache_control"] == {"type": "ephemeral"}

    # 2. messages[1] ("C" * 3000) gets cache_control
    assert isinstance(new_msgs[1]["content"], list)
    assert new_msgs[1]["content"][0]["cache_control"] == {"type": "ephemeral"}

    # 3. messages[0] is too small, doesn't get cache_control
    assert new_msgs[0]["content"] == "small content"


# --- stabilize_message_order ---


def test_stabilize_message_order_hoists_system() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "bye"},
    ]
    ordered = stabilize_message_order(messages)
    assert [m["role"] for m in ordered] == ["system", "user", "assistant", "user"]
    # System dict is the same object, just hoisted.
    assert ordered[0] is messages[2]


def test_stabilize_message_order_preserves_conversation_flow() -> None:
    messages = [
        {"role": "user", "content": "run the tool"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        {"role": "system", "content": "late system note"},
        {"role": "assistant", "content": "done"},
    ]
    ordered = stabilize_message_order(messages)
    non_system = [m for m in ordered if m.get("role") != "system"]
    assert non_system == [messages[0], messages[1], messages[2], messages[4]]
    # tool result still directly follows its tool_calls message
    idx = non_system.index(messages[1])
    assert non_system[idx + 1] is messages[2]


def test_stabilize_message_order_no_system_is_noop_copy() -> None:
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    ordered = stabilize_message_order(messages)
    assert ordered == messages
    assert ordered is not messages


# --- optimize_for_prefix_cache ---


def _stable_system(reps: int = 400) -> dict[str, object]:
    return {"role": "system", "content": "You are a helpful assistant. " * reps}


def test_optimize_for_prefix_cache_openai_cacheable() -> None:
    messages = [
        {"role": "user", "content": "question"},
        _stable_system(),
    ]
    optimized, report = optimize_for_prefix_cache(messages, provider="openai")
    assert isinstance(report, PrefixCacheReport)
    assert report.provider == "openai"
    assert report.stable_prefix_tokens >= 1024
    assert report.cacheable is True
    assert optimized[0]["role"] == "system"
    assert any("hoisted" in h for h in report.hints)


def test_optimize_for_prefix_cache_below_threshold() -> None:
    messages = [
        {"role": "system", "content": "short system"},
        {"role": "user", "content": "question"},
    ]
    optimized, report = optimize_for_prefix_cache(messages, provider="openai")
    assert report.cacheable is False
    assert report.stable_prefix_tokens < 1024
    assert any("1024" in h for h in report.hints)


def test_optimize_for_prefix_cache_google_threshold() -> None:
    big = [_stable_system(), {"role": "user", "content": "q"}]
    _, report_big = optimize_for_prefix_cache(big, provider="google")
    assert report_big.provider == "google"
    assert report_big.cacheable is True

    small = [{"role": "system", "content": "tiny"}, {"role": "user", "content": "q"}]
    _, report_small = optimize_for_prefix_cache(small, provider="google")
    assert report_small.cacheable is False


def test_optimize_for_prefix_cache_anthropic_delegates_breakpoints() -> None:
    messages = [_stable_system(), {"role": "user", "content": "q"}]
    optimized, report = optimize_for_prefix_cache(messages, provider="anthropic")
    assert report.provider == "anthropic"
    assert report.cacheable is True
    # insert_anthropic_cache_control converted the big system content to blocks
    content = optimized[0]["content"]
    assert isinstance(content, list)
    assert content[0]["cache_control"] == {"type": "ephemeral"}


def test_optimize_for_prefix_cache_normalizes_system_only() -> None:
    volatile = "Session 550e8400-e29b-41d4-a716-446655440000 started 2026-06-24T11:26:29Z. "
    messages = [
        {"role": "system", "content": volatile + "Rules apply. " * 200},
        {"role": "user", "content": "my id is 550e8400-e29b-41d4-a716-446655440000"},
    ]
    optimized, report = optimize_for_prefix_cache(messages, provider="openai")
    # System prompt normalized
    assert "<UUID>" in optimized[0]["content"]
    assert "<TIMESTAMP>" in optimized[0]["content"]
    # Conversation turns never rewritten
    assert optimized[1]["content"] == messages[1]["content"]
    # Volatility surfaced as actionable hints
    assert any("timestamp" in h and "system prompt" in h for h in report.hints)
    assert any("uuid" in h for h in report.hints)


def test_optimize_for_prefix_cache_does_not_mutate_input() -> None:
    messages = [
        {"role": "user", "content": "q one"},
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "id 550e8400-e29b-41d4-a716-446655440000 " + "x" * 3000}
            ],
        },
        {"role": "assistant", "content": "a one"},
    ]
    snapshot = copy.deepcopy(messages)
    for provider in ("openai", "google", "anthropic"):
        optimize_for_prefix_cache(messages, provider=provider)
    assert messages == snapshot


def test_optimize_for_prefix_cache_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        optimize_for_prefix_cache([], provider="azure")


# --- find_volatile_spans ---


def test_find_volatile_spans_reports_without_rewriting() -> None:
    text = (
        "at 2026-06-24T11:26:29Z uuid 550e8400-e29b-41d4-a716-446655440000 "
        "token gho_abcdefghijklmnopqrstuvwxyz1234567890ABCD "
        "hash d41d8cd98f00b204e9800998ecf8427e epoch 1719221189"
    )
    spans = find_volatile_spans(text)
    kinds = {s.kind for s in spans}
    assert kinds == {"timestamp", "uuid", "token", "hash"}
    # Sorted by position; each span's text matches the original slice
    starts = [s.start for s in spans]
    assert starts == sorted(starts)
    for span in spans:
        assert text[span.start : span.end] == span.text


def test_find_volatile_spans_clean_text() -> None:
    assert find_volatile_spans("nothing volatile here, just prose") == ()
