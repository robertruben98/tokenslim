from tokenslim.cache import insert_anthropic_cache_control, normalize_dynamic_content


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
