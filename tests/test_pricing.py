import pytest

from tokenslim.pricing import (
    ModelPrice,
    estimate_cost,
    get_price,
    register_price,
)


def test_known_model_cost():
    # gpt-4o: $2.5/1M input, $10/1M output.
    cost = estimate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(2.5)
    cost2 = estimate_cost("gpt-4o", input_tokens=0, output_tokens=1_000_000)
    assert cost2 == pytest.approx(10.0)


def test_cached_tokens_use_cache_rate():
    cost = estimate_cost("gpt-4o", input_tokens=0, output_tokens=0, cached_tokens=1_000_000)
    assert cost == pytest.approx(1.25)


def test_unknown_model_falls_back_to_default():
    cost = estimate_cost("totally-made-up-model", 1_000_000, 0)
    assert cost == pytest.approx(3.0)  # default input rate


def test_prefix_match_resolves_dated_snapshot():
    price = get_price("gpt-4o-2024-08-06")
    assert price is get_price("gpt-4o")


def test_longest_prefix_wins():
    # "gpt-4o-mini-..." should resolve to gpt-4o-mini, not gpt-4o.
    assert get_price("gpt-4o-mini-2024") is get_price("gpt-4o-mini")


def test_zero_cost_for_zero_tokens():
    assert estimate_cost("gpt-4o", 0, 0) == 0.0


def test_register_price_overrides():
    register_price("custom-x", ModelPrice(input=1.0, output=2.0))
    assert estimate_cost("custom-x", 1_000_000, 0) == pytest.approx(1.0)


def test_cached_falls_back_to_input_when_unset():
    p = ModelPrice(input=5.0, output=10.0)
    assert p.cached_or_input() == 5.0
