"""Token pricing / cost model.

A per-model price table and a cost estimator that powers the savings reports.
Prices are USD per 1,000,000 tokens, kept in a plain dict so the table can be
refreshed (overridden / extended) at runtime without touching code — providers
change prices often.

All numbers are *offline reference values* baked into the library; nothing here
makes a network call. Update :data:`PRICES` (or pass a custom table to
:func:`estimate_cost`) to keep current.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ModelPrice", "PRICES", "estimate_cost", "register_price", "get_price"]


@dataclass(frozen=True)
class ModelPrice:
    """USD price per 1,000,000 tokens for a model."""

    input: float
    output: float
    # Price for cached-prompt input tokens (cache reads); falls back to input.
    cached_input: float | None = None

    def cached_or_input(self) -> float:
        return self.cached_input if self.cached_input is not None else self.input


# Reference prices (USD per 1M tokens). Approximate, offline; refresh as needed.
PRICES: dict[str, ModelPrice] = {
    # Anthropic
    "claude-opus-4": ModelPrice(input=15.0, output=75.0, cached_input=1.5),
    "claude-sonnet-4": ModelPrice(input=3.0, output=15.0, cached_input=0.3),
    "claude-haiku-4": ModelPrice(input=0.8, output=4.0, cached_input=0.08),
    "claude-3-5-sonnet": ModelPrice(input=3.0, output=15.0, cached_input=0.3),
    "claude-3-5-haiku": ModelPrice(input=0.8, output=4.0, cached_input=0.08),
    # OpenAI
    "gpt-4o": ModelPrice(input=2.5, output=10.0, cached_input=1.25),
    "gpt-4o-mini": ModelPrice(input=0.15, output=0.6, cached_input=0.075),
    "gpt-4-turbo": ModelPrice(input=10.0, output=30.0),
    "o1": ModelPrice(input=15.0, output=60.0, cached_input=7.5),
    "o3-mini": ModelPrice(input=1.1, output=4.4, cached_input=0.55),
    # Generic fallback used when a model isn't in the table.
    "default": ModelPrice(input=3.0, output=15.0, cached_input=0.3),
}

_PER_TOKEN = 1_000_000.0


def register_price(model: str, price: ModelPrice) -> None:
    """Add or override the price for ``model`` (runtime table refresh)."""
    PRICES[model] = price


def get_price(model: str | None, prices: dict[str, ModelPrice] | None = None) -> ModelPrice:
    """Return the price entry for ``model``, falling back to ``default``.

    Matching is exact first, then a longest-prefix match (so dated snapshots
    like ``gpt-4o-2024-08-06`` resolve to ``gpt-4o``), then ``default``.
    """
    table = prices if prices is not None else PRICES
    if model:
        if model in table:
            return table[model]
        prefix_matches = [key for key in table if key != "default" and model.startswith(key)]
        if prefix_matches:
            return table[max(prefix_matches, key=len)]
    return table["default"]


def estimate_cost(
    model: str | None,
    input_tokens: int,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    prices: dict[str, ModelPrice] | None = None,
) -> float:
    """Estimate the USD cost of a request.

    Args:
        model: Model name (resolved via :func:`get_price`).
        input_tokens: Prompt tokens billed at the full input rate.
        output_tokens: Completion tokens.
        cached_tokens: Prompt tokens served from cache (billed at the cache
            rate); these are counted *in addition* to ``input_tokens``.
    """
    price = get_price(model, prices)
    cost = (
        input_tokens * price.input
        + output_tokens * price.output
        + cached_tokens * price.cached_or_input()
    )
    return cost / _PER_TOKEN
