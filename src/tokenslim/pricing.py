"""Token pricing and cost estimation model.

Defines pricing tables for standard LLM models, computes USD savings from token
counts, and supports downloading updated pricing tables over HTTPS.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import TypedDict

__all__ = ["estimate_cost", "refresh_pricing", "load_pricing", "DEFAULT_PRICING"]


class ModelPrice(TypedDict):
    input_cost_per_million: float
    output_cost_per_million: float


# Default static pricing fallback (USD per 1 million tokens).
# Verified with current market rates for common providers.
DEFAULT_PRICING: dict[str, ModelPrice] = {
    "gpt-4o": {"input_cost_per_million": 2.50, "output_cost_per_million": 10.00},
    "gpt-4o-mini": {"input_cost_per_million": 0.15, "output_cost_per_million": 0.60},
    "gpt-4": {"input_cost_per_million": 30.00, "output_cost_per_million": 60.00},
    "gpt-3.5-turbo": {"input_cost_per_million": 0.50, "output_cost_per_million": 1.50},
    "claude-3-5-sonnet-latest": {
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
    },
    "claude-3-5-sonnet-20241022": {
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
    },
    "claude-3-5-haiku-latest": {
        "input_cost_per_million": 0.80,
        "output_cost_per_million": 4.00,
    },
    "claude-3-haiku-20240307": {
        "input_cost_per_million": 0.25,
        "output_cost_per_million": 1.25,
    },
    "gemini-1.5-flash": {"input_cost_per_million": 0.075, "output_cost_per_million": 0.30},
    "gemini-1.5-pro": {"input_cost_per_million": 1.25, "output_cost_per_million": 5.00},
}

PRICING_CACHE_FILE = "~/.config/tokenslim/pricing.json"


def get_pricing_cache_path() -> str:
    """Return the absolute path to the local pricing cache file."""
    path = os.path.expanduser(PRICING_CACHE_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def load_pricing() -> dict[str, ModelPrice]:
    """Load pricing tables from the local cache file, falling back to defaults."""
    cache_path = get_pricing_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Validate format
                    validated: dict[str, ModelPrice] = {}
                    for model, price in data.items():
                        if (
                            isinstance(price, dict)
                            and "input_cost_per_million" in price
                            and "output_cost_per_million" in price
                        ):
                            validated[model] = {
                                "input_cost_per_million": float(price["input_cost_per_million"]),
                                "output_cost_per_million": float(price["output_cost_per_million"]),
                            }
                    if validated:
                        return validated
        except Exception:
            pass
    return DEFAULT_PRICING


def refresh_pricing(
    url: str = "https://raw.githubusercontent.com/robertruben98/tokenslim/main/pricing.json",
) -> bool:
    """Download the latest pricing JSON table and save it to the cache folder."""
    cache_path = get_pricing_cache_path()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TokenSlim-CLI"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            if isinstance(data, dict):
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                return True
    except Exception:
        pass
    return False


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Estimate the cost in USD for a given model and token counts."""
    pricing = load_pricing()

    # Try exact match first
    price = pricing.get(model)
    if price is None:
        # Try substring match (e.g. "gpt-4o-2024-05-13" -> "gpt-4o")
        for key in pricing:
            if key in model or model in key:
                price = pricing[key]
                break

    if price is None:
        # Default to a safe general-purpose model's pricing
        price = pricing["gpt-4o"]

    in_cost = (input_tokens / 1_000_000.0) * price["input_cost_per_million"]
    out_cost = (output_tokens / 1_000_000.0) * price["output_cost_per_million"]
    return in_cost + out_cost
