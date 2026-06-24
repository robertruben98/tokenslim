import os
import urllib.request
from unittest.mock import MagicMock

import pytest

from tokenslim.pricing import (
    DEFAULT_PRICING,
    estimate_cost,
    get_pricing_cache_path,
    load_pricing,
    refresh_pricing,
)


@pytest.fixture(autouse=True)
def clean_cache():
    cache_path = get_pricing_cache_path()
    if os.path.exists(cache_path):
        os.remove(cache_path)
    yield
    if os.path.exists(cache_path):
        os.remove(cache_path)


def test_estimate_cost_defaults():
    # gpt-4o input: 2.50 / 1M, output: 10.00 / 1M
    # 1M input + 1M output = $12.50
    cost = estimate_cost("gpt-4o", 1_000_000, 1_000_000)
    assert cost == 12.50

    # gpt-4o-mini input: 0.15 / 1M, output: 0.60 / 1M
    # 2M input + 1M output = 0.30 + 0.60 = 0.90
    cost = estimate_cost("gpt-4o-mini", 2_000_000, 1_000_000)
    assert cost == pytest.approx(0.90)


def test_estimate_cost_substring_match():
    # "gpt-4o-2024-05-13" matches "gpt-4o"
    cost = estimate_cost("gpt-4o-2024-05-13", 1_000_000)
    assert cost == 2.50


def test_load_pricing_fallback():
    pricing = load_pricing()
    assert pricing == DEFAULT_PRICING


def test_refresh_pricing_success(monkeypatch):
    mock_response = MagicMock()
    mock_response.read.return_value = (
        b'{"custom-model": {"input_cost_per_million": 1.0, "output_cost_per_million": 2.0}}'
    )
    mock_response.__enter__.return_value = mock_response

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: mock_response)

    success = refresh_pricing("http://example.com/pricing.json")
    assert success is True

    pricing = load_pricing()
    assert "custom-model" in pricing
    assert pricing["custom-model"]["input_cost_per_million"] == 1.0
    assert pricing["custom-model"]["output_cost_per_million"] == 2.0

    cost = estimate_cost("custom-model", 1_000_000, 1_000_000)
    assert cost == 3.0
