"""Tests for tokenslim.predict — output-length prediction."""

from __future__ import annotations

import math

import pytest

from tokenslim import (
    OutputPrediction,
    extract_output_features,
    predict_output_tokens,
    suggest_max_tokens,
)
from tokenslim.predict import (
    DEFAULT_WEIGHTS,
    FEATURE_NAMES,
    MAX_PREDICTED_TOKENS,
    MIN_PREDICTED_TOKENS,
    load_weights,
    save_weights,
)


def _user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def test_yes_no_predicts_fewer_than_essay():
    yes_no = predict_output_tokens(_user("Is Python an interpreted language?"))
    essay = predict_output_tokens(
        _user("Write a comprehensive essay about the history of the Roman Empire.")
    )
    assert yes_no.tokens < essay.tokens, (yes_no.tokens, essay.tokens)
    assert yes_no.features["yes_no_question"] == 1.0
    assert essay.features["asks_detail"] == 1.0
    # The bounded yes/no shape should also carry higher confidence.
    assert yes_no.confidence > essay.confidence


def test_explicit_length_request_dominates():
    features = extract_output_features(_user("Summarize the plot of Hamlet in 200 words."))
    assert features["explicit_length_request"] == pytest.approx(200 * 1.3)
    # Its weighted contribution is the single largest term in the model.
    contributions = {
        name: abs(DEFAULT_WEIGHTS.get(name, 0.0) * value) for name, value in features.items()
    }
    assert max(contributions, key=contributions.get) == "explicit_length_request"

    short = predict_output_tokens(_user("Summarize the plot of Hamlet in 200 words."))
    long = predict_output_tokens(_user("Summarize the plot of Hamlet in 1000 words."))
    assert long.tokens - short.tokens > 500, (short.tokens, long.tokens)
    # Explicit length is a strong signal — confidence rises, band tightens.
    assert short.confidence >= 0.8
    assert short.high / short.tokens < 1.3


def test_explicit_length_units():
    para = extract_output_features(_user("Explain DNS in 3 paragraphs."))
    assert para["explicit_length_request"] == pytest.approx(3 * 80.0)
    lines = extract_output_features(_user("Give me 10 lines of output."))
    assert lines["explicit_length_request"] == pytest.approx(10 * 12.0)
    none = extract_output_features(_user("Explain DNS."))
    assert none["explicit_length_request"] == 0.0


def test_features_present_and_finite():
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Respond in JSON format."},
        {"role": "user", "content": [{"type": "text", "text": "Here is code:\n```py\nx=1\n```"}]},
        {"role": "assistant", "content": "Looks fine."},
        {"role": "user", "content": "Can you fix the bug in this function? Why does it fail?"},
    ]
    features = extract_output_features(messages)
    assert set(features) == set(FEATURE_NAMES)
    for name, value in features.items():
        assert isinstance(value, float), f"{name} is {type(value)}"
        assert math.isfinite(value), f"{name} is not finite: {value}"
    assert features["n_messages"] == 4.0
    assert features["has_code_fence"] == 1.0
    assert features["asks_for_code"] == 1.0  # "fix" + "function"/"bug"
    assert features["n_questions"] == 2.0
    assert features["json_schema_requested"] == 1.0  # from the system message
    assert features["yes_no_question"] == 1.0  # last user msg starts with "Can"
    assert features["system_tokens"] > 0.0
    assert features["prompt_tokens"] > features["last_user_tokens"] > 0.0


def test_brevity_and_list_flags():
    features = extract_output_features(_user("Briefly list the steps to install Arch Linux."))
    assert features["asks_brevity"] == 1.0
    assert features["asks_for_list"] == 1.0
    assert features["asks_detail"] == 0.0


def test_prediction_bounds_and_band():
    for msgs in ([], _user("hi"), _user("Write a comprehensive, detailed 5000 words report.")):
        pred = predict_output_tokens(msgs)
        assert MIN_PREDICTED_TOKENS <= pred.tokens <= MAX_PREDICTED_TOKENS
        assert pred.low <= pred.tokens <= pred.high
        assert 0.0 < pred.confidence < 1.0
        assert isinstance(pred, OutputPrediction)


def test_suggest_max_tokens_floor_and_cap():
    small = OutputPrediction(tokens=20, low=16, high=30, confidence=0.9, features={})
    assert suggest_max_tokens(small) == 64  # default floor
    assert suggest_max_tokens(small, floor=16) == 30  # high bound wins over tokens*safety

    big = OutputPrediction(tokens=6000, low=4000, high=9000, confidence=0.5, features={})
    assert suggest_max_tokens(big) == 9000  # under default cap
    assert suggest_max_tokens(big, cap=8000) == 8000  # cap respected
    assert suggest_max_tokens(big, safety=2.0) == 12000  # tokens*safety wins over high


def test_weights_round_trip(tmp_path):
    path = tmp_path / "calibrated.json"
    saved = save_weights(
        path,
        intercept=42.5,
        weights={"n_messages": 7.0, "prompt_tokens": 0.5},
        meta={"run": "gpu-calibration-1", "r2": 0.83},
    )
    assert saved == path
    blob = load_weights(path)
    assert blob["intercept"] == 42.5
    assert blob["weights"] == {"n_messages": 7.0, "prompt_tokens": 0.5}
    assert blob["meta"] == {"run": "gpu-calibration-1", "r2": 0.83}

    # The loaded blob drives the prediction: intercept + 7 * n_messages only
    # touches features we can compute exactly (prompt_tokens weight on "hi").
    w2 = save_weights(tmp_path / "w2.json", intercept=100.0, weights={"n_messages": 7.0})
    blob2 = load_weights(w2)
    pred = predict_output_tokens(_user("hi"), weights=blob2)
    assert pred.tokens == 107  # 100 + 7 * 1 message

    # Default template save also round-trips.
    default_blob = load_weights(save_weights(tmp_path / "defaults.json"))
    assert default_blob["weights"] == DEFAULT_WEIGHTS


def test_load_weights_rejects_malformed(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_weights(bad)
    bad.write_text('{"weights": {}}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_weights(bad)
    bad.write_text('{"intercept": "abc", "weights": {"prompt_tokens": 1.0}}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_weights(bad)
