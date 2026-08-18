import pytest
from pydantic import ValidationError

from backend.app.signals.schema import TradingSignal


def valid_signal():
    return {
        "timestamp": "2026-08-16T10:00:00",
        "symbol": "RELIANCE",
        "action": "BUY",
        "confidence": 0.72,
        "source": "ML",
        "model": "RandomForest_v1",
        "probability_down": 0.08,
        "probability_neutral": 0.20,
        "probability_up": 0.72,
    }


def test_valid_trading_signal():
    signal = TradingSignal(**valid_signal())

    assert signal.symbol == "RELIANCE"
    assert signal.action.value == "BUY"
    assert signal.confidence == 0.72


def test_probabilities_must_sum_to_one():
    data = valid_signal()
    data["probability_up"] = 0.60

    with pytest.raises(ValidationError):
        TradingSignal(**data)


def test_confidence_must_match_highest_probability():
    data = valid_signal()
    data["confidence"] = 0.80

    with pytest.raises(ValidationError):
        TradingSignal(**data)


def test_probabilities_cannot_be_negative():
    data = valid_signal()
    data["probability_down"] = -0.10

    with pytest.raises(ValidationError):
        TradingSignal(**data)


def test_probabilities_cannot_exceed_one():
    data = valid_signal()
    data["probability_up"] = 1.10

    with pytest.raises(ValidationError):
        TradingSignal(**data)


def test_invalid_action_is_rejected():
    data = valid_signal()
    data["action"] = "MAYBE"

    with pytest.raises(ValidationError):
        TradingSignal(**data)


def test_hold_signal():
    data = valid_signal()
    data["action"] = "HOLD"
    data["probability_down"] = 0.15
    data["probability_neutral"] = 0.72
    data["probability_up"] = 0.13
    data["confidence"] = 0.72

    signal = TradingSignal(**data)

    assert signal.action.value == "HOLD"