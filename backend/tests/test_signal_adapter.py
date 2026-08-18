from datetime import datetime

import pytest

from backend.app.signals.adapter import MLSignalAdapter
from backend.app.signals.schema import SignalAction


def test_buy_signal():
    adapter = MLSignalAdapter()

    signal = adapter.adapt(
        timestamp=datetime(2026, 8, 18, 10, 0),
        symbol="RELIANCE",
        probability_down=0.10,
        probability_neutral=0.18,
        probability_up=0.72,
    )

    assert signal.action == SignalAction.BUY
    assert signal.confidence == 0.72


def test_sell_signal():
    adapter = MLSignalAdapter()

    signal = adapter.adapt(
        timestamp=datetime(2026, 8, 18, 10, 0),
        symbol="RELIANCE",
        probability_down=0.72,
        probability_neutral=0.18,
        probability_up=0.10,
    )

    assert signal.action == SignalAction.SELL
    assert signal.confidence == 0.72


def test_hold_signal():
    adapter = MLSignalAdapter()

    signal = adapter.adapt(
        timestamp=datetime(2026, 8, 18, 10, 0),
        symbol="RELIANCE",
        probability_down=0.30,
        probability_neutral=0.45,
        probability_up=0.25,
    )

    assert signal.action == SignalAction.HOLD
    assert signal.confidence == 0.45


def test_invalid_probability_is_rejected():
    adapter = MLSignalAdapter()

    with pytest.raises(ValueError):
        adapter.adapt(
            timestamp=datetime(2026, 8, 18, 10, 0),
            symbol="RELIANCE",
            probability_down=-0.10,
            probability_neutral=0.30,
            probability_up=0.80,
        )


def test_probabilities_must_sum_to_one():
    adapter = MLSignalAdapter()

    with pytest.raises(ValueError):
        adapter.adapt(
            timestamp=datetime(2026, 8, 18, 10, 0),
            symbol="RELIANCE",
            probability_down=0.10,
            probability_neutral=0.20,
            probability_up=0.80,
        )


def test_invalid_buy_threshold_is_rejected():
    with pytest.raises(ValueError):
        MLSignalAdapter(buy_threshold=1.5)


def test_invalid_sell_threshold_is_rejected():
    with pytest.raises(ValueError):
        MLSignalAdapter(sell_threshold=-0.1)


def test_thresholds_cannot_allow_buy_and_sell_simultaneously():
    with pytest.raises(ValueError):
        MLSignalAdapter(
            buy_threshold=0.40,
            sell_threshold=0.40,
        )


def test_custom_thresholds():
    adapter = MLSignalAdapter(
        buy_threshold=0.70,
        sell_threshold=0.70,
    )

    signal = adapter.adapt(
        timestamp=datetime(2026, 8, 18, 10, 0),
        symbol="TCS",
        probability_down=0.15,
        probability_neutral=0.10,
        probability_up=0.75,
    )

    assert signal.action == SignalAction.BUY


def test_multiple_symbols():
    adapter = MLSignalAdapter()

    reliance = adapter.adapt(
        timestamp=datetime(2026, 8, 18, 10, 0),
        symbol="RELIANCE",
        probability_down=0.10,
        probability_neutral=0.20,
        probability_up=0.70,
    )

    tcs = adapter.adapt(
        timestamp=datetime(2026, 8, 18, 10, 0),
        symbol="TCS",
        probability_down=0.70,
        probability_neutral=0.20,
        probability_up=0.10,
    )

    assert reliance.symbol == "RELIANCE"
    assert reliance.action == SignalAction.BUY

    assert tcs.symbol == "TCS"
    assert tcs.action == SignalAction.SELL


def test_output_is_deterministic():
    adapter = MLSignalAdapter()

    signal_1 = adapter.adapt(
        timestamp=datetime(2026, 8, 18, 10, 0),
        symbol="INFY",
        probability_down=0.10,
        probability_neutral=0.20,
        probability_up=0.70,
    )

    signal_2 = adapter.adapt(
        timestamp=datetime(2026, 8, 18, 10, 0),
        symbol="INFY",
        probability_down=0.10,
        probability_neutral=0.20,
        probability_up=0.70,
    )

    assert signal_1 == signal_2