"""Tests for src.ml.signal_adapter: ML Prediction → TradingSignal conversion.

Covers the BUY/SELL/HOLD rules, configurable thresholds, probability
validation, determinism, multi-symbol/timestamp handling, causality guarantees,
probability preservation, confidence semantics, and architectural separation
from the backtester.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.signal_adapter import (
    DEFAULT_ADAPTER,
    MLSignalAdapter,
    SignalThresholds,
)
from src.ml.signal_types import SignalAction, SignalSource, TradingSignal


# V1 default thresholds
DEFAULT_BUY = 0.55
DEFAULT_SELL = 0.55
DEFAULT_MIN_CONF = 0.50


def _make_adapter(buy=DEFAULT_BUY, sell=DEFAULT_SELL, model_name="gradient_boosting"):
    return MLSignalAdapter(
        thresholds=SignalThresholds(buy_threshold=buy, sell_threshold=sell),
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Basic conversion / action rules
# ---------------------------------------------------------------------------


class TestActionRules:
    def test_bullish_prediction_becomes_buy(self):
        adapter = _make_adapter()
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.09,
            prob_neutral=0.18,
            prob_up=0.73,
        )
        assert sig.action == SignalAction.BUY
        assert sig.confidence == 0.73

    def test_bearish_prediction_becomes_sell(self):
        adapter = _make_adapter()
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=-1,
            prob_down=0.71,
            prob_neutral=0.12,
            prob_up=0.17,
        )
        assert sig.action == SignalAction.SELL
        assert sig.confidence == 0.71

    def test_neutral_prediction_becomes_hold(self):
        adapter = _make_adapter()
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=0,
            prob_down=0.30,
            prob_neutral=0.50,
            prob_up=0.20,
        )
        assert sig.action == SignalAction.HOLD

    def test_insufficient_probability_becomes_hold_bullish(self):
        # y_pred = +1 but P(+1) below threshold → HOLD
        adapter = _make_adapter()
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.30,
            prob_neutral=0.35,
            prob_up=0.35,  # 0.35 < 0.55
        )
        assert sig.action == SignalAction.HOLD

    def test_insufficient_probability_becomes_hold_bearish(self):
        adapter = _make_adapter()
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=-1,
            prob_down=0.40,  # 0.40 < 0.55
            prob_neutral=0.30,
            prob_up=0.30,
        )
        assert sig.action == SignalAction.HOLD

    def test_plurality_does_not_force_buy(self):
        # P(+1) = 0.51 > 0.50 but below buy_threshold=0.55 → HOLD (not BUY)
        adapter = _make_adapter()
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.20,
            prob_neutral=0.29,
            prob_up=0.51,
        )
        assert sig.action == SignalAction.HOLD


# ---------------------------------------------------------------------------
# Threshold behavior (exact boundaries)
# ---------------------------------------------------------------------------


class TestThresholdBoundaries:
    def test_exact_buy_threshold(self):
        adapter = _make_adapter(buy=0.55)
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.10,
            prob_neutral=0.35,
            prob_up=0.55,  # exactly at threshold
        )
        assert sig.action == SignalAction.BUY

    def test_just_below_buy_threshold(self):
        adapter = _make_adapter(buy=0.55)
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.11,
            prob_neutral=0.34,
            prob_up=0.55,  # 0.55 is at threshold, use 0.549 for just below
        )
        # Need to adjust so sum = 1.0
        # 0.11 + 0.341 + 0.549 = 1.0
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.11,
            prob_neutral=0.341,
            prob_up=0.549,  # just below threshold
        )
        assert sig.action == SignalAction.HOLD

    def test_exact_sell_threshold(self):
        adapter = _make_adapter(sell=0.55)
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=-1,
            prob_down=0.55,  # exactly at threshold
            prob_neutral=0.25,
            prob_up=0.20,
        )
        assert sig.action == SignalAction.SELL

    def test_just_below_sell_threshold(self):
        adapter = _make_adapter(sell=0.55)
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=-1,
            prob_down=0.549,
            prob_neutral=0.25,
            prob_up=0.201,
        )
        assert sig.action == SignalAction.HOLD


# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------


class TestConfigurableThresholds:
    def test_custom_high_thresholds(self):
        adapter = _make_adapter(buy=0.80, sell=0.80)
        # P(+1)=0.73 below custom 0.80 → HOLD even though default would BUY
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.09,
            prob_neutral=0.18,
            prob_up=0.73,
        )
        assert sig.action == SignalAction.HOLD

    def test_custom_low_thresholds(self):
        # Custom thresholds with min_confidence=0.40
        thresholds = SignalThresholds(buy_threshold=0.40, sell_threshold=0.40, min_confidence=0.40)
        adapter = MLSignalAdapter(thresholds=thresholds, model_name="test")
        # P(+1)=0.51 above custom 0.40 → BUY
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.20,
            prob_neutral=0.29,
            prob_up=0.51,
        )
        assert sig.action == SignalAction.BUY

    def test_threshold_validation_out_of_range(self):
        with pytest.raises(ValueError, match="buy_threshold"):
            SignalThresholds(buy_threshold=1.5)
        with pytest.raises(ValueError, match="sell_threshold"):
            SignalThresholds(sell_threshold=-0.1)
        with pytest.raises(ValueError, match="min_confidence"):
            SignalThresholds(min_confidence=2.0)

    def test_threshold_validation_buy_below_min(self):
        with pytest.raises(ValueError, match="buy_threshold"):
            SignalThresholds(buy_threshold=0.40, min_confidence=0.50)

    def test_threshold_validation_sell_below_min(self):
        with pytest.raises(ValueError, match="sell_threshold"):
            SignalThresholds(sell_threshold=0.40, min_confidence=0.50)


# ---------------------------------------------------------------------------
# Probability validation
# ---------------------------------------------------------------------------


class TestProbabilityValidation:
    def test_missing_probability_raises(self):
        adapter = _make_adapter()
        with pytest.raises(ValueError):
            adapter.convert(
                symbol="AAPL",
                timestamp=pd.Timestamp("2024-01-02"),
                y_pred=1,
                prob_down=None,
                prob_neutral=0.18,
                prob_up=0.73,
            )

    def test_nan_probability_raises(self):
        adapter = _make_adapter()
        with pytest.raises(ValueError, match="NaN or infinity"):
            adapter.convert(
                symbol="AAPL",
                timestamp=pd.Timestamp("2024-01-02"),
                y_pred=1,
                prob_down=np.nan,
                prob_neutral=0.18,
                prob_up=0.73,
            )

    def test_infinite_probability_raises(self):
        adapter = _make_adapter()
        with pytest.raises(ValueError, match="NaN or infinity"):
            adapter.convert(
                symbol="AAPL",
                timestamp=pd.Timestamp("2024-01-02"),
                y_pred=1,
                prob_down=np.inf,
                prob_neutral=0.18,
                prob_up=0.73,
            )

    def test_probability_outside_range_high_raises(self):
        adapter = _make_adapter()
        with pytest.raises(ValueError, match="outside"):
            adapter.convert(
                symbol="AAPL",
                timestamp=pd.Timestamp("2024-01-02"),
                y_pred=1,
                prob_down=0.09,
                prob_neutral=0.18,
                prob_up=1.73,  # > 1
            )

    def test_probability_outside_range_negative_raises(self):
        adapter = _make_adapter()
        with pytest.raises(ValueError, match="outside"):
            adapter.convert(
                symbol="AAPL",
                timestamp=pd.Timestamp("2024-01-02"),
                y_pred=-1,
                prob_down=-0.10,  # < 0
                prob_neutral=0.18,
                prob_up=0.73,
            )

    def test_inconsistent_probability_vector_raises(self):
        # Sum != 1.0
        adapter = _make_adapter()
        with pytest.raises(ValueError, match="sum to"):
            adapter.convert(
                symbol="AAPL",
                timestamp=pd.Timestamp("2024-01-02"),
                y_pred=1,
                prob_down=0.30,
                prob_neutral=0.30,
                prob_up=0.30,  # sums to 0.90
            )

    def test_valid_probability_vector_accepted(self):
        adapter = _make_adapter()
        # Sums to 1.0 exactly
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.20,
            prob_neutral=0.30,
            prob_up=0.50,
        )
        assert sig.probability_down == 0.20
        assert sig.probability_neutral == 0.30
        assert sig.probability_up == 0.50


# ---------------------------------------------------------------------------
# Predicted class validation
# ---------------------------------------------------------------------------


class TestPredictedClassValidation:
    def test_invalid_predicted_class_raises(self):
        adapter = _make_adapter()
        with pytest.raises(ValueError, match="y_pred"):
            adapter.convert(
                symbol="AAPL",
                timestamp=pd.Timestamp("2024-01-02"),
                y_pred=5,  # not in {-1, 0, +1}
                prob_down=0.09,
                prob_neutral=0.18,
                prob_up=0.73,
            )

    def test_missing_predicted_class_raises(self):
        adapter = _make_adapter()
        with pytest.raises(ValueError, match="y_pred"):
            adapter.convert(
                symbol="AAPL",
                timestamp=pd.Timestamp("2024-01-02"),
                y_pred=None,
                prob_down=0.09,
                prob_neutral=0.18,
                prob_up=0.73,
            )


# ---------------------------------------------------------------------------
# Required field validation
# ---------------------------------------------------------------------------


class TestRequiredFields:
    def test_missing_symbol_raises(self):
        adapter = _make_adapter()
        with pytest.raises(ValueError, match="symbol"):
            adapter.convert(
                symbol="",
                timestamp=pd.Timestamp("2024-01-02"),
                y_pred=1,
                prob_down=0.09,
                prob_neutral=0.18,
                prob_up=0.73,
            )

    def test_none_symbol_raises(self):
        adapter = _make_adapter()
        with pytest.raises(ValueError, match="symbol"):
            adapter.convert(
                symbol=None,
                timestamp=pd.Timestamp("2024-01-02"),
                y_pred=1,
                prob_down=0.09,
                prob_neutral=0.18,
                prob_up=0.73,
            )

    def test_missing_timestamp_raises(self):
        adapter = _make_adapter()
        with pytest.raises(ValueError, match="timestamp"):
            adapter.convert(
                symbol="AAPL",
                timestamp=None,
                y_pred=1,
                prob_down=0.09,
                prob_neutral=0.18,
                prob_up=0.73,
            )


# ---------------------------------------------------------------------------
# Multi-symbol / multi-timestamp
# ---------------------------------------------------------------------------


class TestMultipleSymbolsTimestamps:
    def test_multiple_symbols(self):
        adapter = _make_adapter()
        symbols = ["AAPL", "MSFT", "JPM"]
        signals = []
        for sym in symbols:
            sig = adapter.convert(
                symbol=sym,
                timestamp=pd.Timestamp("2024-01-02"),
                y_pred=1,
                prob_down=0.09,
                prob_neutral=0.18,
                prob_up=0.73,
            )
            signals.append(sig)
        assert [s.symbol for s in signals] == symbols
        assert all(s.action == SignalAction.BUY for s in signals)

    def test_multiple_timestamps(self):
        adapter = _make_adapter()
        ts = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
        signals = []
        for t in ts:
            sig = adapter.convert(
                symbol="AAPL",
                timestamp=t,
                y_pred=-1,
                prob_down=0.71,
                prob_neutral=0.12,
                prob_up=0.17,
            )
            signals.append(sig)
        assert [s.timestamp for s in signals] == ts
        assert all(s.action == SignalAction.SELL for s in signals)

    def test_batch_conversion_multiple_symbols(self):
        adapter = _make_adapter()
        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "JPM"],
                "timestamp": [
                    pd.Timestamp("2024-01-02"),
                    pd.Timestamp("2024-01-02"),
                    pd.Timestamp("2024-01-02"),
                ],
                "y_pred": [1, -1, 0],
                "y_prob_-1": [0.09, 0.71, 0.30],
                "y_prob_0": [0.18, 0.12, 0.50],
                "y_prob_1": [0.73, 0.17, 0.20],
            }
        )
        signals = adapter.convert_batch(df)
        assert len(signals) == 3
        assert signals[0].action == SignalAction.BUY
        assert signals[1].action == SignalAction.SELL
        assert signals[2].action == SignalAction.HOLD
        assert [s.symbol for s in signals] == ["AAPL", "MSFT", "JPM"]

    def test_batch_missing_column_raises(self):
        adapter = _make_adapter()
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "timestamp": [pd.Timestamp("2024-01-02")],
                "y_pred": [1],
                "y_prob_-1": [0.09],
                "y_prob_0": [0.18],
                # y_prob_1 missing
            }
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            adapter.convert_batch(df)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_output(self):
        adapter = _make_adapter()
        kwargs = dict(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.09,
            prob_neutral=0.18,
            prob_up=0.73,
        )
        sig1 = adapter.convert(**kwargs)
        sig2 = adapter.convert(**kwargs)
        assert sig1 == sig2

    def test_batch_preserves_order(self):
        adapter = _make_adapter()
        rows = []
        for i in range(5):
            rows.append(
                {
                    "symbol": f"SYM{i}",
                    "timestamp": pd.Timestamp("2024-01-02"),
                    "y_pred": [1, -1, 0, 1, -1][i],
                    "y_prob_-1": [0.1, 0.7, 0.3, 0.1, 0.6][i],
                    "y_prob_0": [0.2, 0.1, 0.5, 0.2, 0.2][i],
                    "y_prob_1": [0.7, 0.2, 0.2, 0.7, 0.2][i],
                }
            )
        df = pd.DataFrame(rows)
        signals = adapter.convert_batch(df)
        assert [s.symbol for s in signals] == [f"SYM{i}" for i in range(5)]
        assert [s.action for s in signals] == [
            SignalAction.BUY,
            SignalAction.SELL,
            SignalAction.HOLD,
            SignalAction.BUY,
            SignalAction.SELL,
        ]


# ---------------------------------------------------------------------------
# Probability preservation
# ---------------------------------------------------------------------------


class TestProbabilityPreservation:
    def test_all_probabilities_preserved(self):
        adapter = _make_adapter()
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.09,
            prob_neutral=0.18,
            prob_up=0.73,
        )
        assert sig.probability_down == 0.09
        assert sig.probability_neutral == 0.18
        assert sig.probability_up == 0.73

    def test_probabilities_preserved_on_hold(self):
        adapter = _make_adapter()
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.30,
            prob_neutral=0.35,
            prob_up=0.35,
        )
        assert sig.probability_down == 0.30
        assert sig.probability_neutral == 0.35
        assert sig.probability_up == 0.35


# ---------------------------------------------------------------------------
# Confidence semantics
# ---------------------------------------------------------------------------


class TestConfidenceSemantics:
    def test_buy_confidence_is_up_probability(self):
        adapter = _make_adapter()
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.09,
            prob_neutral=0.18,
            prob_up=0.73,
        )
        assert sig.confidence == 0.73

    def test_sell_confidence_is_down_probability(self):
        adapter = _make_adapter()
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=-1,
            prob_down=0.71,
            prob_neutral=0.12,
            prob_up=0.17,
        )
        assert sig.confidence == 0.71

    def test_hold_confidence_is_max_probability(self):
        adapter = _make_adapter()
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=0,
            prob_down=0.30,
            prob_neutral=0.50,
            prob_up=0.20,
        )
        # HOLD confidence = max probability
        assert sig.confidence == 0.50


# ---------------------------------------------------------------------------
# Causality / no future-data use
# ---------------------------------------------------------------------------


class TestCausality:
    def test_realized_return_is_not_used(self):
        adapter = _make_adapter()
        # Provide a misleading realized_return — should NOT affect the signal
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.09,
            prob_neutral=0.18,
            prob_up=0.73,
            realized_return=-0.99,  # very negative next-session return
        )
        assert sig.action == SignalAction.BUY  # still BUY based on prediction only
        assert sig.confidence == 0.73

    def test_no_future_data_dependency(self):
        adapter = _make_adapter()
        # Calling convert with only legitimate-at-signal-time data works fine
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.09,
            prob_neutral=0.18,
            prob_up=0.73,
        )
        # The signal has no y_true / realized_return fields
        assert not hasattr(sig, "y_true")
        assert not hasattr(sig, "realized_return")


# ---------------------------------------------------------------------------
# Metadata: source / model
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_source_is_ml(self):
        adapter = _make_adapter(model_name="gradient_boosting")
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.09,
            prob_neutral=0.18,
            prob_up=0.73,
        )
        assert sig.source == SignalSource.ML
        assert sig.model_name == "gradient_boosting"

    def test_default_adapter_source(self):
        sig = DEFAULT_ADAPTER.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.09,
            prob_neutral=0.18,
            prob_up=0.73,
        )
        assert sig.source == SignalSource.ML


# ---------------------------------------------------------------------------
# Architectural separation
# ---------------------------------------------------------------------------


class TestArchitecturalSeparation:
    def test_adapter_does_not_import_backtester(self):
        import inspect
        import src.ml.signal_adapter as sa

        source = inspect.getsource(sa)
        assert "backend.app.backtesting" not in source
        assert "from backtester" not in source
        assert "import backtester" not in source
        assert "position" not in source
        assert "slippage" not in source
        assert "transaction_cost" not in source
        assert "equity_curve" not in source

    def test_signal_has_no_execution_fields(self):
        adapter = _make_adapter()
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.09,
            prob_neutral=0.18,
            prob_up=0.73,
        )
        # No execution / risk / portfolio fields
        for forbidden in ["position", "cash", "order", "slippage",
                         "transaction_cost", "equity", "portfolio", "risk"]:
            assert forbidden not in sig.model_dump()


# ---------------------------------------------------------------------------
# TradingSignal model
# ---------------------------------------------------------------------------


class TestTradingSignalModel:
    def test_signal_dump_serializes_enums(self):
        adapter = _make_adapter(model_name="logistic_regression")
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.09,
            prob_neutral=0.18,
            prob_up=0.73,
        )
        dump = sig.model_dump()
        assert dump["action"] == "BUY"
        assert dump["source"] == "ML"
        assert dump["model_name"] == "logistic_regression"

    def test_signal_confidence_bounds(self):
        adapter = _make_adapter()
        sig = adapter.convert(
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-02"),
            y_pred=1,
            prob_down=0.09,
            prob_neutral=0.18,
            prob_up=0.73,
        )
        assert 0.0 <= sig.confidence <= 1.0
