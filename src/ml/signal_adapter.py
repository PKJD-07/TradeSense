"""ML Signal Adapter: converts ML predictions into TradingSignal objects.

This module bridges the ML prediction layer and the common TradingSignal
abstraction. It has NO dependency on the backtester, trading engine, or
any execution/risk logic. Its ONLY responsibility is:

    ML Prediction (y_pred, y_prob_*, symbol, timestamp) → TradingSignal

Architectural guarantees:
- No import of backtester, risk, portfolio, or execution modules.
- No use of realized_return, y_true, future prices, or test labels.
- Operates only on model prediction + information available at signal time.
- Deterministic: preserves timestamps, symbols, ordering.
- Configurable thresholds with explicit validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.ml.signal_types import SignalAction, SignalSource, TradingSignal


@dataclass(frozen=True)
class SignalThresholds:
    """Configurable thresholds for BUY/SELL/HOLD decisions.

    Attributes:
        buy_threshold: Minimum P(+1) required for BUY when y_pred == +1.
        sell_threshold: Minimum P(-1) required for SELL when y_pred == -1.
        min_confidence: Absolute floor for confidence (applies to all actions).
    """

    buy_threshold: float = 0.55
    sell_threshold: float = 0.55
    min_confidence: float = 0.50

    def __post_init__(self) -> None:
        """Validate threshold configuration."""
        if not 0.0 <= self.buy_threshold <= 1.0:
            raise ValueError(f"buy_threshold must be in [0, 1], got {self.buy_threshold}")
        if not 0.0 <= self.sell_threshold <= 1.0:
            raise ValueError(f"sell_threshold must be in [0, 1], got {self.sell_threshold}")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {self.min_confidence}")
        if self.buy_threshold < self.min_confidence:
            raise ValueError(
                f"buy_threshold ({self.buy_threshold}) < min_confidence ({self.min_confidence})"
            )
        if self.sell_threshold < self.min_confidence:
            raise ValueError(
                f"sell_threshold ({self.sell_threshold}) < min_confidence ({self.min_confidence})"
            )


class MLSignalAdapter:
    """Convert ML model predictions to TradingSignal objects.

    The adapter takes ML prediction output (symbol, timestamp, y_pred, y_prob_-1,
    y_prob_0, y_prob_1, optionally realized_return) and produces a TradingSignal
    using configurable probability thresholds.

    BUY rule:
        y_pred == +1 AND P(+1) >= buy_threshold

    SELL rule:
        y_pred == -1 AND P(-1) >= sell_threshold

    HOLD rule:
        otherwise (including when confidence < min_confidence)

    All three class probabilities are preserved in the output signal.
    Confidence equals the probability of the predicted class when the action
    is BUY or SELL; for HOLD it equals max(P(+1), P(-1), P(0)).
    """

    # Canonical probability column names from the ML pipeline
    PROB_COLUMNS = ("y_prob_-1", "y_prob_0", "y_prob_1")
    TARGET_CLASSES = (-1, 0, 1)

    def __init__(
        self,
        thresholds: Optional[SignalThresholds] = None,
        model_name: Optional[str] = None,
    ):
        """Initialize the adapter.

        Args:
            thresholds: SignalThresholds instance. Uses V1 defaults if None.
            model_name: Name of the ML model (e.g., "gradient_boosting").
        """
        self.thresholds = thresholds or SignalThresholds()
        self.model_name = model_name

    def _validate_probabilities(
        self,
        prob_down: float,
        prob_neutral: float,
        prob_up: float,
    ) -> tuple[float, float, float]:
        """Validate and sanitize probability values.

        Policy:
        - NaN, infinity, or outside [0, 1] → raise ValueError
        - Probabilities that don't sum to ~1.0 (within 1e-6) → raise ValueError

        Args:
            prob_down: P(target = -1)
            prob_neutral: P(target = 0)
            prob_up: P(target = +1)

        Returns:
            Validated (prob_down, prob_neutral, prob_up) tuple.

        Raises:
            ValueError: If any probability is invalid.
        """
        probs = np.array([prob_down, prob_neutral, prob_up], dtype=float)

        # Check for NaN/inf
        if not np.all(np.isfinite(probs)):
            raise ValueError(
                f"Probabilities contain NaN or infinity: down={prob_down}, "
                f"neutral={prob_neutral}, up={prob_up}"
            )

        # Check bounds
        if np.any(probs < 0.0) or np.any(probs > 1.0):
            raise ValueError(
                f"Probabilities outside [0, 1]: down={prob_down}, "
                f"neutral={prob_neutral}, up={prob_up}"
            )

        # Check sum ≈ 1.0 (allow tiny numerical drift)
        total = probs.sum()
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Probabilities sum to {total:.6f}, not 1.0: "
                f"down={prob_down}, neutral={prob_neutral}, up={prob_up}"
            )

        return float(prob_down), float(prob_neutral), float(prob_up)

    def _determine_action(
        self,
        y_pred: int,
        prob_down: float,
        prob_neutral: float,
        prob_up: float,
    ) -> tuple[SignalAction, float]:
        """Determine action and confidence from prediction and probabilities.

        Args:
            y_pred: Predicted class (-1, 0, +1).
            prob_down: P(target = -1).
            prob_neutral: P(target = 0).
            prob_up: P(target = +1).

        Returns:
            (SignalAction, confidence) tuple.
        """
        t = self.thresholds

        if y_pred == 1 and prob_up >= t.buy_threshold:
            # BUY: predicted bullish AND sufficient probability
            return SignalAction.BUY, prob_up
        elif y_pred == -1 and prob_down >= t.sell_threshold:
            # SELL: predicted bearish AND sufficient probability
            return SignalAction.SELL, prob_down
        else:
            # HOLD: insufficient conviction or neutral prediction
            # Confidence = max probability (reflects highest conviction)
            confidence = max(prob_down, prob_neutral, prob_up)
            return SignalAction.HOLD, confidence

    def _validate_inputs(
        self,
        symbol: str,
        timestamp: datetime,
        y_pred: int,
    ) -> None:
        """Validate required input fields.

        Args:
            symbol: Instrument symbol.
            timestamp: Signal timestamp.
            y_pred: Predicted class.

        Raises:
            ValueError: If any required field is missing or invalid.
        """
        if not symbol or not symbol.strip():
            raise ValueError("symbol is required and must be non-empty")
        if timestamp is None:
            raise ValueError("timestamp is required")
        if y_pred not in self.TARGET_CLASSES:
            raise ValueError(
                f"y_pred must be one of {self.TARGET_CLASSES}, got {y_pred}"
            )

    def convert(
        self,
        symbol: str,
        timestamp: datetime,
        y_pred: int,
        prob_down: float,
        prob_neutral: float,
        prob_up: float,
        realized_return: Optional[float] = None,
    ) -> TradingSignal:
        """Convert a single ML prediction to a TradingSignal.

        Args:
            symbol: Instrument symbol (e.g., "AAPL").
            timestamp: Signal generation time (after close of session t).
            y_pred: Predicted class (-1, 0, +1).
            prob_down: P(target = -1).
            prob_neutral: P(target = 0).
            prob_up: P(target = +1).
            realized_return: Optional realized return for the NEXT session.
                This is IGNORED for signal generation (causality). Provided
                only for potential downstream diagnostics.

        Returns:
            TradingSignal with action, confidence, and all three probabilities.

        Raises:
            ValueError: If inputs are invalid (missing, NaN, out of bounds, etc.).
        """
        # Validate required fields
        self._validate_inputs(symbol, timestamp, y_pred)

        # Validate and sanitize probabilities
        prob_down, prob_neutral, prob_up = self._validate_probabilities(
            prob_down, prob_neutral, prob_up
        )

        # Determine action and confidence
        action, confidence = self._determine_action(
            y_pred, prob_down, prob_neutral, prob_up
        )

        # Build the signal
        return TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            confidence=confidence,
            source=SignalSource.ML,
            model_name=self.model_name,
            probability_down=prob_down,
            probability_neutral=prob_neutral,
            probability_up=prob_up,
        )

    def convert_batch(
        self,
        predictions: pd.DataFrame,
    ) -> list[TradingSignal]:
        """Convert a DataFrame of ML predictions to TradingSignal list.

        Expects columns:
            - symbol (str)
            - timestamp (datetime-like)
            - y_pred (int: -1, 0, +1)
            - y_prob_-1, y_prob_0, y_prob_1 (float probabilities)
            - realized_return (optional, ignored for signal generation)

        Args:
            predictions: DataFrame with ML prediction columns.

        Returns:
            List of TradingSignal in the same order as input.

        Raises:
            ValueError: If required columns are missing or row validation fails.
        """
        required_cols = {"symbol", "timestamp", "y_pred", *self.PROB_COLUMNS}
        missing = required_cols - set(predictions.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        signals = []
        for _, row in predictions.iterrows():
            signal = self.convert(
                symbol=row["symbol"],
                timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                y_pred=int(row["y_pred"]),
                prob_down=float(row["y_prob_-1"]),
                prob_neutral=float(row["y_prob_0"]),
                prob_up=float(row["y_prob_1"]),
                realized_return=row.get("realized_return"),
            )
            signals.append(signal)

        return signals


# V1 default adapter instance
DEFAULT_ADAPTER = MLSignalAdapter()