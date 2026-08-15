"""Shared TradingSignal abstraction.

This module defines the common signal representation used by both the ML layer
and the backtester/trading layer. The ML layer must NOT import the backtester,
and the backtester must NOT care whether a signal came from ML, RSI, MACD,
or another strategy.

A TradingSignal is a signal representation, NOT an executed trade.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SignalAction(str, Enum):
    """Directional trading action."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalSource(str, Enum):
    """Origin of the signal."""

    ML = "ML"
    RULE_BASED = "RULE_BASED"
    MANUAL = "MANUAL"


class TradingSignal(BaseModel):
    """Universal trading signal representation.

    This is the common abstraction between ML predictions and the backtester.
    It contains no execution, position, cash, or risk-management fields.

    Attributes:
        timestamp: When the signal was generated (after close of session t).
        symbol: The instrument symbol (e.g., "AAPL").
        action: Directional action (BUY/SELL/HOLD).
        confidence: Confidence in the action, in [0, 1].
        source: Origin of the signal (ML, RULE_BASED, MANUAL).
        model_name: Specific model identifier when source is ML
            (e.g., "gradient_boosting").
        probability_down: P(target = -1 | features), in [0, 1].
        probability_neutral: P(target = 0 | features), in [0, 1].
        probability_up: P(target = +1 | features), in [0, 1].
    """

    timestamp: datetime
    symbol: str = Field(min_length=1)
    action: SignalAction
    confidence: float = Field(ge=0.0, le=1.0)
    source: SignalSource = SignalSource.ML
    model_name: Optional[str] = None
    probability_down: float = Field(ge=0.0, le=1.0)
    probability_neutral: float = Field(ge=0.0, le=1.0)
    probability_up: float = Field(ge=0.0, le=1.0)

    @field_validator("probability_down", "probability_neutral", "probability_up", mode="after")
    @classmethod
    def _validate_probability_range(cls, v: float) -> float:
        """Ensure probabilities are valid (already handled by Field bounds)."""
        return v

    def model_dump(self, **kwargs) -> dict:
        """Serialize with enum values as strings."""
        data = super().model_dump(**kwargs)
        data["action"] = self.action.value
        data["source"] = self.source.value
        return data