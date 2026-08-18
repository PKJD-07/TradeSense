from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradingSignal(BaseModel):
    timestamp: datetime
    symbol: str = Field(min_length=1)

    action: SignalAction

    confidence: float = Field(ge=0.0, le=1.0)

    source: str = Field(min_length=1)
    model: str = Field(min_length=1)

    probability_down: float = Field(ge=0.0, le=1.0)
    probability_neutral: float = Field(ge=0.0, le=1.0)
    probability_up: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_probabilities(self):
        probability_sum = (
            self.probability_down
            + self.probability_neutral
            + self.probability_up
        )

        if abs(probability_sum - 1.0) > 1e-6:
            raise ValueError(
                "Signal probabilities must sum to 1.0"
            )

        expected_confidence = max(
            self.probability_down,
            self.probability_neutral,
            self.probability_up,
        )

        if abs(self.confidence - expected_confidence) > 1e-6:
            raise ValueError(
                "Confidence must equal the highest signal probability"
            )

        return self