from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class MarketCandle(BaseModel):
    symbol: str = Field(min_length=1)
    timestamp: datetime

    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)

    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_ohlc(self):
        if self.high < max(self.open, self.close):
            raise ValueError(
                "High price must be greater than or equal to open and close"
            )

        if self.low > min(self.open, self.close):
            raise ValueError(
                "Low price must be less than or equal to open and close"
            )

        if self.low > self.high:
            raise ValueError("Low price cannot be greater than high price")

        return self


class HistoricalMarketData(BaseModel):
    candles: list[MarketCandle] = Field(min_length=5)