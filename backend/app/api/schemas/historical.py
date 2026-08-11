from pydantic import BaseModel, Field, model_validator

from backend.app.api.schemas.market import MarketCandle


class HistoricalDataset(BaseModel):
    symbol: str = Field(min_length=1)
    timeframe: str = "1D"
    candles: list[MarketCandle] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dataset(self):
        # Every candle must belong to the same symbol
        for candle in self.candles:
            if candle.symbol != self.symbol:
                raise ValueError(
                    f"Candle symbol {candle.symbol} does not match "
                    f"dataset symbol {self.symbol}"
                )

        # Candles must be in chronological order
        timestamps = [candle.timestamp for candle in self.candles]

        if timestamps != sorted(timestamps):
            raise ValueError(
                "Candles must be sorted in chronological order"
            )

        # No duplicate timestamps
        if len(timestamps) != len(set(timestamps)):
            raise ValueError(
                "Duplicate candle timestamps are not allowed"
            )

        return self