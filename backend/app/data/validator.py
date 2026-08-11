from backend.app.api.schemas.market import MarketCandle


def validate_historical_data(candles: list[MarketCandle]) -> None:
    if not candles:
        raise ValueError("Historical data cannot be empty")

    symbol = candles[0].symbol

    for candle in candles:
        if candle.symbol != symbol:
            raise ValueError("All candles must have the same symbol")

    for i in range(1, len(candles)):
        if candles[i].timestamp <= candles[i - 1].timestamp:
            raise ValueError("Candles must be in chronological order")

    if len(candles) < 5:
        raise ValueError("At least 5 candles are required")