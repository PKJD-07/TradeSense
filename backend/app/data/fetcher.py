import yfinance as yf

from backend.app.api.schemas.market import MarketCandle


def fetch_historical_data(
    symbol: str,
    period: str = "1mo",
    interval: str = "1d",
) -> list[MarketCandle]:

    ticker = yf.Ticker(f"{symbol}.NS")

    data = ticker.history(
        period=period,
        interval=interval,
        auto_adjust=False,
    )

    if data.empty:
        raise ValueError(f"No market data found for NSE symbol: {symbol}")

    candles = []

    for timestamp, row in data.iterrows():
        candle = MarketCandle(
            symbol=symbol,
            timestamp=timestamp.to_pydatetime(),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
        )

        candles.append(candle)

    return candles