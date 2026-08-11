from backend.app.api.schemas.market import MarketCandle
from backend.app.analysis.indicators import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_adx,
)


def analyze_market(candles: list[MarketCandle]) -> dict:
    if len(candles) < 50:
        raise ValueError("Not enough market data for analysis")

    symbol = candles[-1].symbol

    closing_prices = [
        candle.close
        for candle in candles
    ]

    latest_price = closing_prices[-1]

    # Moving averages
    sma_20 = calculate_sma(
        closing_prices,
        20,
    )

    sma_50 = calculate_sma(
        closing_prices,
        50,
    )

    ema_20 = calculate_ema(
        closing_prices,
        20,
    )

    # RSI
    rsi_14 = calculate_rsi(
        closing_prices,
        14,
    )

    # MACD
    macd, macd_signal, macd_histogram = calculate_macd(
        closing_prices
    )

    # ADX
    adx_14 = calculate_adx(
        candles,
        14,
    )

    return {
        "symbol": symbol,
        "latest_price": latest_price,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "ema_20": ema_20,
        "rsi_14": rsi_14,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_histogram": macd_histogram,
        "adx_14": adx_14,
    }