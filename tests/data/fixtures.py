"""
Test fixtures for the data layer tests.

Provides sample valid and invalid candles for testing.
"""

from datetime import datetime, timezone

from src.data.models import Candle


def make_valid_candle(
    symbol: str = "AAPL",
    timestamp: datetime | None = None,
    open_price: float = 210.15,
    high: float = 212.30,
    low: float = 209.80,
    close: float = 211.75,
    volume: int = 1523400,
) -> Candle:
    """
    Create a valid OHLCV candle for testing.

    Args:
        symbol: Ticker symbol
        timestamp: Candle timestamp (defaults to 2026-08-10 00:00:00 UTC)
        open_price: Opening price
        high: High price
        low: Low price
        close: Close price
        volume: Trading volume

    Returns:
        A valid Candle instance
    """
    if timestamp is None:
        timestamp = datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc)

    return Candle(
        symbol=symbol,
        timestamp=timestamp,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def make_valid_candles(count: int = 3) -> list[Candle]:
    """
    Create a list of valid candles for testing.

    Args:
        count: Number of candles to create

    Returns:
        List of valid Candle instances with consecutive days
    """
    candles = []
    base_date = datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc)

    for i in range(count):
        # Each candle is one day apart
        ts = datetime(
            base_date.year,
            base_date.month,
            base_date.day - i,
            0,
            0,
            0,
            tzinfo=timezone.utc,
        )
        candles.append(
            Candle(
                symbol="AAPL",
                timestamp=ts,
                open=210.0 + i,
                high=212.0 + i,
                low=209.0 + i,
                close=211.0 + i,
                volume=1000000 + i * 10000,
            )
        )

    # Return in chronological order
    return list(reversed(candles))


def make_candle_with_invalid_ohlc() -> dict:
    """
    Create a candle dict with invalid OHLC relationship (high < low).

    Returns:
        Dict with invalid OHLC values
    """
    return {
        "symbol": "AAPL",
        "timestamp": datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
        "open": 210.0,
        "high": 209.0,  # Invalid: high < low
        "low": 211.0,
        "close": 210.5,
        "volume": 1000000,
    }


def make_candle_with_negative_price() -> dict:
    """
    Create a candle dict with negative price.

    Returns:
        Dict with negative price value
    """
    return {
        "symbol": "AAPL",
        "timestamp": datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
        "open": -210.0,  # Invalid: negative price
        "high": 212.0,
        "low": 209.0,
        "close": 211.0,
        "volume": 1000000,
    }


def make_candle_with_negative_volume() -> dict:
    """
    Create a candle dict with negative volume.

    Returns:
        Dict with negative volume value
    """
    return {
        "symbol": "AAPL",
        "timestamp": datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
        "open": 210.0,
        "high": 212.0,
        "low": 209.0,
        "close": 211.0,
        "volume": -1000000,  # Invalid: negative volume
    }


def make_candle_with_missing_symbol() -> dict:
    """
    Create a candle dict with empty symbol.

    Returns:
        Dict with empty symbol
    """
    return {
        "symbol": "",  # Invalid: empty symbol
        "timestamp": datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
        "open": 210.0,
        "high": 212.0,
        "low": 209.0,
        "close": 211.0,
        "volume": 1000000,
    }


def make_candle_with_invalid_timestamp() -> dict:
    """
    Create a candle dict with invalid timestamp.

    Returns:
        Dict with invalid timestamp
    """
    return {
        "symbol": "AAPL",
        "timestamp": "not-a-timestamp",  # Invalid: not a datetime
        "open": 210.0,
        "high": 212.0,
        "low": 209.0,
        "close": 211.0,
        "volume": 1000000,
    }


def make_duplicate_timestamp_candles() -> list[Candle]:
    """
    Create a list of candles with duplicate timestamps.

    Returns:
        List with one duplicate timestamp
    """
    ts = datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc)
    return [
        Candle(
            symbol="AAPL",
            timestamp=ts,
            open=210.0,
            high=212.0,
            low=209.0,
            close=211.0,
            volume=1000000,
        ),
        Candle(
            symbol="AAPL",
            timestamp=ts,  # Duplicate!
            open=211.0,
            high=213.0,
            low=210.0,
            close=212.0,
            volume=1100000,
        ),
        Candle(
            symbol="AAPL",
            timestamp=datetime(2026, 8, 9, 0, 0, 0, tzinfo=timezone.utc),
            open=209.0,
            high=211.0,
            low=208.0,
            close=210.0,
            volume=900000,
        ),
    ]


def make_unsorted_candles() -> list[Candle]:
    """
    Create a list of unsorted candles.

    Returns:
        List with timestamps out of order
    """
    return [
        Candle(
            symbol="AAPL",
            timestamp=datetime(2026, 8, 12, 0, 0, 0, tzinfo=timezone.utc),
            open=210.0,
            high=212.0,
            low=209.0,
            close=211.0,
            volume=1000000,
        ),
        Candle(
            symbol="AAPL",
            timestamp=datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),  # Out of order
            open=209.0,
            high=211.0,
            low=208.0,
            close=210.0,
            volume=900000,
        ),
        Candle(
            symbol="AAPL",
            timestamp=datetime(2026, 8, 11, 0, 0, 0, tzinfo=timezone.utc),
            open=211.0,
            high=213.0,
            low=210.0,
            close=212.0,
            volume=1100000,
        ),
    ]


def make_mock_provider_response() -> list[dict]:
    """
    Create a mock provider response.

    Returns:
        List of candle dictionaries as a provider would return
    """
    return [
        {
            "symbol": "AAPL",
            "timestamp": datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
            "open": 210.15,
            "high": 212.30,
            "low": 209.80,
            "close": 211.75,
            "volume": 1523400,
        },
        {
            "symbol": "AAPL",
            "timestamp": datetime(2026, 8, 9, 0, 0, 0, tzinfo=timezone.utc),
            "open": 208.50,
            "high": 210.20,
            "low": 207.90,
            "close": 209.80,
            "volume": 1452000,
        },
        {
            "symbol": "AAPL",
            "timestamp": datetime(2026, 8, 8, 0, 0, 0, tzinfo=timezone.utc),
            "open": 207.00,
            "high": 209.50,
            "low": 206.80,
            "close": 208.40,
            "volume": 1389000,
        },
    ]


def make_malformed_provider_response() -> list[dict]:
    """
    Create a malformed provider response.

    Returns:
        List of candle dictionaries with missing/invalid fields
    """
    return [
        {
            "symbol": "AAPL",
            "timestamp": datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
            "open": 210.15,
            "high": 212.30,
            "low": 209.80,
            "close": 211.75,
            "volume": 1523400,
        },
        {
            # Missing 'close' field
            "symbol": "AAPL",
            "timestamp": datetime(2026, 8, 9, 0, 0, 0, tzinfo=timezone.utc),
            "open": 208.50,
            "high": 210.20,
            "low": 207.90,
            "volume": 1452000,
        },
    ]
