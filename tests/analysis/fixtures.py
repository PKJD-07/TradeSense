"""Deterministic synthetic OHLCV fixtures for analysis tests.

All price series are generated with a fixed random seed (no live API), so every
run is reproducible. Timestamps follow the model convention: timezone-aware UTC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.models import Candle, CandleCollection


def make_close_series(
    n: int = 250,
    seed: int = 42,
    start: str = "2021-01-04",
    drift: float = 0.0005,
    vol: float = 0.01,
) -> pd.Series:
    """Geometric random walk of adjusted closes (UTC-aware DatetimeIndex)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n).tz_localize("UTC")
    log_returns = rng.normal(loc=drift, scale=vol, size=n)
    log_returns[0] = 0.0
    close = pd.Series(100.0 * np.exp(np.cumsum(log_returns)), index=dates, name="close")
    return close


def make_candle_df(
    n: int = 250,
    seed: int = 42,
    start: str = "2021-01-04",
    drift: float = 0.0005,
    vol: float = 0.01,
) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with valid OHLC relationships and UTC index."""
    close = make_close_series(n=n, seed=seed, start=start, drift=drift, vol=vol)
    rng = np.random.default_rng(seed + 1)

    open_ = close.shift(1) * (1.0 + rng.normal(0.0, 0.003, size=n))
    open_ = open_.fillna(close * (1.0 + rng.normal(0.0, 0.003)))

    spread = rng.uniform(0.001, 0.01, size=n)
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.integers(1_000_000, 5_000_000, size=n)

    df = pd.DataFrame(
        {
            "open": open_.to_numpy(),
            "high": high,
            "low": low,
            "close": close.to_numpy(),
            "volume": volume,
        },
        index=close.index,
    )
    df.index.name = "timestamp"
    return df


def make_candles(n: int = 250, seed: int = 42) -> list[Candle]:
    """Synthetic Candle objects consistent with make_candle_df (same seed)."""
    df = make_candle_df(n=n, seed=seed)
    candles = []
    for ts, row in df.iterrows():
        candles.append(
            Candle(
                symbol="AAPL",
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            )
        )
    return candles


def make_candle_collection(n: int = 250, seed: int = 42) -> CandleCollection:
    """Synthetic CandleCollection for a single symbol."""
    return CandleCollection(symbol="AAPL", candles=make_candles(n=n, seed=seed))


def make_long_dataframe(
    symbols: tuple[str, ...] = ("AAPL", "MSFT", "JPM", "XOM", "SPY"),
    n: int = 250,
    seed: int = 7,
) -> pd.DataFrame:
    """Long-form multi-symbol DataFrame on a common set of sessions.

    Columns: symbol, timestamp, open, high, low, close, volume.
    """
    frames = []
    for i, symbol in enumerate(symbols):
        df = make_candle_df(n=n, seed=seed + i, start="2021-01-04").reset_index()
        df.insert(0, "symbol", symbol)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
