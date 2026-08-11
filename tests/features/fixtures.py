"""
Deterministic synthetic OHLCV fixtures for feature-engineering tests.

All price series are generated with fixed random seeds (no live API).
Timestamps are timezone-aware UTC. Supports multi-symbol datasets with
configurable missing timestamps and edge cases.
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


def make_long_ohlcv_df(
    symbols: tuple[str, ...] = ("AAPL", "MSFT", "JPM", "XOM", "SPY"),
    n: int = 100,
    seed: int = 7,
    start: str = "2021-01-04",
) -> pd.DataFrame:
    """Long-form multi-symbol OHLCV DataFrame.

    Columns: symbol, timestamp, open, high, low, close, volume
    """
    frames = []
    for i, symbol in enumerate(symbols):
        df = make_candle_df(n=n, seed=seed + i, start=start).reset_index()
        df.insert(0, "symbol", symbol)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_long_ohlcv_with_gaps(
    symbols: tuple[str, ...] = ("AAPL", "SPY"),
    n: int = 50,
    seed: int = 11,
    drop_timestamps: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """Multi-symbol OHLCV with intentional timestamp gaps.

    Args:
        symbols: Symbols to include
        n: Base number of sessions per symbol
        seed: Random seed
        drop_timestamps: Dict mapping symbol -> list of indices to drop

    Returns:
        Long-form DataFrame with gaps in specified symbols
    """
    df = make_long_ohlcv_df(symbols=symbols, n=n, seed=seed)

    if drop_timestamps:
        for symbol, indices in drop_timestamps.items():
            symbol_mask = df["symbol"] == symbol
            symbol_df = df[symbol_mask].reset_index(drop=True)
            timestamps_to_drop = symbol_df.loc[indices, "timestamp"].tolist()
            df = df[~((df["symbol"] == symbol) & (df["timestamp"].isin(timestamps_to_drop)))]

    return df.reset_index(drop=True)


def make_candles(n: int = 250, seed: int = 42, symbol: str = "AAPL") -> list[Candle]:
    """Synthetic Candle objects for a single symbol."""
    df = make_candle_df(n=n, seed=seed)
    candles = []
    for ts, row in df.iterrows():
        candles.append(
            Candle(
                symbol=symbol,
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            )
        )
    return candles


def make_zero_volume_df(n: int = 30, seed: int = 99, zero_at: list[int] | None = None) -> pd.DataFrame:
    """OHLCV DataFrame with zero volume at specified indices.

    Args:
        n: Number of sessions
        seed: Random seed
        zero_at: Indices where volume should be zero
    """
    df = make_candle_df(n=n, seed=seed)
    if zero_at:
        for idx in zero_at:
            if 0 <= idx < n:
                df.iloc[idx, df.columns.get_loc("volume")] = 0
    return df
