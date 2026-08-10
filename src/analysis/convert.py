"""Bridging Candle/CandleCollection objects to pandas DataFrames.

The data layer (``src.data``) deliberately has no pandas dependency. This
module is the single bridge that converts the model objects into pandas
DataFrames so the analysis layer can operate on them.

All output frames use the candle timestamp (timezone-aware UTC) as the index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.models import Candle, CandleCollection

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def candles_to_dataframe(candles: Iterable[Candle]) -> pd.DataFrame:
    """Convert candles to a wide per-symbol DataFrame.

    The returned frame is indexed by timestamp (timezone-aware UTC, ascending)
    with columns: open, high, low, close, volume.
    """
    candles = list(candles)
    if not candles:
        return pd.DataFrame(columns=OHLCV_COLUMNS).rename_axis("timestamp")

    records = [{name: getattr(c, name) for name in OHLCV_COLUMNS} for c in candles]
    index = pd.DatetimeIndex([c.timestamp for c in candles], name="timestamp")
    df = pd.DataFrame(records, index=index)
    df = df.sort_index()
    return df[OHLCV_COLUMNS]


def collections_to_long_dataframe(
    collections: dict[str, Iterable[Candle]] | Iterable[CandleCollection],
) -> pd.DataFrame:
    """Convert one or more symbol collections into a long-form DataFrame.

    The returned frame has columns: symbol, timestamp, open, high, low, close,
    volume. Useful for multi-symbol quality reports and correlation work.

    Accepts either a mapping {symbol: candles} or an iterable of
    CandleCollection objects.
    """
    items: list[tuple[str, Candle]] = []
    if isinstance(collections, dict):
        for symbol, candles in collections.items():
            for candle in candles:
                items.append((symbol, candle))
    else:
        for collection in collections:
            for candle in collection:
                items.append((candle.symbol, candle))

    if not items:
        return pd.DataFrame(columns=["symbol", "timestamp", *OHLCV_COLUMNS])

    records = []
    for symbol, c in items:
        records.append(
            {
                "symbol": symbol,
                "timestamp": c.timestamp,
                **{name: getattr(c, name) for name in OHLCV_COLUMNS},
            }
        )
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def pivot_close_prices(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-form data into a wide close-price frame (symbols as columns)."""
    return long_df.pivot(index="timestamp", columns="symbol", values="close").sort_index()


def save_candles_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a single-symbol candle DataFrame to CSV (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def load_candles_csv(path: str | Path) -> pd.DataFrame:
    """Load a candle DataFrame saved by :func:`save_candles_csv`.

    The timestamp index is restored as timezone-aware UTC regardless of how
    pandas wrote/read it.
    """
    path = Path(path)
    df = pd.read_csv(path, index_col=0)
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    df.index.freq = None  # clear freq to avoid comparison issues
    return df.sort_index()
