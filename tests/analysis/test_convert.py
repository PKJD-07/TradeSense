"""Tests for the Candle -> DataFrame bridge and CSV persistence."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.convert import (
    candles_to_dataframe,
    collections_to_long_dataframe,
    pivot_close_prices,
    save_candles_csv,
    load_candles_csv,
    OHLCV_COLUMNS,
)
from tests.analysis.fixtures import (
    make_candle_df,
    make_candles,
    make_candle_collection,
    make_long_dataframe,
)


def test_candles_to_dataframe_shape_and_order():
    df = candles_to_dataframe(make_candles(n=50, seed=1))
    assert list(df.columns) == OHLCV_COLUMNS
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert df.index.is_monotonic_increasing
    assert len(df) == 50


def test_candles_to_dataframe_values_match_models():
    candles = make_candles(n=3, seed=2)
    df = candles_to_dataframe(candles)
    for i, candle in enumerate(sorted(candles, key=lambda c: c.timestamp)):
        assert df.iloc[i]["open"] == candle.open
        assert df.iloc[i]["high"] == candle.high
        assert df.iloc[i]["low"] == candle.low
        assert df.iloc[i]["close"] == candle.close
        assert df.iloc[i]["volume"] == candle.volume


def test_candles_to_dataframe_empty():
    df = candles_to_dataframe([])
    assert list(df.columns) == OHLCV_COLUMNS
    assert df.empty


def test_collections_to_long_dataframe_from_collection():
    coll = make_candle_collection(n=10, seed=1)
    long_df = collections_to_long_dataframe([coll])
    assert list(long_df.columns) == ["symbol", "timestamp", *OHLCV_COLUMNS]
    assert set(long_df["symbol"]) == {"AAPL"}
    assert len(long_df) == 10
    assert long_df["timestamp"].dt.tz is not None


def test_collections_to_long_dataframe_from_dict():
    coll_a = make_candle_collection(n=5, seed=1)
    coll_b = make_candle_collection(n=5, seed=2)
    long_df = collections_to_long_dataframe({"AAPL": coll_a, "MSFT": coll_b})
    assert set(long_df["symbol"]) == {"AAPL", "MSFT"}
    assert len(long_df) == 10


def test_pivot_close_prices():
    long_df = make_long_dataframe()
    wide = pivot_close_prices(long_df)
    assert list(wide.columns) == sorted(long_df["symbol"].unique())
    assert isinstance(wide.index, pd.DatetimeIndex)
    assert wide.index.tz is not None


def test_csv_round_trip_preserves_tz_and_values(tmp_path):
    df = make_candle_df(n=25, seed=1)
    path = tmp_path / "AAPL.csv"
    save_candles_csv(df, path)
    loaded = load_candles_csv(path)
    # Clear freq on both to avoid freq comparison issues
    df_check = df.copy()
    df_check.index.freq = None
    pd.testing.assert_frame_equal(loaded, df_check)
    assert loaded.index.tz == df.index.tz
    assert loaded.index.is_monotonic_increasing
