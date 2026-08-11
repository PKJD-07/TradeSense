"""Tests for volume features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.volume import (
    volume_change_1d,
    relative_volume_10d,
    volume_trend_5d,
)
from tests.features.fixtures import make_candle_df, make_zero_volume_df


class TestVolumeChange1d:
    """Tests for 1-session volume change."""

    def test_math_correctness(self):
        df = make_candle_df(n=30, seed=1)
        change = volume_change_1d(df)
        assert pd.isna(change.iloc[0])
        assert change.iloc[1] == pytest.approx(df["volume"].iloc[1] / df["volume"].iloc[0] - 1)

    def test_nan_at_start(self):
        df = make_candle_df(n=10, seed=2)
        change = volume_change_1d(df)
        assert pd.isna(change.iloc[0])
        assert change.iloc[1:].notna().all()

    def test_zero_volume_returns_nan(self):
        """If volume[t-1] is zero, should return NaN rather than inf."""
        df = make_zero_volume_df(n=10, seed=3, zero_at=[5])
        change = volume_change_1d(df)
        # Index 6 has volume[t-1] = 0, so it should be NaN
        assert pd.isna(change.iloc[6])


class TestRelativeVolume10d:
    """Tests for volume / SMA(volume, 10)."""

    def test_math_correctness(self):
        df = make_candle_df(n=30, seed=1)
        rel_vol = relative_volume_10d(df)
        sma = df["volume"].rolling(window=10, min_periods=10).mean()
        assert rel_vol.iloc[10] == pytest.approx(df["volume"].iloc[10] / sma.iloc[10])

    def test_nan_at_start(self):
        df = make_candle_df(n=20, seed=2)
        rel_vol = relative_volume_10d(df)
        # SMA(10) with min_periods=10: first valid at index 9
        assert rel_vol.iloc[:9].isna().all()

    def test_zero_average_volume_returns_nan(self):
        """If 10-day average volume is zero, should return NaN."""
        df = make_candle_df(n=15, seed=3)
        df.iloc[:10, df.columns.get_loc("volume")] = 0
        rel_vol = relative_volume_10d(df)
        # SMA at index 9 would be zero
        assert pd.isna(rel_vol.iloc[9])


class TestVolumeTrend5d:
    """Tests for SMA(volume, 5) / SMA(volume, 20) - 1."""

    def test_math_correctness(self):
        df = make_candle_df(n=40, seed=1)
        trend = volume_trend_5d(df)
        sma_5 = df["volume"].rolling(window=5, min_periods=5).mean()
        sma_20 = df["volume"].rolling(window=20, min_periods=20).mean()
        expected = sma_5 / sma_20 - 1
        for i in range(20, 25):
            assert trend.iloc[i] == pytest.approx(expected.iloc[i])

    def test_nan_at_start(self):
        df = make_candle_df(n=30, seed=2)
        trend = volume_trend_5d(df)
        # SMA(20) with min_periods=20: first valid at index 19
        assert trend.iloc[:19].isna().all()

    def test_zero_sma_20_returns_nan(self):
        df = make_candle_df(n=25, seed=3)
        df.iloc[:20, df.columns.get_loc("volume")] = 0
        trend = volume_trend_5d(df)
        # SMA(20) at index 19 would be zero
        assert pd.isna(trend.iloc[19])
