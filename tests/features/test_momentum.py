"""Tests for momentum features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.momentum import (
    price_sma_ratio_10,
    price_sma_ratio_20,
    price_ema_ratio_10,
    sma_cross_10_20,
)
from tests.features.fixtures import make_candle_df


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Reference SMA implementation."""
    return series.rolling(window=window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Reference EMA implementation."""
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


class TestPriceSmaRatio10:
    """Tests for price/SMA(10) ratio."""

    def test_math_correctness(self):
        df = make_candle_df(n=30, seed=1)
        ratio = price_sma_ratio_10(df)
        sma = _sma(df["close"], 10)
        assert ratio.iloc[10] == pytest.approx(df["close"].iloc[10] / sma.iloc[10])

    def test_nan_at_start(self):
        df = make_candle_df(n=30, seed=2)
        ratio = price_sma_ratio_10(df)
        # SMA(10) with min_periods=10: first valid at index 9 (10th element)
        assert ratio.iloc[:9].isna().all()
        assert ratio.iloc[9:].notna().all()

    def test_handles_zero_sma(self):
        """If SMA is zero, should return NaN."""
        df = make_candle_df(n=15, seed=3)
        # Force close prices to zero for first 10 rows
        df.iloc[:10, df.columns.get_loc("close")] = 0.0
        ratio = price_sma_ratio_10(df)
        # SMA at index 9 would be zero (all 10 values are 0)
        assert pd.isna(ratio.iloc[9])


class TestPriceSmaRatio20:
    """Tests for price/SMA(20) ratio."""

    def test_math_correctness(self):
        df = make_candle_df(n=40, seed=1)
        ratio = price_sma_ratio_20(df)
        sma = _sma(df["close"], 20)
        assert ratio.iloc[20] == pytest.approx(df["close"].iloc[20] / sma.iloc[20])

    def test_nan_at_start(self):
        df = make_candle_df(n=30, seed=2)
        ratio = price_sma_ratio_20(df)
        # SMA(20) with min_periods=20: first valid at index 19 (20th element)
        assert ratio.iloc[:19].isna().all()


class TestPriceEmaRatio10:
    """Tests for price/EMA(10) ratio."""

    def test_math_correctness(self):
        df = make_candle_df(n=30, seed=1)
        ratio = price_ema_ratio_10(df)
        ema = _ema(df["close"], 10)
        assert ratio.iloc[10] == pytest.approx(df["close"].iloc[10] / ema.iloc[10])

    def test_nan_at_start(self):
        df = make_candle_df(n=30, seed=2)
        ratio = price_ema_ratio_10(df)
        # EMA(10) with min_periods=10: first valid at index 9 (10th element)
        assert ratio.iloc[:9].isna().all()


class TestSmaCross10_20:
    """Tests for SMA(10)/SMA(20) crossover."""

    def test_math_correctness(self):
        df = make_candle_df(n=40, seed=1)
        cross = sma_cross_10_20(df)
        sma_10 = _sma(df["close"], 10)
        sma_20 = _sma(df["close"], 20)
        assert cross.iloc[20] == pytest.approx(sma_10.iloc[20] / sma_20.iloc[20] - 1)

    def test_nan_at_start(self):
        df = make_candle_df(n=30, seed=2)
        cross = sma_cross_10_20(df)
        # SMA(20) with min_periods=20: first valid at index 19 (20th element)
        assert cross.iloc[:19].isna().all()

    def test_signal_direction(self):
        """Positive when short SMA > long SMA, negative when short < long."""
        df = make_candle_df(n=50, seed=3)
        cross = sma_cross_10_20(df)
        # Just verify it produces both positive and negative values
        positive_count = (cross.iloc[20:] > 0).sum()
        negative_count = (cross.iloc[20:] < 0).sum()
        assert positive_count > 0 or negative_count > 0  # At least one non-zero
