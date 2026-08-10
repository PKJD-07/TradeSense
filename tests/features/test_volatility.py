"""Tests for volatility features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.volatility import (
    volatility_10d,
    volatility_20d,
    high_low_range,
    atr_ratio_14,
    volatility_ratio,
)
from tests.features.fixtures import make_candle_df


class TestVolatility10d:
    """Tests for 10-session rolling volatility."""

    def test_math_correctness(self):
        df = make_candle_df(n=40, seed=1)
        vol = volatility_10d(df)
        # Volatility at index 10 uses log returns from indices 1-10 (10 returns)
        # log_return[0] is NaN, so volatility_10d[10] uses log_return[1:11]
        log_ret = np.log(df["close"]).diff()
        expected = log_ret.iloc[1:11].std(ddof=1) * np.sqrt(252)
        assert vol.iloc[10] == pytest.approx(expected)

    def test_nan_at_start(self):
        """Requires 10 log returns = 10 valid log returns (indices 1-10 for vol at index 10)."""
        df = make_candle_df(n=20, seed=2)
        vol = volatility_10d(df)
        # volatility_10d with min_periods=10: needs 10 log returns
        # log_return[0] is NaN, log_return[1:11] are first 10 valid
        # So volatility_10d[10] is first valid (uses log_return[1:11])
        assert vol.iloc[:10].isna().all()
        assert vol.iloc[10:].notna().all()


class TestVolatility20d:
    """Tests for 20-session rolling volatility."""

    def test_nan_at_start(self):
        """Requires 20 log returns = 20 valid log returns."""
        df = make_candle_df(n=30, seed=1)
        vol = volatility_20d(df)
        # volatility_20d with min_periods=20: needs 20 log returns
        # log_return[0] is NaN, log_return[1:21] are first 20 valid
        # So volatility_20d[20] is first valid
        assert vol.iloc[:20].isna().all()


class TestHighLowRange:
    """Tests for high-low range feature."""

    def test_math_correctness(self):
        df = make_candle_df(n=30, seed=1)
        r = high_low_range(df)
        for i in range(len(df)):
            expected = (df["high"].iloc[i] - df["low"].iloc[i]) / df["close"].iloc[i]
            assert r.iloc[i] == pytest.approx(expected)

    def test_no_nan_with_valid_data(self):
        df = make_candle_df(n=30, seed=2)
        r = high_low_range(df)
        assert r.notna().all()

    def test_handles_zero_close(self):
        df = make_candle_df(n=10, seed=3)
        df.iloc[5, df.columns.get_loc("close")] = 0.0
        r = high_low_range(df)
        assert pd.isna(r.iloc[5])


class TestAtrRatio14:
    """Tests for ATR(14) / close ratio."""

    def test_nan_at_start(self):
        """ATR(14) requires 14 true range values."""
        df = make_candle_df(n=20, seed=1)
        atr = atr_ratio_14(df)
        # ATR(14) with min_periods=14: first valid at index 13 (14th element)
        assert atr.iloc[:13].isna().all()

    def test_math_correctness(self):
        df = make_candle_df(n=30, seed=2)
        atr = atr_ratio_14(df)

        # Manual TR calculation
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        expected_atr = tr.rolling(window=14, min_periods=14).mean()
        expected_ratio = expected_atr / close

        for i in range(14, 20):
            assert atr.iloc[i] == pytest.approx(expected_ratio.iloc[i])


class TestVolatilityRatio:
    """Tests for volatility_10d / volatility_20d ratio."""

    def test_nan_determined_by_volatility_20d(self):
        """Ratio is NaN wherever volatility_20d is NaN."""
        df = make_candle_df(n=30, seed=1)
        ratio = volatility_ratio(df)
        # volatility_20d needs 20 log returns, first valid at index 20
        assert ratio.iloc[:20].isna().all()

    def test_ratio_math(self):
        df = make_candle_df(n=40, seed=2)
        ratio = volatility_ratio(df)
        vol_10 = volatility_10d(df)
        vol_20 = volatility_20d(df)
        # First valid at index 20
        assert ratio.iloc[20] == pytest.approx(vol_10.iloc[20] / vol_20.iloc[20])
