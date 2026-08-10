"""Tests for price-based features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.price import (
    return_1d,
    return_5d,
    return_10d,
    return_20d,
    log_return_1d,
    intraday_return,
)
from tests.features.fixtures import make_candle_df


class TestReturn1d:
    """Tests for 1-session simple return."""

    def test_math_correctness(self):
        df = make_candle_df(n=30, seed=1)
        r = return_1d(df)
        assert pd.isna(r.iloc[0])
        assert r.iloc[1] == pytest.approx(df["close"].iloc[1] / df["close"].iloc[0] - 1)
        assert r.iloc[10] == pytest.approx(df["close"].iloc[10] / df["close"].iloc[9] - 1)

    def test_nan_at_start(self):
        df = make_candle_df(n=10, seed=2)
        r = return_1d(df)
        assert pd.isna(r.iloc[0])
        assert r.iloc[1:].notna().all()

    def test_length_preserved(self):
        df = make_candle_df(n=50, seed=3)
        r = return_1d(df)
        assert len(r) == len(df)


class TestReturn5d:
    """Tests for 5-session simple return."""

    def test_math_correctness(self):
        df = make_candle_df(n=30, seed=1)
        r = return_5d(df)
        assert r.iloc[5] == pytest.approx(df["close"].iloc[5] / df["close"].iloc[0] - 1)
        assert r.iloc[10] == pytest.approx(df["close"].iloc[10] / df["close"].iloc[5] - 1)

    def test_nan_at_start(self):
        df = make_candle_df(n=20, seed=2)
        r = return_5d(df)
        assert r.iloc[:5].isna().all()
        assert r.iloc[5:].notna().all()


class TestReturn10d:
    """Tests for 10-session simple return."""

    def test_math_correctness(self):
        df = make_candle_df(n=30, seed=1)
        r = return_10d(df)
        assert r.iloc[10] == pytest.approx(df["close"].iloc[10] / df["close"].iloc[0] - 1)

    def test_nan_at_start(self):
        df = make_candle_df(n=20, seed=2)
        r = return_10d(df)
        assert r.iloc[:10].isna().all()


class TestReturn20d:
    """Tests for 20-session simple return."""

    def test_math_correctness(self):
        df = make_candle_df(n=50, seed=1)
        r = return_20d(df)
        assert r.iloc[20] == pytest.approx(df["close"].iloc[20] / df["close"].iloc[0] - 1)

    def test_nan_at_start(self):
        df = make_candle_df(n=30, seed=2)
        r = return_20d(df)
        assert r.iloc[:20].isna().all()


class TestLogReturn1d:
    """Tests for 1-session log return."""

    def test_math_correctness(self):
        df = make_candle_df(n=30, seed=1)
        r = log_return_1d(df)
        assert pd.isna(r.iloc[0])
        assert r.iloc[1] == pytest.approx(np.log(df["close"].iloc[1] / df["close"].iloc[0]))

    def test_relation_to_simple_return(self):
        df = make_candle_df(n=30, seed=2)
        simple = return_1d(df)
        log_ret = log_return_1d(df)
        # log(1 + r_simple) = r_log
        for i in range(1, len(df)):
            assert log_ret.iloc[i] == pytest.approx(np.log(1 + simple.iloc[i]))


class TestIntradayReturn:
    """Tests for intraday (open-to-close) return."""

    def test_math_correctness(self):
        df = make_candle_df(n=30, seed=1)
        r = intraday_return(df)
        for i in range(len(df)):
            assert r.iloc[i] == pytest.approx(df["close"].iloc[i] / df["open"].iloc[i] - 1)

    def test_no_nan_with_valid_data(self):
        df = make_candle_df(n=30, seed=2)
        r = intraday_return(df)
        assert r.notna().all()

    def test_handles_zero_open(self):
        """If open is zero, should return NaN rather than inf."""
        df = make_candle_df(n=10, seed=3)
        df.iloc[5, df.columns.get_loc("open")] = 0.0
        r = intraday_return(df)
        assert pd.isna(r.iloc[5])
        # Other rows should be fine
        assert r.iloc[:5].notna().all()
        assert r.iloc[6:].notna().all()
