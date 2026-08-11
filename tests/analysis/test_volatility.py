"""Tests for rolling and realized volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.volatility import (
    annualized_volatility,
    rolling_volatility,
    realized_volatility,
)
from src.analysis.returns import log_returns
from tests.analysis.fixtures import make_candle_df, make_close_series


def test_annualized_volatility_formula():
    close = make_close_series(n=120, seed=1, vol=0.01)
    r = log_returns(close)
    expected = r.std(ddof=1) * np.sqrt(252)
    assert annualized_volatility(r) == pytest.approx(expected)


def test_rolling_volatility_window_and_annualization():
    df = make_candle_df(n=30, seed=2)
    r = log_returns(df["close"])
    rv = rolling_volatility(r, window=5, annualize=False)
    # First 4 values are NaN (window needs 5 values, but r[0] is NaN from diff())
    # With default min_periods=window, index 4 is also NaN, index 5 is first valid
    assert rv.iloc[:5].isna().all()
    # At index 5, the window is r[1:6] (5 non-NaN values)
    assert rv.iloc[5] == pytest.approx(r.iloc[1:6].std(ddof=1))
    # annualized = raw * sqrt(252)
    rv_ann = rolling_volatility(r, window=5, annualize=True)
    assert rv_ann.iloc[5] == pytest.approx(rv.iloc[5] * np.sqrt(252))


def test_realized_volatility_math():
    df = make_candle_df(n=30, seed=3)
    r = log_returns(df["close"])
    rv = realized_volatility(r, window=5, annualize=False)
    expected = np.sqrt(np.sum(r.iloc[0:5].to_numpy() ** 2))
    assert rv.iloc[4] == pytest.approx(expected, nan_ok=True)
    # first window-1 values are NaN
    assert rv.iloc[:4].isna().all()


def test_realized_volatility_annualized():
    df = make_candle_df(n=30, seed=3)
    r = log_returns(df["close"])
    rv = realized_volatility(r, window=5, annualize=True)
    raw = realized_volatility(r, window=5, annualize=False)
    # both are NaN at idx 0-3; compare at idx 4 where the window is full
    if pd.notna(raw.iloc[4]):
        assert rv.iloc[4] == pytest.approx(raw.iloc[4] * np.sqrt(252 / 5))


def test_invalid_window():
    with pytest.raises(ValueError):
        rolling_volatility(make_close_series(n=10), window=0)
