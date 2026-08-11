"""Tests for return computations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.returns import (
    simple_returns,
    log_returns,
    n_period_returns,
    open_to_close_returns,
)
from tests.analysis.fixtures import make_close_series, make_candle_df


def test_simple_returns_matches_pct_change():
    close = make_close_series(n=50, seed=1)
    r = simple_returns(close)
    pd.testing.assert_series_equal(r, close.pct_change())
    assert pd.isna(r.iloc[0])
    assert not r.iloc[1:].isna().any()


def test_log_returns_match_manual():
    close = make_close_series(n=50, seed=1)
    r = log_returns(close)
    assert pd.isna(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(close.iloc[1] / close.iloc[0]))


def test_n_period_returns_simple():
    close = make_close_series(n=50, seed=1)
    r5 = n_period_returns(close, n=5)
    assert r5.iloc[5] == pytest.approx(close.iloc[5] / close.iloc[0] - 1.0)
    assert pd.isna(r5.iloc[4])


def test_n_period_returns_log():
    close = make_close_series(n=50, seed=1)
    r5 = n_period_returns(close, n=5, log=True)
    assert r5.iloc[5] == pytest.approx(np.log(close.iloc[5] / close.iloc[0]))


def test_n_period_returns_invalid_n():
    with pytest.raises(ValueError):
        n_period_returns(make_close_series(n=10), n=0)


def test_open_to_close_returns():
    df = make_candle_df(n=20, seed=1)
    oc = open_to_close_returns(df)
    assert oc.iloc[3] == pytest.approx(df.iloc[3]["close"] / df.iloc[3]["open"] - 1.0)
    assert len(oc) == len(df)
