"""Tests for descriptive statistics, autocorrelation, and cross-asset correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.returns import log_returns, simple_returns
from src.analysis.statistics import (
    describe_returns,
    autocorrelation,
    cross_asset_correlation,
)
from tests.analysis.fixtures import make_candle_df, make_close_series


def manual_acf(series: pd.Series, lag: int) -> float:
    """Pearson correlation between the series and its lag-shifted self."""
    x = series.iloc[lag:].to_numpy()
    y = series.iloc[:-lag].to_numpy()
    return float(np.corrcoef(x, y)[0, 1])


def test_describe_returns_fields():
    df = make_candle_df(n=250, seed=1)
    r = log_returns(df["close"]).dropna()
    stats = describe_returns(r)
    assert stats["count"] == len(r)
    assert stats["mean"] == pytest.approx(r.mean())
    assert stats["std"] == pytest.approx(r.std(ddof=1))
    assert stats["annualized_vol"] == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    assert stats["annualized_mean"] == pytest.approx(r.mean() * 252)
    assert np.isfinite(stats["skew"])
    assert np.isfinite(stats["kurtosis"])


def test_describe_returns_empty():
    stats = describe_returns(pd.Series([], dtype=float))
    assert len(stats) == 0


def test_autocorrelation_matches_manual():
    df = make_candle_df(n=150, seed=2)
    r = simple_returns(df["close"]).dropna()
    acf = autocorrelation(r, lags=5)
    assert list(acf.index) == [1, 2, 3, 4, 5]
    for lag in range(1, 6):
        assert acf[lag] == pytest.approx(manual_acf(r, lag))


def test_autocorrelation_invalid_lags():
    with pytest.raises(ValueError):
        autocorrelation(make_close_series(n=20), lags=0)


def test_cross_asset_correlation_perfect_and_independent():
    close_a = make_close_series(n=200, seed=1)
    close_b = close_a.copy()  # identical returns -> perfect correlation
    close_c = make_close_series(n=200, seed=99)  # independent random walk

    wide = pd.concat(
        [close_a.rename("A"), close_b.rename("B"), close_c.rename("C")], axis=1
    )
    corr = cross_asset_correlation(wide)

    assert corr.shape == (3, 3)
    assert corr.loc["A", "A"] == pytest.approx(1.0)
    assert corr.loc["A", "B"] == pytest.approx(1.0)
    assert corr.loc["A", "C"] == pytest.approx(0.0, abs=0.4)
    assert corr.loc["B", "A"] == pytest.approx(corr.loc["A", "B"])
