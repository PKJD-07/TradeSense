"""Statistical summaries for return series.

Implements descriptive statistics, return autocorrelation (ACF only), and
cross-asset return correlation. Deliberately does NOT implement PACF / partial
autocorrelation or an ARIMA-style toolkit for V1 (no statsmodels dependency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.analysis.volatility import (
    annualized_volatility,
    TRADING_SESSIONS_PER_YEAR,
)


def describe_returns(
    returns: pd.Series,
    periods_per_year: int = TRADING_SESSIONS_PER_YEAR,
) -> pd.Series:
    """Descriptive statistics for a return series (daily scale + annualized).

    Returns count, mean, std, min, max, skew, excess kurtosis, annualized mean
    and annualized volatility. NaN rows are dropped first.
    """
    returns = returns.dropna()
    if returns.empty:
        return pd.Series(dtype=float)

    n = len(returns)
    if n >= 3:
        skew_val = float(sp_stats.skew(returns, bias=False))
        kurt_val = float(sp_stats.kurtosis(returns, bias=False, fisher=True))
    else:
        skew_val = float("nan")
        kurt_val = float("nan")

    return pd.Series(
        {
            "count": float(n),
            "mean": float(returns.mean()),
            "std": float(returns.std(ddof=1)),
            "min": float(returns.min()),
            "max": float(returns.max()),
            "skew": skew_val,
            "kurtosis": kurt_val,
            "annualized_mean": float(returns.mean() * periods_per_year),
            "annualized_vol": annualized_volatility(returns, periods_per_year),
        }
    )


def autocorrelation(series: pd.Series, lags: int = 10) -> pd.Series:
    """Sample autocorrelation (ACF) of a series for lags 1..lags.

    Uses the Pearson correlation between the series and its ``lag``-shifted
    self over the overlapping window (``series.corr(series.shift(lag))``).
    """
    if lags < 1:
        raise ValueError("lags must be >= 1")
    values = {}
    for lag in range(1, lags + 1):
        values[lag] = series.corr(series.shift(lag))
    return pd.Series(values, dtype=float, name="autocorrelation")


def cross_asset_correlation(close_wide: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of simple returns across assets.

    ``close_wide`` has symbols as columns and timestamps as index (e.g. the
    output of :func:`src.analysis.convert.pivot_close_prices`).
    """
    returns = close_wide.pct_change()
    return returns.corr()
