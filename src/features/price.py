"""
Return and price-based features.

All features are CAUSAL: each value at timestamp t uses only data available
at or before close_t.

Features in this module are CANDIDATE features. Predictive value must be
established through out-of-sample ML evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_ratio(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> pd.Series:
    """Compute ratio, returning NaN where denominator is zero or NaN."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full_like(numerator, fill_value=np.nan, dtype=float)
    valid = (denominator != 0) & np.isfinite(denominator) & np.isfinite(numerator)
    result[valid] = numerator[valid] / denominator[valid]
    return pd.Series(result)


def return_1d(df: pd.DataFrame) -> pd.Series:
    """1-session simple return: close_t / close_{t-1} - 1.

    Information used: close[t], close[t-1]
    NaN: first row (no prior close)
    """
    return df["close"].pct_change()


def return_5d(df: pd.DataFrame) -> pd.Series:
    """5-session simple return: close_t / close_{t-5} - 1.

    Information used: close[t], close[t-5]
    NaN: first 5 rows (insufficient lookback)
    """
    return df["close"].pct_change(5)


def return_10d(df: pd.DataFrame) -> pd.Series:
    """10-session simple return: close_t / close_{t-10} - 1.

    Information used: close[t], close[t-10]
    NaN: first 10 rows (insufficient lookback)
    """
    return df["close"].pct_change(10)


def return_20d(df: pd.DataFrame) -> pd.Series:
    """20-session simple return: close_t / close_{t-20} - 1.

    Information used: close[t], close[t-20]
    NaN: first 20 rows (insufficient lookback)
    """
    return df["close"].pct_change(20)


def log_return_1d(df: pd.DataFrame) -> pd.Series:
    """1-session log return: ln(close_t / close_{t-1}).

    Information used: close[t], close[t-1]
    NaN: first row (no prior close)
    """
    return np.log(df["close"]).diff()


def intraday_return(df: pd.DataFrame) -> pd.Series:
    """Intraday return: close_t / open_t - 1.

    This is a valid causal feature because it uses information available at
    close_t to predict the subsequent session's target.

    Information used: open[t], close[t]
    NaN: never (assuming valid OHLCV data)
    """
    return _safe_ratio(df["close"], df["open"]) - 1.0
