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


def _safe_ratio(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
) -> pd.Series:
    """Compute ratio while preserving the original Series index."""
    if isinstance(numerator, pd.Series):
        index = numerator.index
    elif isinstance(denominator, pd.Series):
        index = denominator.index
    else:
        index = None

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    result = np.full_like(numerator, fill_value=np.nan, dtype=float)

    valid = (
        (denominator != 0)
        & np.isfinite(denominator)
        & np.isfinite(numerator)
    )

    result[valid] = numerator[valid] / denominator[valid]

    return pd.Series(result, index=index)


def return_1d(df: pd.DataFrame) -> pd.Series:
    """1-session simple return: close_t / close_{t-1} - 1."""
    return df["close"].pct_change()


def return_5d(df: pd.DataFrame) -> pd.Series:
    """5-session simple return: close_t / close_{t-5} - 1."""
    return df["close"].pct_change(5)


def return_10d(df: pd.DataFrame) -> pd.Series:
    """10-session simple return: close_t / close_{t-10} - 1."""
    return df["close"].pct_change(10)


def return_20d(df: pd.DataFrame) -> pd.Series:
    """20-session simple return: close_t / close_{t-20} - 1."""
    return df["close"].pct_change(20)


def log_return_1d(df: pd.DataFrame) -> pd.Series:
    """1-session log return: ln(close_t / close_{t-1})."""
    return np.log(df["close"]).diff()


def intraday_return(df: pd.DataFrame) -> pd.Series:
    """Intraday return: close_t / open_t - 1."""
    return _safe_ratio(df["close"], df["open"]) - 1.0