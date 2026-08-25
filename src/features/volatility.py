"""
Volatility and range features.

All features are CAUSAL: each value at timestamp t uses only data available
at or before close_t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_SESSIONS_PER_YEAR = 252


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


def _log_returns(close: pd.Series) -> pd.Series:
    """Log returns."""
    return np.log(close).diff()


def volatility_10d(df: pd.DataFrame) -> pd.Series:
    """Annualized 10-session rolling volatility."""
    log_ret = _log_returns(df["close"])

    vol = log_ret.rolling(
        window=10,
        min_periods=10,
    ).std(ddof=1)

    return vol * np.sqrt(TRADING_SESSIONS_PER_YEAR)


def volatility_20d(df: pd.DataFrame) -> pd.Series:
    """Annualized 20-session rolling volatility."""
    log_ret = _log_returns(df["close"])

    vol = log_ret.rolling(
        window=20,
        min_periods=20,
    ).std(ddof=1)

    return vol * np.sqrt(TRADING_SESSIONS_PER_YEAR)


def high_low_range(df: pd.DataFrame) -> pd.Series:
    """Intraday high-low range normalized by close."""
    return _safe_ratio(
        df["high"] - df["low"],
        df["close"],
    )


def atr_ratio_14(df: pd.DataFrame) -> pd.Series:
    """Average True Range over 14 sessions normalized by close."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(
        window=14,
        min_periods=14,
    ).mean()

    return _safe_ratio(atr, close)


def volatility_ratio(df: pd.DataFrame) -> pd.Series:
    """10-session volatility divided by 20-session volatility."""
    vol_10 = volatility_10d(df)
    vol_20 = volatility_20d(df)

    return _safe_ratio(vol_10, vol_20)