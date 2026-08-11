"""
Volatility and range features.

All features are CAUSAL: each value at timestamp t uses only data available
at or before close_t.

Features in this module are CANDIDATE features. Predictive value must be
established through out-of-sample ML evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_SESSIONS_PER_YEAR = 252


def _safe_ratio(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> pd.Series:
    """Compute ratio, returning NaN where denominator is zero or NaN."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full_like(numerator, fill_value=np.nan, dtype=float)
    valid = (denominator != 0) & np.isfinite(denominator) & np.isfinite(numerator)
    result[valid] = numerator[valid] / denominator[valid]
    return pd.Series(result)


def _log_returns(close: pd.Series) -> pd.Series:
    """Log returns: ln(close_t / close_{t-1})."""
    return np.log(close).diff()


def volatility_10d(df: pd.DataFrame) -> pd.Series:
    """Annualized 10-session rolling volatility.

    volatility_10d[t] = std(log_return[t-9:t]) × √252

    This requires 10 log returns, which requires price observations through t-10.

    Information used: close[t-10:t]
    NaN: first 11 rows (need t-10 for first log return, then 10 returns for std)
    """
    log_ret = _log_returns(df["close"])
    vol = log_ret.rolling(window=10, min_periods=10).std(ddof=1)
    return vol * np.sqrt(TRADING_SESSIONS_PER_YEAR)


def volatility_20d(df: pd.DataFrame) -> pd.Series:
    """Annualized 20-session rolling volatility.

    volatility_20d[t] = std(log_return[t-19:t]) × √252

    This requires 20 log returns, which requires price observations through t-20.

    Information used: close[t-20:t]
    NaN: first 21 rows (need t-20 for first log return, then 20 returns for std)
    """
    log_ret = _log_returns(df["close"])
    vol = log_ret.rolling(window=20, min_periods=20).std(ddof=1)
    return vol * np.sqrt(TRADING_SESSIONS_PER_YEAR)


def high_low_range(df: pd.DataFrame) -> pd.Series:
    """Intraday high-low range normalized by close: (high - low) / close.

    Information used: high[t], low[t], close[t]
    NaN: never (assuming valid OHLCV data)
    """
    return _safe_ratio(df["high"] - df["low"], df["close"])


def atr_ratio_14(df: pd.DataFrame) -> pd.Series:
    """Average True Range (14-session) normalized by close: ATR(14) / close.

    True Range[t] = max(
        high[t] - low[t],
        |high[t] - close[t-1]|,
        |low[t] - close[t-1]|
    )
    ATR(14)[t] = SMA(True Range, 14)[t]

    Information used: high[t-13:t], low[t-13:t], close[t-14:t]
    NaN: first 14 rows (insufficient lookback for ATR)
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range calculation
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()

    return _safe_ratio(atr, close)


def volatility_ratio(df: pd.DataFrame) -> pd.Series:
    """Short-term to long-term volatility ratio: volatility_10d / volatility_20d.

    Information used: close[t-20:t]
    NaN: first 21 rows (determined by volatility_20d)
    """
    vol_10 = volatility_10d(df)
    vol_20 = volatility_20d(df)
    return _safe_ratio(vol_10, vol_20)
