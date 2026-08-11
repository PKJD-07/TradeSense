"""
Volume-based features.

All features are CAUSAL: each value at timestamp t uses only data available
at or before close_t.

Features in this module are CANDIDATE features. Predictive value must be
established through out-of-sample ML evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_ratio(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> pd.Series:
    """Compute ratio, returning NaN where denominator is zero or NaN.

    Preserves the input index so the result aligns when a DataFrame is built
    from feature series (a RangeIndex would fail to align to a DatetimeIndex).
    """
    index = getattr(numerator, "index", None)
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full_like(numerator, fill_value=np.nan, dtype=float)
    valid = (denominator != 0) & np.isfinite(denominator) & np.isfinite(numerator)
    result[valid] = numerator[valid] / denominator[valid]
    return pd.Series(result, index=index)


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average with explicit window."""
    return series.rolling(window=window, min_periods=window).mean()


def volume_change_1d(df: pd.DataFrame) -> pd.Series:
    """1-session volume change: volume_t / volume_{t-1} - 1.

    Returns NaN if volume_{t-1} is zero (avoids division by zero).

    Information used: volume[t], volume[t-1]
    NaN: first row, or any row where prior volume is zero
    """
    vol = df["volume"].astype(float)
    vol_prev = vol.shift(1)
    return _safe_ratio(vol, vol_prev) - 1.0


def relative_volume_10d(df: pd.DataFrame) -> pd.Series:
    """Volume relative to 10-session average: volume_t / SMA(volume, 10)_t.

    Returns NaN if the 10-session average volume is zero.

    Information used: volume[t-9:t]
    NaN: first 10 rows, or any row where 10-day average is zero
    """
    vol = df["volume"].astype(float)
    sma_vol = _sma(vol, 10)
    return _safe_ratio(vol, sma_vol)


def volume_trend_5d(df: pd.DataFrame) -> pd.Series:
    """Volume trend: SMA(volume, 5) / SMA(volume, 20) - 1.

    Positive values indicate short-term volume above long-term average.
    Returns NaN if SMA(20) is zero.

    Information used: volume[t-19:t]
    NaN: first 20 rows, or any row where 20-day average is zero
    """
    vol = df["volume"].astype(float)
    sma_5 = _sma(vol, 5)
    sma_20 = _sma(vol, 20)
    return _safe_ratio(sma_5, sma_20) - 1.0
