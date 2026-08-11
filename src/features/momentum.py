"""
Momentum and trend features using moving averages.

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


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (pandas ewm with span)."""
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


def price_sma_ratio_10(df: pd.DataFrame) -> pd.Series:
    """Price relative to 10-session SMA: close_t / SMA(close, 10)_t.

    Information used: close[t-9:t]
    NaN: first 10 rows (insufficient lookback)
    """
    sma = _sma(df["close"], 10)
    return _safe_ratio(df["close"], sma)


def price_sma_ratio_20(df: pd.DataFrame) -> pd.Series:
    """Price relative to 20-session SMA: close_t / SMA(close, 20)_t.

    Information used: close[t-19:t]
    NaN: first 20 rows (insufficient lookback)
    """
    sma = _sma(df["close"], 20)
    return _safe_ratio(df["close"], sma)


def price_ema_ratio_10(df: pd.DataFrame) -> pd.Series:
    """Price relative to 10-session EMA: close_t / EMA(close, 10)_t.

    EMA uses exponential weighting with alpha = 2/(span+1).

    Information used: close[t-9:t] (exp-weighted)
    NaN: first 10 rows (insufficient lookback)
    """
    ema = _ema(df["close"], 10)
    return _safe_ratio(df["close"], ema)


def sma_cross_10_20(df: pd.DataFrame) -> pd.Series:
    """SMA crossover: SMA(10)_t / SMA(20)_t - 1.

    Positive values indicate short-term trend above long-term trend.
    Negative values indicate short-term trend below long-term trend.

    Information used: close[t-19:t]
    NaN: first 20 rows (insufficient lookback for SMA(20))
    """
    sma_10 = _sma(df["close"], 10)
    sma_20 = _sma(df["close"], 20)
    return _safe_ratio(sma_10, sma_20) - 1.0
