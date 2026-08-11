"""
Market-context (cross-asset) features using SPY as a market proxy.

All features are CAUSAL: each value at timestamp t uses only data available
at or before close_t.

Cross-asset alignment uses LEFT JOIN from each stock's timestamps onto SPY.
Stock observations are always preserved. If SPY is missing at a stock timestamp,
market-context features are set to NaN.

Features in this module are CANDIDATE features. Predictive value must be
established through out-of-sample ML evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPY_SYMBOL = "SPY"


def _safe_ratio(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> pd.Series:
    """Compute ratio, returning NaN where denominator is zero or NaN."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full_like(numerator, fill_value=np.nan, dtype=float)
    valid = (denominator != 0) & np.isfinite(denominator) & np.isfinite(numerator)
    result[valid] = numerator[valid] / denominator[valid]
    return pd.Series(result)


def _compute_spy_return_1d(spy_df: pd.DataFrame) -> pd.Series:
    """Compute SPY's 1-session return series."""
    if spy_df.empty:
        return pd.Series(dtype=float, name="spy_return_1d")
    return spy_df["close"].pct_change().rename("spy_return_1d")


def _filter_spy_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract SPY data from multi-symbol DataFrame."""
    if "symbol" not in df.columns:
        raise ValueError("Input DataFrame must have 'symbol' column for cross-asset features")
    spy_data = df[df["symbol"] == SPY_SYMBOL].copy()
    if spy_data.empty:
        return pd.DataFrame(columns=["timestamp", "close"])
    return spy_data[["timestamp", "close"]].reset_index(drop=True)


def compute_market_context(
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute market-context features for all symbols in df.

    Uses LEFT JOIN from each stock's timestamps onto SPY. Stock observations
    are always preserved. Missing SPY data results in NaN market-context features.

    Args:
        df: Long-form DataFrame with columns [symbol, timestamp, ...]
        spy_df: Optional SPY-specific DataFrame. If None, SPY is extracted from df.

    Returns:
        DataFrame with columns [timestamp, symbol, spy_return_1d, relative_return_1d]
    """
    if "symbol" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("Input DataFrame must have 'symbol' and 'timestamp' columns")

    # Extract SPY data
    if spy_df is None:
        spy_data = _filter_spy_data(df)
    else:
        spy_data = spy_df[["timestamp", "close"]].copy()

    # Compute SPY return
    if spy_data.empty:
        # No SPY data available
        result = df[["timestamp", "symbol"]].copy()
        result["spy_return_1d"] = np.nan
        result["relative_return_1d"] = np.nan
        return result

    spy_data = spy_data.set_index("timestamp")
    spy_data["spy_return_1d"] = spy_data["close"].pct_change()
    spy_returns = spy_data[["spy_return_1d"]].reset_index()

    # Compute stock returns (for relative return)
    stock_returns = df[["timestamp", "symbol", "close"]].copy()
    stock_returns = stock_returns.sort_values(["symbol", "timestamp"])
    stock_returns["return_1d"] = stock_returns.groupby("symbol")["close"].transform(
        lambda x: x.pct_change()
    )

    # LEFT JOIN: stock timestamps preserved, SPY data may be missing
    merged = stock_returns.merge(
        spy_returns,
        on="timestamp",
        how="left",
    )

    # Compute relative return
    merged["relative_return_1d"] = merged["return_1d"] - merged["spy_return_1d"]

    return merged[["timestamp", "symbol", "spy_return_1d", "relative_return_1d"]]


def spy_return_1d(df: pd.DataFrame, spy_df: pd.DataFrame | None = None) -> pd.Series:
    """SPY's 1-session return aligned to each stock's timestamps.

    Uses LEFT JOIN from stock timestamps onto SPY. Missing SPY data results in NaN.

    Information used: SPY close[t], SPY close[t-1]
    NaN: first SPY observation, or any timestamp where SPY is missing
    """
    market_ctx = compute_market_context(df, spy_df)
    result = market_ctx.set_index(["timestamp", "symbol"])["spy_return_1d"]
    return result


def relative_return_1d(df: pd.DataFrame, spy_df: pd.DataFrame | None = None) -> pd.Series:
    """Stock return minus SPY return: return_1d_stock - return_1d_spy.

    Uses LEFT JOIN from stock timestamps onto SPY. Missing SPY data results in NaN.

    Information used: stock close[t-1:t], SPY close[t-1:t]
    NaN: first observation per symbol, or any timestamp where SPY is missing
    """
    market_ctx = compute_market_context(df, spy_df)
    result = market_ctx.set_index(["timestamp", "symbol"])["relative_return_1d"]
    return result
