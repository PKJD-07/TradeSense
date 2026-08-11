"""Volatility estimation: rolling and realized volatility.

These are causal statistics (each value uses only data up to and including the
current session). The forward-looking volatility *target* is defined in
``targets.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_SESSIONS_PER_YEAR = 252


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = TRADING_SESSIONS_PER_YEAR,
) -> float:
    """Annualized volatility of a return series (sample std * sqrt(periods))."""
    std = returns.std(ddof=1)
    if np.isnan(std):
        return float("nan")
    return float(std * np.sqrt(periods_per_year))


def rolling_volatility(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True,
    periods_per_year: int = TRADING_SESSIONS_PER_YEAR,
) -> pd.Series:
    """Rolling (trailing) volatility over a window of sessions.

    At session ``t`` the value uses returns of sessions ``t-window+1..t``.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    vol = returns.rolling(window=window).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def realized_volatility(
    returns: pd.Series,
    window: int = 5,
    annualize: bool = False,
    periods_per_year: int = TRADING_SESSIONS_PER_YEAR,
) -> pd.Series:
    """Rolling realized volatility = sqrt of the sum of squared returns.

    At session ``t`` the value uses returns of sessions ``t-window+1..t``
    (causal). The forward-looking version used as an ML target is in
    ``targets.future_realized_volatility``.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    rv = np.sqrt((returns**2).rolling(window=window).sum())
    if annualize:
        rv = rv * np.sqrt(periods_per_year / window)
    return rv
