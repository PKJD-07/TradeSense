"""Return computations for OHLCV close prices.

All return functions here are CAUSAL: the return at session ``t`` uses only data
up to and including session ``t``. Forward-looking *target* labels live in
``targets.py`` under the explicit execution convention.

ADJUSTED PRICES: return math must be computed on the *adjusted* close series
(the pipeline default ``auto_adjust=True``). Adjusted closes are restated so
returns are continuous across dividends and stock splits. They are an analytical
construct for research — NOT necessarily an executable market price. Future
backtesting must distinguish adjusted research prices from execution prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(close: pd.Series) -> pd.Series:
    """Simple one-session returns: r_t = P_t / P_{t-1} - 1."""
    return close.pct_change()


def log_returns(close: pd.Series) -> pd.Series:
    """Log returns: R_t = ln(P_t / P_{t-1})."""
    return np.log(close).diff()


def n_period_returns(close: pd.Series, n: int = 5, log: bool = False) -> pd.Series:
    """Trailing N-session return (causal): uses P_t and P_{t-n} only.

    Args:
        close: Close-price series.
        n: Number of sessions in the look-back window.
        log: If True return ln(P_t / P_{t-n}), else P_t / P_{t-n} - 1.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if log:
        return np.log(close).diff(n)
    return close.pct_change(n)


def open_to_close_returns(df: pd.DataFrame) -> pd.Series:
    """Open-to-close return of each session: r_t^OC = close_t / open_t - 1.

    This is the return a position entered at session ``t``'s open and closed at
    session ``t``'s close would realize. Under TradeSense's execution
    convention (execute at the next session open), this is the tradable return.
    """
    return df["close"] / df["open"] - 1.0
