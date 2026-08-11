"""Drawdown analysis.

A drawdown is the decline from a historical peak. The underwater curve is
non-positive and returns to zero whenever a new peak is made.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(close: pd.Series) -> pd.Series:
    """Underwater curve: close / running_max(close) - 1 (<= 0)."""
    running_max = close.cummax()
    return close / running_max - 1.0


def max_drawdown(close: pd.Series) -> float:
    """Maximum peak-to-trough decline (a negative fraction)."""
    dd = drawdown_series(close)
    return float(dd.min()) if len(dd) else 0.0


def max_drawdown_duration(close: pd.Series) -> int:
    """Longest number of consecutive sessions spent below a prior peak."""
    dd = drawdown_series(close)
    if dd.empty:
        return 0
    is_new_peak = (dd == 0).to_numpy()
    positions = np.arange(len(dd))
    last_peak = pd.Series(
        np.where(is_new_peak, positions, np.nan), index=dd.index, dtype="float64"
    ).ffill()
    duration = pd.Series(positions - last_peak.to_numpy(), index=dd.index)
    return int(duration.max())
