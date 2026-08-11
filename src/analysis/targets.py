"""ML target (label) definitions for TradeSense.

EXECUTION / TIMESTAMP CONVENTION (read this first)
--------------------------------------------------
* A sample is indexed by session ``t`` (one daily bar).
* **Features** for sample ``t`` may use information available at or before the
  close of session ``t``. Nothing after ``close_t`` may enter the features.
* The **signal** is generated after the close of session ``t``.
* The position is **executed at the open of session ``t+1``** — the earliest
  moment the signal can be acted on.
* **Targets** for sample ``t`` therefore use information strictly *after*
  ``close_t`` and are anchored at the day ``t+1`` open.

This convention deliberately avoids the classic look-ahead error of defining a
target on the close-to-close return ``close_{t+1}/close_t - 1``: that quantity
includes the overnight move from ``close_t`` to ``open_{t+1}``, which is NOT
capturable under our convention (we cannot transact at ``close_t``). Anchoring
labels at ``open_{t+1}`` makes the target consistent with the execution model.

MATH (all labels are strictly future)
-------------------------------------
r_{t+1}^OC = close_{t+1}/open_{t+1} - 1          (tradable open-to-close move)

1. next_session_direction   y_t = +1 if r^OC >  eps
                                  -1 if r^OC < -eps
                                   0 if |r^OC| <= eps
                            eps = 0.001 default (configurable; do NOT tune on
                            the test set). Row ``t`` is the label for session
                            ``t+1``.

2. forward_return           y_t = close_{t+N}/open_{t+1} - 1     (N default 5)
                            Return from entry at open_{t+1} to exit at
                            close_{t+N}.

3. future_realized_vol      y_t = sqrt( sum_{i=1..N} R_{t+i}^2 )
                            R_{t+i} = ln(close_{t+i}/open_{t+i})
                            (optionally annualized by sqrt(252/N))

All functions return a pandas Series aligned to the input index; rows for which
no future session exists are NaN.

This module constructs LABELS ONLY. It does not build models or features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_EPSILON = 0.001
DEFAULT_HORIZON = 5


def next_session_direction(
    df: pd.DataFrame,
    epsilon: float = DEFAULT_EPSILON,
) -> pd.Series:
    """Primary V1 target: direction of the next session's open-to-close move.

    y_t = +1 if close_{t+1}/open_{t+1} - 1 > epsilon
          -1 if close_{t+1}/open_{t+1} - 1 < -epsilon
           0 otherwise (neutral zone).

    Rows labeled 0 (|move| <= epsilon) are typically dropped for a binary fit,
    or kept as a third class. The final row is NaN (no future session).
    """
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    oc = df["close"] / df["open"] - 1.0
    direction = pd.Series(0.0, index=oc.index, dtype=float)
    direction[oc > epsilon] = 1.0
    direction[oc < -epsilon] = -1.0
    return direction.shift(-1)


def forward_return(
    df: pd.DataFrame,
    n: int = DEFAULT_HORIZON,
) -> pd.Series:
    """Secondary target: return from entry at open_{t+1} to exit at close_{t+N}.

    y_t = close_{t+n} / open_{t+1} - 1
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    close = df["close"].shift(-n)  # close at t+n
    open_ = df["open"].shift(-1)   # open at t+1
    return close / open_ - 1.0


def future_realized_volatility(
    df: pd.DataFrame,
    n: int = DEFAULT_HORIZON,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """Secondary target: realized volatility of the next n open-to-close moves.

    y_t = sqrt( sum_{i=1..n} ln(close_{t+i}/open_{t+i})^2 )

    Computed on log open-to-close returns of sessions t+1..t+n, matching the
    execution convention. Optionally annualized by sqrt(periods_per_year / n).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    log_oc = np.log(df["close"] / df["open"])
    rv = np.sqrt((log_oc**2).rolling(window=n).sum())
    rv = rv.shift(-n)
    if annualize:
        rv = rv * np.sqrt(periods_per_year / n)
    return rv
