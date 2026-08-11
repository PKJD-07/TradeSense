"""Tests for ML target definitions and the execution/timestamp convention.

These tests encode the TradeSense execution convention:

    features  <= close of session t
    signal     after close of session t
    execution  at the OPEN of session t+1
    targets    strictly after close t, anchored at open_{t+1}

The no-look-ahead tests assert that a target at row ``t`` is a pure function of
rows strictly after ``t``, and is UNAFFECTED by perturbing the feature row ``t``
itself (while the target at row ``t-1`` is correctly affected by row ``t``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.targets import (
    next_session_direction,
    forward_return,
    future_realized_volatility,
    DEFAULT_EPSILON,
    DEFAULT_HORIZON,
)
from tests.analysis.fixtures import make_candle_df


def _sign_eps(x: float, epsilon: float) -> float:
    if x > epsilon:
        return 1.0
    if x < -epsilon:
        return -1.0
    return 0.0


class TestNextSessionDirection:
    """Primary V1 target: next session's open-to-close move."""

    def test_defaults(self):
        assert DEFAULT_EPSILON == 0.001
        assert DEFAULT_HORIZON == 5

    def test_label_alignment_matches_manual_shift(self):
        df = make_candle_df(n=40, seed=1)
        y = next_session_direction(df, epsilon=DEFAULT_EPSILON)

        oc = df["close"] / df["open"] - 1.0
        expected = pd.Series(
            [_sign_eps(v, DEFAULT_EPSILON) for v in oc], index=oc.index
        ).shift(-1)

        pd.testing.assert_series_equal(
            y, expected.astype(float), check_names=False
        )
        # Last row has no future session -> NaN
        assert pd.isna(y.iloc[-1])

    def test_epsilon_boundaries(self):
        # Sessions with known open-to-close moves:
        #   session 0: oc = +0.000   -> label at t=-1
        #   session 1: oc = +0.002   -> label at t=0  = +1
        #   session 2: oc = -0.002   -> label at t=1  = -1
        #   session 3: oc = +0.0005  -> |oc| <= eps   -> label at t=2 = 0
        df = make_candle_df(n=5, seed=9)
        oc_values = [0.0, 0.002, -0.002, 0.0005]
        for i, oc in enumerate(oc_values):
            df.iloc[i, df.columns.get_loc("open")] = 100.0
            df.iloc[i, df.columns.get_loc("close")] = 100.0 * (1.0 + oc)

        y = next_session_direction(df, epsilon=DEFAULT_EPSILON)
        assert y.iloc[0] == 1.0
        assert y.iloc[1] == -1.0
        assert y.iloc[2] == 0.0
        assert pd.isna(y.iloc[-1])

    def test_epsilon_zero_has_no_neutral_zone(self):
        df = make_candle_df(n=5, seed=9)
        y = next_session_direction(df, epsilon=0.0)
        assert set(y.dropna().unique()) <= {1.0, -1.0}

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError):
            next_session_direction(make_candle_df(n=5), epsilon=-0.1)

    def test_no_look_ahead_feature_row_perturbation(self):
        """Radically changing row t must not change the label at row t."""
        df = make_candle_df(n=12, seed=5)
        t = 4
        y = next_session_direction(df, epsilon=DEFAULT_EPSILON)

        df2 = df.copy()
        df2.iloc[t, :] = [1.0, 2.0, 0.5, 3.0, 1]  # open high low close volume
        y2 = next_session_direction(df2, epsilon=DEFAULT_EPSILON)

        # Label at t is unaffected (uses only session t+1)
        assert y.iloc[t] == y2.iloc[t]
        # Label at t-1 legitimately uses session t and therefore changes:
        # new oc at t = 3.0/1.0 - 1 = 2.0 -> +1
        assert y2.iloc[t - 1] == 1.0

    def test_label_uses_session_t_plus_one_not_t_plus_two(self):
        df = make_candle_df(n=15, seed=6)
        t = 5
        y = next_session_direction(df, epsilon=DEFAULT_EPSILON)

        # Change session t+2's open/close -> label at t must NOT change
        df2 = df.copy()
        df2.iloc[t + 2, df2.columns.get_loc("close")] = df2.iloc[t + 2]["close"] * 10.0
        y2 = next_session_direction(df2, epsilon=DEFAULT_EPSILON)
        assert y.iloc[t] == y2.iloc[t]

        # Change session t+1's close -> label at t MUST change accordingly
        df3 = df.copy()
        df3.iloc[t + 1, df3.columns.get_loc("close")] = df3.iloc[t + 1]["close"] * 10.0
        y3 = next_session_direction(df3, epsilon=DEFAULT_EPSILON)
        oc_new = df3.iloc[t + 1]["close"] / df3.iloc[t + 1]["open"] - 1.0
        assert y3.iloc[t] == _sign_eps(oc_new, DEFAULT_EPSILON)


class TestForwardReturn:
    """Secondary target: close_{t+n} / open_{t+1} - 1."""

    def test_math_matches_manual(self):
        df = make_candle_df(n=20, seed=2)
        n = DEFAULT_HORIZON
        y = forward_return(df, n=n)
        for t in range(len(df) - n):
            expected = df.iloc[t + n]["close"] / df.iloc[t + 1]["open"] - 1.0
            assert y.iloc[t] == pytest.approx(expected)
        assert y.iloc[-n:].isna().all()

    def test_no_look_ahead(self):
        df = make_candle_df(n=15, seed=3)
        t, n = 4, 5
        y = forward_return(df, n=n)

        # Perturb feature row t -> label at t unchanged
        df2 = df.copy()
        df2.iloc[t, df2.columns.get_loc("close")] *= 10.0
        y2 = forward_return(df2, n=n)
        assert y.iloc[t] == y2.iloc[t]

        # Perturb the execution row (t+1 open) -> label at t changes
        df3 = df.copy()
        df3.iloc[t + 1, df3.columns.get_loc("open")] *= 10.0
        y3 = forward_return(df3, n=n)
        expected = df3.iloc[t + n]["close"] / df3.iloc[t + 1]["open"] - 1.0
        assert y3.iloc[t] == pytest.approx(expected)

    def test_horizon_one_equals_execution_anchored_return(self):
        df = make_candle_df(n=20, seed=4)
        y = forward_return(df, n=1)
        for t in range(len(df) - 1):
            expected = df.iloc[t + 1]["close"] / df.iloc[t + 1]["open"] - 1.0
            assert y.iloc[t] == pytest.approx(expected)


class TestFutureRealizedVolatility:
    """Secondary target: sqrt(sum of squared log OC returns over next n days)."""

    def test_math_matches_manual(self):
        df = make_candle_df(n=20, seed=4)
        n = DEFAULT_HORIZON
        y = future_realized_volatility(df, n=n)
        log_oc = np.log(df["close"] / df["open"])
        for t in range(len(df) - n):
            expected = np.sqrt(np.sum(log_oc.iloc[t + 1 : t + 1 + n].to_numpy() ** 2))
            assert y.iloc[t] == pytest.approx(expected)
        assert y.iloc[-n:].isna().all()

    def test_annualized(self):
        df = make_candle_df(n=30, seed=5)
        y_raw = future_realized_volatility(df, n=5, annualize=False)
        y_ann = future_realized_volatility(df, n=5, annualize=True)
        assert y_ann.iloc[0] == pytest.approx(y_raw.iloc[0] * np.sqrt(252 / 5))

    def test_no_look_ahead(self):
        df = make_candle_df(n=15, seed=6)
        t, n = 4, 5
        y = future_realized_volatility(df, n=n)

        # Perturb feature row t -> label at t unchanged
        df2 = df.copy()
        df2.iloc[t, df2.columns.get_loc("close")] *= 10.0
        y2 = future_realized_volatility(df2, n=n)
        assert y.iloc[t] == y2.iloc[t]

        # Perturb a held-day row (t+1) -> label at t changes
        df3 = df.copy()
        df3.iloc[t + 1, df3.columns.get_loc("close")] *= 10.0
        y3 = future_realized_volatility(df3, n=n)
        log_oc3 = np.log(df3["close"] / df3["open"])
        expected = np.sqrt(np.sum(log_oc3.iloc[t + 1 : t + 1 + n].to_numpy() ** 2))
        assert y3.iloc[t] == pytest.approx(expected)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            future_realized_volatility(make_candle_df(n=10), n=0)


class TestConventionShared:
    """Shared invariant: features <= t, targets strictly after t."""

    def test_all_targets_are_nan_at_the_tail(self):
        df = make_candle_df(n=30, seed=7)
        assert next_session_direction(df).iloc[-1:].isna().all()
        assert forward_return(df, n=5).iloc[-5:].isna().all()
        assert future_realized_volatility(df, n=5).iloc[-5:].isna().all()

    def test_all_targets_are_finite_on_the_interior(self):
        df = make_candle_df(n=40, seed=8)
        interior = slice(1, -5)
        assert next_session_direction(df).iloc[interior].notna().all()
        assert forward_return(df, n=5).iloc[interior].notna().all()
        assert future_realized_volatility(df, n=5).iloc[interior].notna().all()
