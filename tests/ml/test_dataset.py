"""Tests for src.ml.dataset: alignment, per-symbol targets, NaN accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.constants import DEFAULT_EPSILON, TARGET_COLUMN, WARMUP_ROWS
from src.ml.dataset import build_ml_dataset
from tests.ml.fixtures import (
    make_ml_dataset,
    make_ml_dataset_with_gaps,
)
from tests.features.fixtures import make_long_ohlcv_df, make_long_ohlcv_with_gaps


def _expected_direction_for_row(long_ohlcv, symbol, ts, epsilon):
    """Hand-rolled target for row (symbol, ts): y = sign_eps(OC_{t+1})."""
    sub = (
        long_ohlcv[long_ohlcv["symbol"] == symbol]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    pos = sub.index[sub["timestamp"] == ts][0]
    if pos == len(sub) - 1:
        return None  # no future session
    oc = sub.loc[pos + 1, "close"] / sub.loc[pos + 1, "open"] - 1.0
    if oc > epsilon:
        return 1
    if oc < -epsilon:
        return -1
    return 0


class TestPanelStructure:
    def test_columns_and_ordering(self):
        ds = make_ml_dataset(symbols=("AAPL", "MSFT"), n=40, seed=3)
        assert TARGET_COLUMN in ds.df.columns
        assert "timestamp" in ds.df.columns and "symbol" in ds.df.columns
        for col in ds.feature_columns:
            assert col in ds.df.columns
        # canonical ordering: grouped by symbol then timestamp ascending
        df = ds.df.reset_index(drop=True)
        for symbol in df["symbol"].unique():
            sub = df[df["symbol"] == symbol]
            assert sub["timestamp"].is_monotonic_increasing
        # symbols appear grouped (no interleaving)
        assert (df["symbol"].ne(df["symbol"].shift()).cumsum().max() ==
                df["symbol"].nunique())

    def test_feature_target_metadata_separation(self):
        ds = make_ml_dataset(symbols=("AAPL", "SPY"), n=40, seed=5)
        assert ds.metadata_columns == ["timestamp", "symbol"]
        assert list(ds.X.columns) == ds.feature_columns
        assert ds.y.name == TARGET_COLUMN
        assert len(ds.feature_columns) == 20  # full causal feature set
        # target is never among the feature columns
        assert TARGET_COLUMN not in ds.feature_columns
        for keyword in ("target", "direction", "label"):
            assert all(keyword not in c for c in ds.feature_columns)

    def test_timestamps_traceable_and_utc(self):
        ds = make_ml_dataset(symbols=("AAPL", "MSFT"), n=30, seed=6)
        ts = pd.to_datetime(ds.df["timestamp"], utc=True)
        assert ts.dt.tz is not None
        assert ts.dt.tz.utcoffset(None) == pd.Timedelta(0)

    def test_empty_input_raises(self):
        empty = pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
        with pytest.raises(ValueError):
            build_ml_dataset(empty)

    def test_missing_columns_raises(self):
        with pytest.raises(ValueError):
            build_ml_dataset(pd.DataFrame({"foo": [1]}))


class TestPerSymbolTargets:
    """The shift(-1) inside target functions must never cross symbol boundaries."""

    def test_target_math_matches_manual_per_row(self):
        long_ohlcv = make_long_ohlcv_df(symbols=("AAPL", "MSFT", "SPY"), n=30, seed=9)
        ds = build_ml_dataset(long_ohlcv)
        for _, row in ds.df.iterrows():
            expected = _expected_direction_for_row(
                long_ohlcv, row["symbol"], row["timestamp"], ds.report.epsilon
            )
            assert expected is not None  # no-target rows are dropped
            assert int(row[TARGET_COLUMN]) == expected

    def test_last_row_of_each_symbol_is_dropped_not_cross_leaked(self):
        """If targets were pooled, the last AAPL row would read MSFT's first
        session. It must instead be dropped as an unavailable target."""
        long_ohlcv = make_long_ohlcv_df(symbols=("AAPL", "MSFT"), n=40, seed=11)
        ds = build_ml_dataset(long_ohlcv, include_market_context=False)
        # 1 unavailable-target row per symbol
        assert ds.report.dropped_no_target == 2
        # last AAPL timestamp is absent from the panel
        aapl_ts = sorted(long_ohlcv[long_ohlcv["symbol"] == "AAPL"]["timestamp"])[-1]
        assert not (ds.df["timestamp"] == aapl_ts).any()

    def test_target_uses_only_session_t_plus_one(self):
        """Perturbing session t+1 changes y_t; perturbing session t does not."""
        long_ohlcv = make_long_ohlcv_df(symbols=("AAPL",), n=30, seed=13)
        sub = long_ohlcv[long_ohlcv["symbol"] == "AAPL"].sort_values("timestamp").reset_index(drop=True)
        t = 24  # session index within AAPL, beyond warm-up (rows 0..20) and not last
        ts_t = sub.loc[t, "timestamp"]
        ts_next = sub.loc[t + 1, "timestamp"]

        base = build_ml_dataset(long_ohlcv, include_market_context=False)
        y_base = base.df.loc[
            (base.df["symbol"] == "AAPL") & (base.df["timestamp"] == ts_t), TARGET_COLUMN
        ].iloc[0]

        # Perturb session t (the feature row) -> y_t unchanged
        df2 = long_ohlcv.copy()
        mask = (df2["symbol"] == "AAPL") & (df2["timestamp"] == ts_t)
        df2.loc[mask, "close"] = df2.loc[mask, "close"] * 10.0
        ds2 = build_ml_dataset(df2, include_market_context=False)
        y2 = ds2.df.loc[
            (ds2.df["symbol"] == "AAPL") & (ds2.df["timestamp"] == ts_t), TARGET_COLUMN
        ].iloc[0]
        assert int(y2) == int(y_base)

        # Perturb session t+1's close -> y_t changes to the forced sign
        df3 = long_ohlcv.copy()
        mask3 = (df3["symbol"] == "AAPL") & (df3["timestamp"] == ts_next)
        # force a large positive open-to-close move at session t+1
        df3.loc[mask3, "close"] = df3.loc[mask3, "open"] * 1.02
        ds3 = build_ml_dataset(df3, include_market_context=False)
        y3 = ds3.df.loc[
            (ds3.df["symbol"] == "AAPL") & (ds3.df["timestamp"] == ts_t), TARGET_COLUMN
        ].iloc[0]
        assert int(y3) == 1


class TestNoLookAhead:
    def test_future_perturbation_does_not_change_earlier_X(self):
        long_ohlcv = make_long_ohlcv_df(symbols=("AAPL", "MSFT"), n=40, seed=15)
        mutation_idx = 30
        ds_base = build_ml_dataset(long_ohlcv, include_market_context=False)

        df2 = long_ohlcv.copy()
        df2.loc[mutation_idx, "close"] *= 2.0
        df2.loc[mutation_idx, "volume"] *= 10.0
        ds2 = build_ml_dataset(df2, include_market_context=False)

        # For all panel rows whose timestamp precedes the mutated session,
        # feature values must be bit-identical.
        mutated_ts = long_ohlcv.loc[mutation_idx, "timestamp"]
        mask_base = ds_base.df["timestamp"] < mutated_ts
        mask_2 = ds2.df["timestamp"] < mutated_ts
        pd.testing.assert_frame_equal(
            ds_base.df.loc[mask_base, ds_base.feature_columns].reset_index(drop=True),
            ds2.df.loc[mask_2, ds2.feature_columns].reset_index(drop=True),
        )

    def test_target_not_present_in_X(self):
        ds = make_ml_dataset(symbols=("AAPL",), n=30, seed=17)
        for col in ds.X.columns:
            assert col != TARGET_COLUMN
            assert not col.lower().startswith("target")


class TestNeutralClass:
    def test_three_classes_retained(self):
        ds = make_ml_dataset(symbols=("AAPL", "MSFT", "JPM"), n=120, seed=19)
        classes = set(ds.y.unique())
        assert classes <= {-1, 0, 1}
        assert 0 in classes  # neutral class is present, not silently dropped

    def test_epsilon_zero_removes_neutral_band(self):
        long_ohlcv = make_long_ohlcv_df(symbols=("AAPL",), n=60, seed=21)
        ds = build_ml_dataset(long_ohlcv, epsilon=0.0, include_market_context=False)
        assert set(ds.y.unique()) <= {-1, 1}
        assert 0 not in set(ds.y.unique())

    def test_class_counts_reported(self):
        ds = make_ml_dataset(symbols=("AAPL",), n=100, seed=23)
        assert sum(ds.report.class_counts.values()) == len(ds.df)
        assert set(ds.report.class_counts.keys()) <= {-1, 0, 1}


class TestDropAccounting:
    def test_warmup_rows_dropped_per_symbol(self):
        n, n_symbols = 40, 4
        ds = make_ml_dataset(symbols=("AAPL", "MSFT", "JPM", "XOM"), n=n, seed=25)
        assert ds.report.warmup_rows == WARMUP_ROWS
        assert ds.report.dropped_warmup == WARMUP_ROWS * n_symbols

    def test_no_target_rows_dropped_one_per_symbol(self):
        n_symbols = 5
        ds = make_ml_dataset(n=60, seed=27)
        assert ds.report.dropped_no_target == n_symbols
        for symbol in ds.symbols:
            assert ds.report.dropped_no_target_by_symbol[symbol] == 1

    def test_feature_nan_drops_recorded_with_gaps(self):
        ds = make_ml_dataset_with_gaps(drop_timestamps={"SPY": [25, 30]})
        # AAPL rows at the missing-SPY timestamps get NaN spy features -> dropped
        assert ds.report.dropped_nan_features >= 2
        assert ds.report.feature_nan_counts["spy_return_1d"] >= 2
        assert ds.report.dropped_nan_features_by_symbol["AAPL"] >= 2

    def test_max_nan_fraction_guard_raises(self):
        long_ohlcv = make_long_ohlcv_with_gaps(
            symbols=("AAPL", "SPY"), n=50, seed=29, drop_timestamps={"SPY": [25, 30]}
        )
        with pytest.raises(ValueError, match="spy_return_1d"):
            build_ml_dataset(long_ohlcv, max_nan_fraction=0.0)

    def test_counts_are_consistent(self):
        ds = make_ml_dataset(symbols=("AAPL", "MSFT"), n=40, seed=31)
        assert (
            ds.report.final_rows
            == ds.report.rows_before_drops
            - ds.report.dropped_warmup
            - ds.report.dropped_no_target
            - ds.report.dropped_nan_features
        )
        assert ds.report.final_rows == len(ds.df)


class TestDeterminism:
    def test_build_is_deterministic(self):
        long_ohlcv = make_long_ohlcv_df(symbols=("AAPL", "MSFT"), n=50, seed=33)
        ds1 = build_ml_dataset(long_ohlcv, include_market_context=False)
        ds2 = build_ml_dataset(long_ohlcv.copy(), include_market_context=False)
        pd.testing.assert_frame_equal(ds1.df, ds2.df)
        assert ds1.report == ds2.report

    def test_symbols_and_date_ranges_reported(self):
        ds = make_ml_dataset(symbols=("AAPL", "MSFT", "SPY"), n=40, seed=35)
        assert ds.symbols == ["AAPL", "MSFT", "SPY"]
        assert set(ds.report.date_range_by_symbol.keys()) == set(ds.symbols)
        for symbol, (lo, hi) in ds.report.date_range_by_symbol.items():
            assert lo <= hi
