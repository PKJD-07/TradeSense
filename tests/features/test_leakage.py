"""
Leakage regression tests.

These tests are CRITICAL for preventing look-ahead bias. They deliberately
mutate future data and verify that earlier features remain unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.builder import build_features, ALL_FEATURE_NAMES
from tests.features.fixtures import make_long_ohlcv_df, make_candle_df


class TestFutureDataMutation:
    """Test 1: Mutate future data, verify earlier features unchanged."""

    def test_mutating_future_close_does_not_change_past_features(self):
        """Modify close[t+5] and verify features at t remain unchanged."""
        df = make_long_ohlcv_df(symbols=("AAPL",), n=50, seed=1)
        features_original = build_features(df)

        # Mutate future data
        df_mutated = df.copy()
        mutation_idx = 30
        df_mutated.loc[mutation_idx, "close"] *= 2.0

        features_mutated = build_features(df_mutated)

        # Features at indices < mutation_idx should be unchanged
        for feature in ALL_FEATURE_NAMES:
            orig_values = features_original[features_original["symbol"] == "AAPL"][feature].iloc[:mutation_idx]
            mut_values = features_mutated[features_mutated["symbol"] == "AAPL"][feature].iloc[:mutation_idx]
            pd.testing.assert_series_equal(orig_values, mut_values, check_names=False)

    def test_mutating_future_volume_does_not_change_past_features(self):
        df = make_long_ohlcv_df(symbols=("AAPL",), n=50, seed=2)
        features_original = build_features(df)

        df_mutated = df.copy()
        mutation_idx = 25
        df_mutated.loc[mutation_idx, "volume"] *= 10.0

        features_mutated = build_features(df_mutated)

        for feature in ALL_FEATURE_NAMES:
            orig_values = features_original[features_original["symbol"] == "AAPL"][feature].iloc[:mutation_idx]
            mut_values = features_mutated[features_mutated["symbol"] == "AAPL"][feature].iloc[:mutation_idx]
            pd.testing.assert_series_equal(orig_values, mut_values, check_names=False)


class TestRollingWindowBoundary:
    """Test 2: Rolling windows never cross symbol boundaries."""

    def test_outlier_in_one_symbol_does_not_affect_another(self):
        """Set symbol A's last value to outlier, verify B unaffected."""
        df = make_long_ohlcv_df(symbols=("AAPL", "MSFT"), n=40, seed=3)

        # Create extreme outlier in AAPL's last row
        df_mutated = df.copy()
        aapl_last_idx = df_mutated[df_mutated["symbol"] == "AAPL"].index[-1]
        df_mutated.loc[aapl_last_idx, "close"] *= 100.0

        features = build_features(df_mutated)

        # MSFT's features should be unchanged from the non-mutated version
        df_clean = make_long_ohlcv_df(symbols=("AAPL", "MSFT"), n=40, seed=3)
        features_clean = build_features(df_clean)

        msft_features = features[features["symbol"] == "MSFT"]
        msft_features_clean = features_clean[features_clean["symbol"] == "MSFT"]

        for feature in ALL_FEATURE_NAMES:
            pd.testing.assert_series_equal(
                msft_features[feature].reset_index(drop=True),
                msft_features_clean[feature].reset_index(drop=True),
                check_names=False,
            )


class TestCrossAssetAlignment:
    """Test 3: Cross-asset alignment doesn't introduce future values."""

    def test_no_forward_timestamps_introduced(self):
        """Ensure alignment doesn't add timestamps not in the original data."""
        df = make_long_ohlcv_df(symbols=("AAPL", "SPY"), n=30, seed=4)
        features = build_features(df)

        original_timestamps = set(df["timestamp"].unique())
        feature_timestamps = set(features["timestamp"].unique())

        assert feature_timestamps == original_timestamps

    def test_missing_spy_produces_nan_not_interpolation(self):
        """When SPY is missing at a timestamp, features should be NaN."""
        from tests.features.fixtures import make_long_ohlcv_with_gaps

        df = make_long_ohlcv_with_gaps(
            symbols=("AAPL", "SPY"),
            n=25,
            seed=5,
            drop_timestamps={"SPY": [10, 15]},
        )

        features = build_features(df)

        # Find timestamps where SPY is missing
        spy_ts = set(df[df["symbol"] == "SPY"]["timestamp"].unique())
        aapl_ts = set(df[df["symbol"] == "AAPL"]["timestamp"].unique())
        missing_spy_ts = aapl_ts - spy_ts

        for ts in missing_spy_ts:
            aapl_row = features[(features["timestamp"] == ts) & (features["symbol"] == "AAPL")]
            assert aapl_row["spy_return_1d"].isna().all()
            assert aapl_row["relative_return_1d"].isna().all()


class TestTargetIsolation:
    """Test 4: Target columns are not in feature matrix."""

    def test_no_target_columns_in_features(self):
        df = make_long_ohlcv_df(symbols=("AAPL",), n=30, seed=6)
        features = build_features(df)

        target_keywords = ["target", "label", "direction", "forward_return", "realized_vol"]
        for col in features.columns:
            col_lower = col.lower()
            for keyword in target_keywords:
                assert keyword not in col_lower, f"Column '{col}' appears to be a target"


class TestTimestampStability:
    """Test 5: Timestamps remain stable regardless of input order."""

    def test_shuffled_input_produces_same_timestamps(self):
        df = make_long_ohlcv_df(symbols=("AAPL", "MSFT"), n=30, seed=7)

        # Build features from ordered data
        features_ordered = build_features(df)

        # Shuffle input rows
        df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        features_shuffled = build_features(df_shuffled)

        # Extract and sort timestamps per symbol
        for symbol in ["AAPL", "MSFT"]:
            ts_ordered = features_ordered[features_ordered["symbol"] == symbol]["timestamp"].sort_values().reset_index(drop=True)
            ts_shuffled = features_shuffled[features_shuffled["symbol"] == symbol]["timestamp"].sort_values().reset_index(drop=True)
            pd.testing.assert_series_equal(ts_ordered, ts_shuffled)


class TestNoInfiniteValues:
    """Verify no feature produces +inf or -inf values."""

    def test_no_inf_with_valid_data(self):
        df = make_long_ohlcv_df(symbols=("AAPL",), n=50, seed=8)
        features = build_features(df)

        for feature in ALL_FEATURE_NAMES:
            inf_count = (features[feature] == np.inf).sum() + (features[feature] == -np.inf).sum()
            assert inf_count == 0, f"Feature '{feature}' contains {inf_count} infinite values"

    def test_no_inf_with_zero_volume(self):
        from tests.features.fixtures import make_zero_volume_df

        # Create single-symbol DataFrame with zero volume
        df = make_candle_df(n=30, seed=9)
        df.iloc[10:15, df.columns.get_loc("volume")] = 0

        # Convert to long form for build_features
        df_long = df.reset_index()
        df_long.insert(0, "symbol", "AAPL")

        features = build_features(df_long)

        for feature in ALL_FEATURE_NAMES:
            inf_count = (features[feature] == np.inf).sum() + (features[feature] == -np.inf).sum()
            assert inf_count == 0, f"Feature '{feature}' contains {inf_count} infinite values with zero volume"
