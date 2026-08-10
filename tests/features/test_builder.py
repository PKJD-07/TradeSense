"""Tests for the feature builder orchestrator."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.builder import (
    build_features,
    get_feature_names,
    ALL_FEATURE_NAMES,
    STOCK_FEATURES,
)
from tests.features.fixtures import make_long_ohlcv_df, make_long_ohlcv_with_gaps


class TestBuildFeatures:
    """Tests for the build_features orchestrator."""

    def test_output_has_all_features(self):
        df = make_long_ohlcv_df(symbols=("AAPL", "SPY"), n=50, seed=1)
        features = build_features(df)
        assert "timestamp" in features.columns
        assert "symbol" in features.columns
        for name in ALL_FEATURE_NAMES:
            assert name in features.columns

    def test_per_symbol_isolation(self):
        """Features for one symbol should not affect another."""
        df = make_long_ohlcv_df(symbols=("AAPL", "MSFT"), n=30, seed=2)
        features = build_features(df)

        aapl_features = features[features["symbol"] == "AAPL"]
        msft_features = features[features["symbol"] == "MSFT"]

        # Should have same number of rows per symbol
        assert len(aapl_features) == len(msft_features)

        # NaN patterns should be independent (determined by each symbol's data)
        # Just verify they're computed separately
        assert aapl_features["return_1d"].isna().sum() == 1  # First row
        assert msft_features["return_1d"].isna().sum() == 1

    def test_no_target_columns_in_output(self):
        df = make_long_ohlcv_df(symbols=("AAPL",), n=30, seed=3)
        features = build_features(df)

        # Verify no target-related columns
        target_keywords = ["target", "label", "direction", "forward_return", "realized_vol"]
        for col in features.columns:
            col_lower = col.lower()
            for keyword in target_keywords:
                assert keyword not in col_lower, f"Column '{col}' may be a target"

    def test_length_preserved(self):
        df = make_long_ohlcv_df(symbols=("AAPL", "MSFT", "SPY"), n=40, seed=4)
        features = build_features(df)
        # Each symbol has 40 rows
        assert len(features) == 120

    def test_timestamps_sorted_per_symbol(self):
        df = make_long_ohlcv_df(symbols=("AAPL", "MSFT"), n=30, seed=5)
        features = build_features(df)

        for symbol in ["AAPL", "MSFT"]:
            symbol_features = features[features["symbol"] == symbol]
            timestamps = symbol_features["timestamp"].tolist()
            assert timestamps == sorted(timestamps)

    def test_market_context_can_be_disabled(self):
        df = make_long_ohlcv_df(symbols=("AAPL",), n=30, seed=6)
        features_with = build_features(df, include_market_context=True)
        features_without = build_features(df, include_market_context=False)

        # Both should have the same number of rows
        assert len(features_with) == len(features_without)

        # Market context columns present in one, not the other
        assert "spy_return_1d" in features_with.columns
        assert "spy_return_1d" not in features_without.columns

        # All other feature columns should be identical
        common_cols = [c for c in features_without.columns if c not in ["timestamp", "symbol"]]
        for col in common_cols:
            pd.testing.assert_series_equal(
                features_with[col].reset_index(drop=True),
                features_without[col].reset_index(drop=True),
                check_names=False,
            )

    def test_missing_spy_produces_nan_market_context(self):
        df = make_long_ohlcv_df(symbols=("AAPL",), n=30, seed=7)  # No SPY
        features = build_features(df)
        assert features["spy_return_1d"].isna().all()
        assert features["relative_return_1d"].isna().all()

    def test_requires_symbol_column(self):
        df = make_long_ohlcv_df(n=20, seed=1).drop(columns=["symbol"])
        with pytest.raises(ValueError, match="symbol"):
            build_features(df)

    def test_requires_timestamp_column(self):
        df = make_long_ohlcv_df(n=20, seed=1).drop(columns=["timestamp"])
        with pytest.raises(ValueError, match="timestamp"):
            build_features(df)


class TestGetFeatureNames:
    """Tests for feature name listing."""

    def test_with_market_context(self):
        names = get_feature_names(include_market_context=True)
        assert "spy_return_1d" in names
        assert "relative_return_1d" in names

    def test_without_market_context(self):
        names = get_feature_names(include_market_context=False)
        assert "spy_return_1d" not in names
        assert "relative_return_1d" not in names

    def test_count(self):
        names_with = get_feature_names(include_market_context=True)
        names_without = get_feature_names(include_market_context=False)
        assert len(names_with) == len(names_without) + 2
        assert len(names_with) == 20


class TestFeatureColumnOrder:
    """Verify stable column ordering."""

    def test_column_order_is_deterministic(self):
        df = make_long_ohlcv_df(symbols=("AAPL",), n=30, seed=1)
        features1 = build_features(df)
        features2 = build_features(df)

        assert list(features1.columns) == list(features2.columns)

    def test_column_order_matches_expected(self):
        df = make_long_ohlcv_df(symbols=("AAPL",), n=30, seed=1)
        features = build_features(df)
        expected = ["timestamp", "symbol"] + ALL_FEATURE_NAMES
        assert list(features.columns) == expected
