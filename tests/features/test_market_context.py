"""Tests for market-context (cross-asset) features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.market_context import (
    compute_market_context,
    spy_return_1d,
    relative_return_1d,
    SPY_SYMBOL,
)
from tests.features.fixtures import make_long_ohlcv_df, make_long_ohlcv_with_gaps


class TestComputeMarketContext:
    """Tests for the market-context computation orchestrator."""

    def test_basic_computation(self):
        df = make_long_ohlcv_df(symbols=("AAPL", "SPY"), n=30, seed=1)
        result = compute_market_context(df)

        assert "spy_return_1d" in result.columns
        assert "relative_return_1d" in result.columns
        assert len(result) == 60  # 30 sessions × 2 symbols

    def test_left_join_preserves_stock_observations(self):
        """Stock rows must be preserved even when SPY is missing."""
        df = make_long_ohlcv_with_gaps(
            symbols=("AAPL", "SPY"),
            n=20,
            seed=2,
            drop_timestamps={"SPY": [5, 10, 15]},  # Drop SPY at indices 5, 10, 15
        )

        result = compute_market_context(df)

        # AAPL should have all 20 rows
        aapl_rows = result[result["symbol"] == "AAPL"]
        assert len(aapl_rows) == 20

        # SPY-missing timestamps should have NaN market-context features for AAPL
        aapl_timestamps = aapl_rows["timestamp"].unique()
        spy_df = df[df["symbol"] == "SPY"]
        missing_spy_ts = set(aapl_timestamps) - set(spy_df["timestamp"].unique())

        for ts in missing_spy_ts:
            aapl_at_ts = aapl_rows[aapl_rows["timestamp"] == ts]
            assert aapl_at_ts["spy_return_1d"].isna().all()
            assert aapl_at_ts["relative_return_1d"].isna().all()

    def test_no_spy_in_universe(self):
        """If SPY is not in the data, market-context features are all NaN."""
        df = make_long_ohlcv_df(symbols=("AAPL", "MSFT"), n=20, seed=3)
        result = compute_market_context(df)

        assert result["spy_return_1d"].isna().all()
        assert result["relative_return_1d"].isna().all()


class TestSpyReturn1d:
    """Tests for SPY return aligned to stock timestamps."""

    def test_alignment(self):
        df = make_long_ohlcv_df(symbols=("AAPL", "SPY"), n=20, seed=1)
        spy_ret = spy_return_1d(df)

        # Should have values for both AAPL and SPY
        assert len(spy_ret) == 40

    def test_correct_values(self):
        df = make_long_ohlcv_df(symbols=("AAPL", "SPY"), n=20, seed=2)
        spy_ret = spy_return_1d(df).reset_index()

        # Compute expected SPY return
        spy_data = df[df["symbol"] == SPY_SYMBOL].sort_values("timestamp")
        expected_spy_return = spy_data["close"].pct_change()

        # For each timestamp, both AAPL and SPY rows should have the same spy_return_1d
        for ts in spy_data["timestamp"].iloc[1:]:  # Skip first (NaN)
            aapl_row = spy_ret[(spy_ret["timestamp"] == ts) & (spy_ret["symbol"] == "AAPL")]
            spy_row = spy_ret[(spy_ret["timestamp"] == ts) & (spy_ret["symbol"] == SPY_SYMBOL)]

            expected = expected_spy_return[spy_data["timestamp"] == ts].iloc[0]
            assert aapl_row["spy_return_1d"].iloc[0] == pytest.approx(expected)
            assert spy_row["spy_return_1d"].iloc[0] == pytest.approx(expected)


class TestRelativeReturn1d:
    """Tests for stock return minus SPY return."""

    def test_math_correctness(self):
        df = make_long_ohlcv_df(symbols=("AAPL", "SPY"), n=20, seed=1)
        rel_ret = relative_return_1d(df).reset_index()

        # Compute expected
        for symbol in ["AAPL", "SPY"]:
            if symbol == "SPY":
                # SPY relative to itself should be 0 (or NaN for first)
                continue

            symbol_data = df[df["symbol"] == symbol].sort_values("timestamp")
            stock_return = symbol_data["close"].pct_change()

            spy_data = df[df["symbol"] == SPY_SYMBOL].sort_values("timestamp")
            spy_return = spy_data["close"].pct_change()

            for i, ts in enumerate(symbol_data["timestamp"].iloc[1:], start=1):
                if ts in spy_data["timestamp"].values:
                    spy_idx = spy_data["timestamp"].tolist().index(ts)
                    expected = stock_return.iloc[i] - spy_return.iloc[spy_idx]
                    actual = rel_ret[(rel_ret["timestamp"] == ts) & (rel_ret["symbol"] == symbol)][
                        "relative_return_1d"
                    ].iloc[0]
                    assert actual == pytest.approx(expected)


class TestCrossAssetNoForwardFill:
    """Verify that SPY data is never forward-filled."""

    def test_no_forward_fill(self):
        """If SPY is missing at a timestamp, its value should be NaN, not filled from a prior timestamp."""
        df = make_long_ohlcv_with_gaps(
            symbols=("AAPL", "SPY"),
            n=30,
            seed=5,
            drop_timestamps={"SPY": [10, 15, 20]},
        )

        result = compute_market_context(df)

        # Get the timestamps where SPY was dropped
        spy_data = df[df["symbol"] == SPY_SYMBOL]
        all_ts = df[df["symbol"] == "AAPL"]["timestamp"].unique()
        missing_spy_ts = set(all_ts) - set(spy_data["timestamp"].unique())

        for ts in missing_spy_ts:
            aapl_at_ts = result[(result["timestamp"] == ts) & (result["symbol"] == "AAPL")]
            assert aapl_at_ts["spy_return_1d"].isna().all()
            # Ensure it's not filled from a prior value
            assert not aapl_at_ts["spy_return_1d"].notna().any()
