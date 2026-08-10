"""Tests for data-quality assessment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.quality import assess_quality, DataQualityReport
from tests.analysis.fixtures import make_candle_df


def bdays(start, end):
    return pd.bdate_range(start=start, end=end)


def test_clean_data_reports_no_issues():
    df = make_candle_df(n=100, seed=1)
    report = assess_quality(df, symbol="AAPL", trading_calendar=bdays)
    assert isinstance(report, DataQualityReport)
    assert report.symbol == "AAPL"
    assert report.n_observations == 100
    assert report.start_date == df.index.min()
    assert report.end_date == df.index.max()
    assert report.n_missing_values == 0
    assert report.n_duplicate_timestamps == 0
    assert report.n_missing_trading_days == 0
    assert report.n_zero_volume_days == 0
    assert report.n_ohlc_violations == 0
    assert report.n_extreme_moves == 0
    assert report.n_anomalies == 0


def test_duplicate_timestamps_detected():
    df = make_candle_df(n=30, seed=2)
    df2 = pd.concat([df, df.iloc[[10]]]).sort_index()
    report = assess_quality(df2, trading_calendar=bdays)
    assert report.n_duplicate_timestamps == 1
    assert report.n_observations == len(df2)


def test_missing_values_detected():
    df = make_candle_df(n=30, seed=2)
    df.iloc[20, df.columns.get_loc("close")] = np.nan
    report = assess_quality(df, trading_calendar=bdays)
    assert report.n_missing_values == 1


def test_missing_trading_day_detected():
    df = make_candle_df(n=50, seed=2)
    drop_day = df.index[30]
    df2 = df.drop(index=drop_day)
    report = assess_quality(df2, trading_calendar=bdays)
    assert report.n_missing_trading_days == 1
    # missing_days contains naive timestamps, convert drop_day for comparison
    assert drop_day.tz_localize(None) in report.missing_days


def test_zero_volume_flagged():
    df = make_candle_df(n=20, seed=3)
    df.iloc[5, df.columns.get_loc("volume")] = 0
    report = assess_quality(df, trading_calendar=bdays)
    assert report.n_zero_volume_days == 1
    assert report.n_anomalies == 1


def test_ohlc_violation_flagged():
    df = make_candle_df(n=20, seed=3)
    df.iloc[6, df.columns.get_loc("high")] = 0.0  # high < low -> violation
    report = assess_quality(df, trading_calendar=bdays)
    assert report.n_ohlc_violations == 1
    assert report.n_anomalies == 1


def test_extreme_move_flagged():
    df = make_candle_df(n=30, seed=4)
    # Create a +50% log close-to-close move at index 10 (well beyond 20% threshold)
    # This will also affect index 11's log return (the reversal)
    original_close_10 = df.iloc[10]["close"]
    df.iloc[10, df.columns.get_loc("close")] = original_close_10 * 1.5
    report = assess_quality(df, trading_calendar=bdays)
    # The 50% move and its reversal both exceed threshold -> 2 extreme moves
    assert report.n_extreme_moves == 2
    assert any("extreme move" in r for r in report.anomalies["reason"])


def test_empty_dataframe_report():
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    report = assess_quality(df, symbol="EMPTY")
    assert report.n_observations == 0
    assert report.start_date is None
    assert report.end_date is None
    assert report.n_missing_trading_days == 0


def test_summary_contains_fields():
    df = make_candle_df(n=40, seed=5)
    summary = assess_quality(df, symbol="AAPL", trading_calendar=bdays).summary()
    assert "AAPL" in summary
    assert "Observations: 40" in summary
