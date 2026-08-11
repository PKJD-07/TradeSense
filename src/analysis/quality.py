"""Data-quality assessment for OHLCV time series.

Builds on the guarantees enforced by the ingestion pipeline (finite, positive
prices; aware UTC timestamps; deduplication) and reports the quality issues
that remain meaningful at the analysis stage: missing values, duplicate
timestamps, missing trading days, zero/negative volume, OHLC relationship
violations, and extreme daily moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

TradingCalendar = Callable[[datetime, datetime], pd.DatetimeIndex]


def default_trading_calendar(start: datetime, end: datetime) -> pd.DatetimeIndex:
    """US business days (weekends + US federal holidays) between start and end."""
    cbd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    return pd.date_range(start=start, end=end, freq=cbd)


@dataclass
class DataQualityReport:
    """Result of a single-symbol data-quality assessment."""

    symbol: str | None
    n_observations: int
    start_date: datetime | None
    end_date: datetime | None
    n_missing_values: int
    n_duplicate_timestamps: int
    n_missing_trading_days: int
    n_zero_volume_days: int
    n_ohlc_violations: int
    n_extreme_moves: int
    anomalies: pd.DataFrame = field(default_factory=pd.DataFrame)
    missing_days: list = field(default_factory=list)

    @property
    def n_anomalies(self) -> int:
        """Total number of flagged anomalous rows."""
        return len(self.anomalies)

    def summary(self) -> str:
        """Human-readable summary of the report."""
        lines = [
            "Data Quality Report",
            f"  Symbol: {self.symbol or '-'}",
            f"  Observations: {self.n_observations}",
            f"  Date Range: {self.start_date} to {self.end_date}",
            f"  Missing values: {self.n_missing_values}",
            f"  Duplicate timestamps: {self.n_duplicate_timestamps}",
            f"  Missing trading days: {self.n_missing_trading_days}",
            f"  Zero-volume days: {self.n_zero_volume_days}",
            f"  OHLC violations: {self.n_ohlc_violations}",
            f"  Extreme moves: {self.n_extreme_moves}",
            f"  Anomalous rows: {self.n_anomalies}",
        ]
        if self.n_missing_trading_days and self.missing_days:
            shown = ", ".join(str(d.date()) for d in self.missing_days[:5])
            if len(self.missing_days) > 5:
                shown += f" (and {len(self.missing_days) - 5} more)"
            lines.append(f"  Missing days: {shown}")
        return "\n".join(lines)


def assess_quality(
    df: pd.DataFrame,
    symbol: str | None = None,
    trading_calendar: TradingCalendar | None = None,
    extreme_move_threshold: float = 0.20,
) -> DataQualityReport:
    """Assess the quality of a single-symbol OHLCV DataFrame.

    Args:
        df: Wide OHLCV frame from :func:`src.analysis.convert.candles_to_dataframe`.
        symbol: Optional symbol label for the report.
        trading_calendar: Callable (start, end) -> expected DatetimeIndex of
            trading days. Defaults to US business days (weekends + federal
            holidays).
        extreme_move_threshold: Absolute log close-to-close return above which a
            session is flagged as an extreme move (default 0.20 = 20%).

    Returns:
        A :class:`DataQualityReport`.
    """
    if df.empty:
        return DataQualityReport(
            symbol=symbol,
            n_observations=0,
            start_date=None,
            end_date=None,
            n_missing_values=0,
            n_duplicate_timestamps=0,
            n_missing_trading_days=0,
            n_zero_volume_days=0,
            n_ohlc_violations=0,
            n_extreme_moves=0,
        )

    index = pd.DatetimeIndex(df.index)
    if index.tz is None:
        index = index.tz_localize("UTC")

    n_missing_values = int(df.isna().sum().sum())

    dup_mask = index.duplicated(keep="first")
    n_duplicates = int(dup_mask.sum())

    calendar = trading_calendar or default_trading_calendar
    # Get naive datetime range for calendar lookup
    start_naive = index.min().tz_convert("UTC").tz_localize(None)
    end_naive = index.max().tz_convert("UTC").tz_localize(None)
    expected = set(calendar(start_naive, end_naive))
    actual = set(pd.DatetimeIndex(index).tz_convert("UTC").tz_localize(None))
    missing_days = sorted(expected - actual)
    n_missing_days = len(missing_days)

    anomalies: list[dict] = []

    volume = pd.to_numeric(df["volume"], errors="coerce")
    zero_volume = volume <= 0
    n_zero_volume = int(zero_volume.sum())
    for ts in index[zero_volume]:
        anomalies.append({"timestamp": ts, "reason": "non-positive volume"})

    has_ohlc = {"open", "high", "low", "close"}.issubset(df.columns)
    if has_ohlc:
        ohlc_bad = (
            (df["high"] < df[["open", "close"]].max(axis=1))
            | (df["low"] > df[["open", "close"]].min(axis=1))
            | (df["high"] < df["low"])
        )
        n_ohlc = int(ohlc_bad.sum())
        for ts in index[ohlc_bad]:
            anomalies.append({"timestamp": ts, "reason": "OHLC relationship violation"})

        log_ret = np.log(df["close"]).diff().dropna()
        extreme = log_ret.abs() > extreme_move_threshold
        n_extreme = int(extreme.sum())
        for ts, val in zip(log_ret.index[extreme], log_ret.to_numpy()[extreme]):
            anomalies.append(
                {"timestamp": ts, "reason": f"extreme move (|log return|={val:.3f})"}
            )
    else:
        n_ohlc = 0
        n_extreme = 0

    anomaly_df = pd.DataFrame(anomalies, columns=["timestamp", "reason"])

    return DataQualityReport(
        symbol=symbol,
        n_observations=len(df),
        start_date=index.min(),
        end_date=index.max(),
        n_missing_values=n_missing_values,
        n_duplicate_timestamps=n_duplicates,
        n_missing_trading_days=n_missing_days,
        n_zero_volume_days=n_zero_volume,
        n_ohlc_violations=n_ohlc,
        n_extreme_moves=n_extreme,
        anomalies=anomaly_df,
        missing_days=missing_days,
    )
