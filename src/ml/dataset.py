"""ML dataset construction: align causal features with the target.

Builds a single long-form ML panel from OHLCV + the causal feature matrix, so
that downstream modules all consume the same row set. One row is one
``(symbol, timestamp=t)`` session.

Alignment invariants (each is enforced by a regression test):
    - ``X_t`` uses only information through ``close_t``.
    - ``y_t = f(session t+1 only)`` (next-session direction).
    - ``timestamp`` and ``symbol`` remain traceable on every surviving row.
    - The target is never included as a feature.
    - The neutral class (``0``) is retained: the target stays ``{-1, 0, +1}``.

CRITICAL: targets are computed PER SYMBOL. The target functions use
``shift(-1)`` internally; applied to a long-form multi-symbol frame they would
cross symbol boundaries at the last row of each symbol. This module is the only
place that computes targets for the ML pipeline, and it never lets a ``shift``
cross a symbol boundary.

Row-drop policy (explicit, recorded in ``DatasetReport``):
    1. Warm-up rows: the first ``warmup_rows`` (default 21) of every symbol are
       dropped (structural feature lookback, fixed and causal).
    2. Unavailable targets: rows with no future session (the final session of
       each symbol) are dropped.
    3. Remaining feature NaNs: rows with any NaN feature are dropped, with a
       guard that fails loudly if any feature's post-warm-up NaN fraction
       exceeds ``max_nan_fraction``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analysis.targets import next_session_direction
from src.features.builder import build_features, get_feature_names
from src.features.validation import check_no_target_leakage
from src.ml.constants import (
    DEFAULT_EPSILON,
    DEFAULT_MAX_NAN_FRACTION,
    TARGET_COLUMN,
    WARMUP_ROWS,
)


@dataclass
class DatasetReport:
    """Metadata about dataset construction and every row that was dropped.

    Attributes:
        symbols: Symbols present in the final panel.
        date_range_by_symbol: (min, max) timestamp per symbol in the final panel.
        feature_names: The feature column names.
        total_input_rows: Number of OHLCV rows passed in.
        warmup_rows: Warm-up length used (rows dropped per symbol).
        rows_before_drops: Number of panel rows (features + target merged)
            before any drop.
        dropped_warmup: Rows dropped as structural warm-up.
        dropped_no_target: Rows dropped because the target was unavailable
            (no future session). Broken out by symbol.
        dropped_nan_features: Rows dropped because a feature was NaN after
            warm-up. Broken out by symbol.
        final_rows: Rows in the final panel.
        feature_nan_counts: Post-warm-up NaN count per feature (pre-drop).
        class_counts: Target class distribution {-1, 0, +1} in the final panel.
        max_nan_fraction: The NaN-fraction guard used.
        epsilon: Target epsilon used.
    """

    symbols: list[str]
    date_range_by_symbol: dict[str, tuple[pd.Timestamp, pd.Timestamp]]
    feature_names: list[str]
    total_input_rows: int
    warmup_rows: int
    rows_before_drops: int
    dropped_warmup: int
    dropped_no_target: int
    dropped_nan_features: int
    final_rows: int
    feature_nan_counts: dict[str, int]
    class_counts: dict[int, int]
    max_nan_fraction: float
    epsilon: float
    dropped_no_target_by_symbol: dict[str, int] = field(default_factory=dict)
    dropped_nan_features_by_symbol: dict[str, int] = field(default_factory=dict)


@dataclass
class MLDataset:
    """A single ML-ready panel with explicit feature/target/metadata separation.

    Attributes:
        df: Long-form panel sorted by (symbol, timestamp) with columns
            [timestamp, symbol, *feature_columns, target_column].
        feature_columns: The causal feature names (the X columns).
        target_column: The target column name.
        metadata_columns: Traceability columns ["timestamp", "symbol"].
        report: DatasetReport describing construction and dropped rows.
    """

    df: pd.DataFrame
    feature_columns: list[str]
    target_column: str = TARGET_COLUMN
    metadata_columns: list[str] = field(default_factory=lambda: ["timestamp", "symbol"])
    report: DatasetReport = None  # type: ignore[assignment]

    @property
    def X(self) -> pd.DataFrame:
        """Feature matrix (causal features only; target never included)."""
        return self.df[self.feature_columns]

    @property
    def y(self) -> pd.Series:
        """Target series (int labels {-1, 0, +1})."""
        return self.df[self.target_column]

    @property
    def meta(self) -> pd.DataFrame:
        """Traceability columns (timestamp, symbol)."""
        return self.df[self.metadata_columns]

    @property
    def symbols(self) -> list[str]:
        return list(self.report.symbols)

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def timestamp(self) -> pd.Series:
        return self.df["timestamp"]

    @property
    def symbol(self) -> pd.Series:
        return self.df["symbol"]


def _warmup_mask(df: pd.DataFrame, warmup_rows: int) -> pd.Series:
    """Boolean mask marking the first ``warmup_rows`` rows of each symbol.

    ``df`` must be sorted by (symbol, timestamp).
    """
    mask = pd.Series(False, index=df.index)
    for symbol in df["symbol"].unique():
        symbol_idx = df.index[df["symbol"] == symbol]
        mask.loc[symbol_idx[:warmup_rows]] = True
    return mask


def _count_by_symbol(df: pd.DataFrame, mask: pd.Series) -> dict[str, int]:
    """Count masked rows per symbol, symbols sorted."""
    counts = {}
    for symbol in sorted(df.loc[mask, "symbol"].unique()):
        counts[symbol] = int(mask[df["symbol"] == symbol].sum())
    return counts


def build_ml_dataset(
    long_ohlcv: pd.DataFrame,
    epsilon: float = DEFAULT_EPSILON,
    include_market_context: bool = True,
    warmup_rows: int = WARMUP_ROWS,
    max_nan_fraction: float = DEFAULT_MAX_NAN_FRACTION,
) -> MLDataset:
    """Build the ML panel from long-form OHLCV data.

    Args:
        long_ohlcv: DataFrame with columns
            [symbol, timestamp, open, high, low, close, volume].
        epsilon: Target epsilon (neutral-zone half-width).
        include_market_context: Whether to include the SPY market-context
            features.
        warmup_rows: Structural warm-up rows dropped per symbol.
        max_nan_fraction: Guard on post-warm-up per-feature NaN fraction.

    Returns:
        MLDataset with a canonical (symbol, timestamp)-sorted panel.

    Raises:
        ValueError: If required columns are missing, no rows survive, or a
            feature's post-warm-up NaN fraction exceeds ``max_nan_fraction``.
    """
    if "symbol" not in long_ohlcv.columns or "timestamp" not in long_ohlcv.columns:
        raise ValueError("Input DataFrame must have 'symbol' and 'timestamp' columns")
    if len(long_ohlcv) == 0:
        raise ValueError("Input DataFrame is empty")
    if warmup_rows < 0:
        raise ValueError("warmup_rows must be >= 0")
    if not 0.0 <= max_nan_fraction <= 1.0:
        raise ValueError("max_nan_fraction must be in [0, 1]")

    feature_names = get_feature_names(include_market_context=include_market_context)
    total_input_rows = int(len(long_ohlcv))

    # 1. Features (causal, computed per symbol by the feature layer).
    features = build_features(long_ohlcv, include_market_context=include_market_context)
    features = features[["timestamp", "symbol"] + feature_names].copy()

    # 2. Targets computed PER SYMBOL. A pooled shift(-1) would cross symbol
    #    boundaries; per-symbol computation is mandatory.
    target_frames = []
    for symbol, group in long_ohlcv.groupby("symbol"):
        symbol_data = group.sort_values("timestamp").set_index("timestamp")
        target = next_session_direction(symbol_data, epsilon=epsilon).rename(
            TARGET_COLUMN
        )
        target = target.reset_index()  # columns: timestamp, target
        target["symbol"] = symbol
        target_frames.append(target)
    target_df = (
        pd.concat(target_frames, ignore_index=True)
        if target_frames
        else pd.DataFrame(columns=["timestamp", TARGET_COLUMN, "symbol"])
    )

    # 3. Merge features with targets on (symbol, timestamp). LEFT join preserves
    #    every feature row; unavailable targets are handled explicitly below.
    panel = features.merge(target_df, on=["symbol", "timestamp"], how="left")

    # 4. Canonical deterministic ordering.
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    panel[TARGET_COLUMN] = pd.to_numeric(panel[TARGET_COLUMN], errors="coerce")

    rows_before_drops = int(len(panel))

    # 5. Row-drop accounting.
    #    a) Structural warm-up rows.
    warmup = _warmup_mask(panel, warmup_rows)
    remaining = panel.loc[~warmup].copy()

    #    b) Unavailable targets (final session of each symbol).
    no_target = remaining[TARGET_COLUMN].isna()
    dropped_no_target_by_symbol = _count_by_symbol(remaining, no_target)
    remaining = remaining.loc[~no_target].copy()

    #    c) Remaining feature NaNs, with a loud guard.
    nan_counts = {col: int(remaining[col].isna().sum()) for col in feature_names}
    for col in feature_names:
        fraction = nan_counts[col] / len(remaining) if len(remaining) else 0.0
        if fraction > max_nan_fraction:
            raise ValueError(
                f"Feature '{col}' has {fraction:.2%} NaN after warm-up "
                f"(guard max_nan_fraction={max_nan_fraction:.2%}); refusing to "
                "silently drop a material share of the data"
            )

    nan_feature_rows = remaining[feature_names].isna().any(axis=1)
    dropped_nan_features_by_symbol = _count_by_symbol(remaining, nan_feature_rows)
    remaining = remaining.loc[~nan_feature_rows].copy()

    if remaining.empty:
        raise ValueError(
            "No rows survived dataset construction; check warmup_rows, data "
            "availability, and the NaN guard"
        )

    # 6. Final panel: int labels, canonical column order.
    remaining[TARGET_COLUMN] = remaining[TARGET_COLUMN].astype(int)
    remaining = remaining[
        ["timestamp", "symbol"] + feature_names + [TARGET_COLUMN]
    ].reset_index(drop=True)

    # 7. Sanity gates. The feature names legitimately contain "y_" (e.g.
    #    intraday_return, volatility_10d, spy_return_1d), so the generic
    #    "y_" keyword heuristic is too aggressive; we scope it to actual
    #    target-ish names plus the structural exclusion of the target column.
    if TARGET_COLUMN in feature_names:
        raise ValueError(f"Target column '{TARGET_COLUMN}' appears in feature columns")
    warnings = check_no_target_leakage(
        feature_names,
        target_keywords=[
            "target",
            "label",
            "forward_return",
            "realized_vol",
            "direction",
        ],
    )
    if warnings:
        raise ValueError(
            "Feature matrix may contain target columns: " + "; ".join(warnings)
        )

    # 8. Report.
    symbols = sorted(remaining["symbol"].unique().tolist())
    date_range_by_symbol: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for symbol in symbols:
        ts = remaining.loc[remaining["symbol"] == symbol, "timestamp"]
        if len(ts):
            date_range_by_symbol[symbol] = (ts.min(), ts.max())

    class_counts = (
        remaining[TARGET_COLUMN].value_counts().sort_index().astype(int).to_dict()
    )

    report = DatasetReport(
        symbols=symbols,
        date_range_by_symbol=date_range_by_symbol,
        feature_names=list(feature_names),
        total_input_rows=total_input_rows,
        warmup_rows=warmup_rows,
        rows_before_drops=rows_before_drops,
        dropped_warmup=int(warmup.sum()),
        dropped_no_target=int(sum(dropped_no_target_by_symbol.values())),
        dropped_nan_features=int(sum(dropped_nan_features_by_symbol.values())),
        final_rows=int(len(remaining)),
        feature_nan_counts=nan_counts,
        class_counts=class_counts,
        max_nan_fraction=max_nan_fraction,
        epsilon=epsilon,
        dropped_no_target_by_symbol=dropped_no_target_by_symbol,
        dropped_nan_features_by_symbol=dropped_nan_features_by_symbol,
    )

    return MLDataset(
        df=remaining,
        feature_columns=list(feature_names),
        target_column=TARGET_COLUMN,
        metadata_columns=["timestamp", "symbol"],
        report=report,
    )
