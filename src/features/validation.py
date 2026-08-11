"""
Feature-level validation utilities.

Validates that feature matrices satisfy causal constraints and data quality
requirements without using future information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    max_lookback: int = 20,
) -> dict:
    """Validate a feature matrix for causal correctness and data quality.

    Args:
        df: Feature DataFrame with timestamp index or (timestamp, symbol) multiindex
        feature_columns: List of feature column names to validate
        max_lookback: Maximum lookback period for warm-up calculation

    Returns:
        Dictionary with validation results:
        - valid: bool
        - nan_counts: dict mapping feature -> NaN count
        - inf_counts: dict mapping feature -> inf count
        - warmup_rows: int, rows that may have NaN due to lookback
        - issues: list of problem descriptions
    """
    issues = []
    nan_counts = {}
    inf_counts = {}

    for col in feature_columns:
        if col not in df.columns:
            issues.append(f"Missing feature column: {col}")
            continue

        series = df[col]
        nan_counts[col] = int(series.isna().sum())
        inf_count = int((series == np.inf).sum() + (series == -np.inf).sum())
        inf_counts[col] = inf_count

        if inf_count > 0:
            issues.append(f"Feature '{col}' contains {inf_count} infinite values")

    # Check for NaN in the "stable" region (after warm-up)
    n_rows = len(df)
    warmup_rows = min(max_lookback + 1, n_rows)

    valid = len(issues) == 0

    return {
        "valid": valid,
        "nan_counts": nan_counts,
        "inf_counts": inf_counts,
        "warmup_rows": warmup_rows,
        "issues": issues,
    }


def check_no_target_leakage(feature_columns: list[str], target_keywords: list[str] | None = None) -> list[str]:
    """Check that no target-related columns are in feature list.

    Args:
        feature_columns: List of feature column names
        target_keywords: Keywords that indicate target columns

    Returns:
        List of warnings for suspicious column names
    """
    if target_keywords is None:
        target_keywords = ["target", "label", "y_", "direction", "forward_return", "realized_vol"]

    warnings = []
    for col in feature_columns:
        col_lower = col.lower()
        for keyword in target_keywords:
            if keyword.lower() in col_lower:
                warnings.append(f"Feature '{col}' may be a target column (contains '{keyword}')")
                break

    return warnings


def check_minimum_data(df: pd.DataFrame, min_rows: int = 25) -> dict:
    """Check that dataset has sufficient rows for feature computation.

    Args:
        df: Input OHLCV DataFrame
        min_rows: Minimum rows required (default 25 for 20-day lookback + buffer)

    Returns:
        Dictionary with:
        - sufficient: bool
        - n_rows: int
        - min_required: int
    """
    n_rows = len(df)
    sufficient = n_rows >= min_rows

    return {
        "sufficient": sufficient,
        "n_rows": n_rows,
        "min_required": min_rows,
    }
