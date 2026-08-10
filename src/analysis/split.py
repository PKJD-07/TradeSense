"""Temporal (time-ordered) data splitting utilities.

Financial time series are not independent and identically distributed: returns
are autocorrelated, volatility clusters, and N-period forward targets make
labels overlap across nearby samples. A random split would place "yesterday" in
the test set and "today" in training, leaking the future.

These helpers produce index partitions that respect time ordering and support a
purge ``gap`` between segments to prevent overlapping-label leakage.
"""

from __future__ import annotations

import pandas as pd


def chronological_split(
    index: pd.Index,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    gap: int = 0,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """Split a chronologically sorted index into train/validation/test segments.

    Args:
        index: Chronologically sorted index (e.g. a DatetimeIndex).
        ratios: (train, validation, test) fractions summing to 1.0. A 2-tuple
            is accepted as (train, test).
        gap: Number of rows dropped between segments to purge overlapping
            labels / rolling-feature windows.

    Returns:
        (train_idx, validation_idx, test_idx) as contiguous slices preserving
        time order. Validation is empty when a 2-tuple of ratios is given.
    """
    idx = pd.Index(index)
    n = len(idx)
    if n < 1:
        raise ValueError("index is empty")

    ratios = list(ratios)
    if len(ratios) == 2:
        ratios = [ratios[0], 0.0, ratios[1]]
    if len(ratios) != 3:
        raise ValueError("ratios must be a 2- or 3-tuple")
    if any(r < 0 for r in ratios):
        raise ValueError("ratios must be non-negative")
    if not abs(sum(ratios) - 1.0) < 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")
    if gap < 0:
        raise ValueError("gap must be >= 0")

    train_frac, val_frac, test_frac = ratios
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    if n_val > 0:
        n_train = n - n_test - n_val - 2 * gap
        if n_train < 1:
            raise ValueError("not enough rows for the requested split and gap")
        train = idx[:n_train]
        val_start = n_train + gap
        val = idx[val_start : val_start + n_val]
        test = idx[val_start + n_val + gap :]
    else:
        n_train = n - n_test - gap
        if n_train < 1:
            raise ValueError("not enough rows for the requested split and gap")
        train = idx[:n_train]
        val = idx[:0]
        test = idx[n_train + gap :]

    return train, val, test


def walk_forward_windows(
    index: pd.Index,
    train_size: int,
    test_size: int,
    step: int | None = None,
    gap: int = 0,
    expanding: bool = False,
) -> list[tuple[pd.Index, pd.Index]]:
    """Generate rolling-origin (walk-forward) (train, test) index pairs.

    Args:
        index: Chronologically sorted index.
        train_size: Number of sessions in each training block.
        test_size: Number of sessions in each test block.
        step: Advance between folds (defaults to test_size).
        gap: Rows dropped between train and test to purge label overlap.
        expanding: If True, training uses all data from the start up to the
            block end (grows with each fold) instead of a fixed-size window.

    Returns:
        List of (train_idx, test_idx) tuples in chronological order.
    """
    idx = pd.Index(index)
    n = len(idx)
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size and test_size must be >= 1")
    if gap < 0:
        raise ValueError("gap must be >= 0")
    step = step if step is not None and step > 0 else test_size

    windows = []
    start = 0
    while start + train_size + gap + test_size <= n:
        train_end = start + train_size
        train = idx[:train_end] if expanding else idx[start:train_end]
        test_start = train_end + gap
        test = idx[test_start : test_start + test_size]
        windows.append((train, test))
        start += step
    return windows
