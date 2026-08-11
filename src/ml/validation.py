"""Temporal validation: date-based split and expanding walk-forward.

The pooled multi-symbol panel cannot use row-count splits (symbols have unequal
histories), so splitting is **date-based** (docs/ml_pipeline.md, §4):

    - ``split_by_date`` draws boundaries at the 70th/85th percentiles of the
      panel's sorted UNIQUE session dates. Both boundaries are materialized
      into a :class:`SplitConfig` and recorded, so appending new data never
      silently redraws them. Assignment is identical for every symbol:
          train = {timestamp <= train_end}
          val   = {train_end < timestamp <= val_end}
          test  = {timestamp > val_end}
    - ``walk_forward_folds`` runs over ``train ∪ val`` ONLY (everything
      ``<= val_end``); the untouched test region is excluded from every fold.
      Expanding training window, minimum warm start ``min_train_rows``, test
      block ``test_block`` sessions, non-overlapping step ``step``, and a
      ``purge_gap`` of sessions dropped between each train block end and test
      block start. Folds are a deterministic function of the canonical sorted
      panel and config.

Percentile convention (deterministic): with ``n`` sorted unique dates, the
``q``-quantile date is ``dates[floor(q * (n - 1))]``. This is documented so
boundaries can be reproduced exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

#: Default split boundaries (70th / 85th percentiles), per the locked design.
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VAL_FRACTION = 0.15

#: Default walk-forward geometry.
DEFAULT_PURGE_GAP = 1
DEFAULT_TEST_BLOCK = 63
DEFAULT_STEP = 63
DEFAULT_MIN_TRAIN_ROWS = 504


def _session_index(timestamps) -> tuple[np.ndarray, np.ndarray]:
    """Unique sorted session dates and the session index of every row.

    Returns ``(unique_dates, row_session_idx)`` where ``row_session_idx[i]`` is
    the position of row ``i``'s timestamp in ``unique_dates``.
    """
    ts = pd.to_datetime(pd.Series(timestamps), utc=True).to_numpy()
    uniq = np.unique(ts)
    if len(uniq) == 0:
        raise ValueError("timestamps must not be empty")
    session_idx = np.searchsorted(uniq, ts)
    return uniq, session_idx


def _quantile_date(dates: np.ndarray, q: float) -> np.datetime64:
    """The ``q``-quantile of sorted unique dates (floor convention)."""
    pos = int(np.floor(q * (len(dates) - 1)))
    return dates[pos]


@dataclass
class SplitConfig:
    """Materialized date-based split boundaries.

    Attributes:
        train_end: Boundary date; ``train = {timestamp <= train_end}``.
        val_end: Boundary date; ``val = {train_end < timestamp <= val_end}``.
        purge_gap: Execution-boundary buffer (sessions) used by the
            walk-forward between each train block end and test block start.
        train_fraction: Percentile used for ``train_end``.
        val_fraction: Percentile (offset) used for ``val_end``.
    """

    train_end: np.datetime64
    val_end: np.datetime64
    purge_gap: int = DEFAULT_PURGE_GAP
    train_fraction: float = DEFAULT_TRAIN_FRACTION
    val_fraction: float = DEFAULT_VAL_FRACTION


@dataclass
class DateSplit:
    """A date-based three-way split of the pooled panel.

    Attributes:
        config: The materialized boundaries.
        train_index / val_index / test_index: Row indices (into the array
            passed to :func:`split_by_date`) for each phase. The three sets are
            disjoint and cover every row.
    """

    config: SplitConfig
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray


@dataclass
class WalkForwardFold:
    """One expanding walk-forward fold over ``train ∪ val``.

    Attributes:
        fold: 0-based fold index.
        train_start / train_end: First/last session date of the training block.
        test_start / test_end: First/last session date of the test block.
        n_train_sessions: Sessions in the training block.
        n_test_sessions: Sessions in the test block (<= ``test_block``).
        train_index / test_index: Row indices into the universe array passed
            to :func:`walk_forward_folds`. ``purge_gap`` sessions between the
            train block and the test block are excluded from both.
    """

    fold: int
    train_start: np.datetime64
    train_end: np.datetime64
    test_start: np.datetime64
    test_end: np.datetime64
    n_train_sessions: int
    n_test_sessions: int
    train_index: np.ndarray
    test_index: np.ndarray


def split_by_date(
    timestamps,
    *,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    purge_gap: int = DEFAULT_PURGE_GAP,
) -> DateSplit:
    """Date-based pooled-panel split at the 70th/85th percentile dates.

    Args:
        timestamps: Session timestamps of the pooled panel (one per row).
        train_fraction: Percentile for the train boundary.
        val_fraction: Additional percentile for the val boundary
            (so ``val_end`` is at ``train_fraction + val_fraction``).
        purge_gap: Stored in the config for the walk-forward.

    Returns:
        A :class:`DateSplit` with disjoint train/val/test index sets.

    Raises:
        ValueError: If any phase would be empty (panel too short for the
            requested boundaries) or the boundaries are degenerate.
    """
    uniq, session_idx = _session_index(timestamps)
    n = len(uniq)

    train_end = _quantile_date(uniq, train_fraction)
    val_end = _quantile_date(uniq, train_fraction + val_fraction)

    train_pos = int(np.searchsorted(uniq, train_end))
    val_pos = int(np.searchsorted(uniq, val_end))

    # Degenerate-boundary guard: the val boundary must be a strictly later
    # session than the train boundary, and each phase must be non-empty.
    if val_pos <= train_pos:
        raise ValueError(
            "Split boundaries are degenerate: val_end must be strictly after "
            f"train_end (n_unique_sessions={n}, train_fraction="
            f"{train_fraction}, val_fraction={val_fraction})"
        )

    config = SplitConfig(
        train_end=train_end,
        val_end=val_end,
        purge_gap=purge_gap,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )

    train_mask = session_idx <= train_pos
    val_mask = (session_idx > train_pos) & (session_idx <= val_pos)
    test_mask = session_idx > val_pos

    train_index = np.flatnonzero(train_mask)
    val_index = np.flatnonzero(val_mask)
    test_index = np.flatnonzero(test_mask)

    if len(train_index) == 0 or len(val_index) == 0 or len(test_index) == 0:
        raise ValueError(
            "Date-based split produced an empty phase; the panel has too few "
            f"distinct session dates ({n}) for these boundaries"
        )

    return DateSplit(
        config=config,
        train_index=train_index,
        val_index=val_index,
        test_index=test_index,
    )


def walk_forward_folds(
    timestamps,
    *,
    purge_gap: int = DEFAULT_PURGE_GAP,
    test_block: int = DEFAULT_TEST_BLOCK,
    step: int = DEFAULT_STEP,
    min_train_rows: int = DEFAULT_MIN_TRAIN_ROWS,
) -> list[WalkForwardFold]:
    """Expanding walk-forward folds over the provided universe.

    Args:
        timestamps: Session timestamps of the ``train ∪ val`` universe rows
            (the untouched test region must be EXCLUDED before calling).
        purge_gap: Sessions dropped between each train block end and test block
            start (1 by default; conservative execution-boundary buffer).
        test_block: Session length of each test block (63 = one quarter).
        step: Session offset between test blocks (63 = non-overlapping).
        min_train_rows: Minimum training sessions before a fold is emitted.

    Returns:
        A list of :class:`WalkForwardFold` (empty if no fold meets the
        minimum-warm-start requirement). Folds are deterministic.
    """
    if purge_gap < 0:
        raise ValueError("purge_gap must be >= 0")
    if test_block < 1:
        raise ValueError("test_block must be >= 1")
    if step < 1:
        raise ValueError("step must be >= 1")
    if min_train_rows < 0:
        raise ValueError("min_train_rows must be >= 0")

    uniq, session_idx = _session_index(timestamps)
    n_sessions = len(uniq)

    folds: list[WalkForwardFold] = []
    test_start = 0
    fold_id = 0
    while test_start < n_sessions:
        test_end = min(test_start + test_block, n_sessions)
        train_end_idx = test_start - purge_gap

        if train_end_idx >= min_train_rows:
            train_mask = session_idx < train_end_idx
            test_mask = (session_idx >= test_start) & (session_idx < test_end)
            folds.append(
                WalkForwardFold(
                    fold=fold_id,
                    train_start=uniq[0],
                    train_end=uniq[train_end_idx - 1],
                    test_start=uniq[test_start],
                    test_end=uniq[test_end - 1],
                    n_train_sessions=int(train_end_idx),
                    n_test_sessions=int(test_end - test_start),
                    train_index=np.flatnonzero(train_mask),
                    test_index=np.flatnonzero(test_mask),
                )
            )
            fold_id += 1
        test_start += step
    return folds
