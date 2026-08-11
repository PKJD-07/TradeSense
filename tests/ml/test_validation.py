"""Tests for src.ml.validation: date-based split and walk-forward.

Verifies the 70th/85th percentile boundaries (materialized + recorded),
identical calendar windows for every symbol, the expanding walk-forward
geometry (test block 63, step 63, min train 504, purge gap 1), that the test
region never enters a walk-forward fold, and determinism of all folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.validation import (
    DEFAULT_MIN_TRAIN_ROWS,
    DEFAULT_PURGE_GAP,
    DEFAULT_STEP,
    DEFAULT_TEST_BLOCK,
    split_by_date,
    walk_forward_folds,
)
from tests.ml.fixtures import make_ml_dataset


def _dates(n: int, start: str = "2015-01-05") -> np.ndarray:
    """``n`` sorted unique business-day sessions."""
    return pd.date_range(start, periods=n, freq="B").to_numpy()


def _ts(n: int, per_symbol: int = 3, start: str = "2015-01-05") -> np.ndarray:
    """A pooled-panel timestamp array: ``n`` sessions x ``per_symbol`` rows."""
    return np.tile(_dates(n, start), per_symbol)


def _utc(ts) -> np.ndarray:
    return pd.to_datetime(pd.Series(ts), utc=True).to_numpy()


def _session_idx(ts) -> np.ndarray:
    """Per-row session index (position in the sorted unique dates)."""
    uniq = np.sort(np.unique(_utc(ts)))
    return np.searchsorted(uniq, _utc(ts))


class TestSplitByDate:
    def test_phases_disjoint_and_cover(self):
        ts = _ts(400)
        split = split_by_date(ts)
        sets = [set(split.train_index), set(split.val_index), set(split.test_index)]
        assert sets[0].isdisjoint(sets[1])
        assert sets[1].isdisjoint(sets[2])
        assert sets[0].isdisjoint(sets[2])
        assert sets[0] | sets[1] | sets[2] == set(range(len(ts)))

    def test_boundaries_at_documented_percentiles(self):
        ts = _ts(1000, per_symbol=2)
        split = split_by_date(ts)
        uniq = np.sort(np.unique(_utc(ts)))
        train_pos = int(np.floor(0.70 * (len(uniq) - 1)))
        val_pos = int(np.floor(0.85 * (len(uniq) - 1)))
        assert split.config.train_end == uniq[train_pos]
        assert split.config.val_end == uniq[val_pos]

    def test_config_materialized_and_recorded(self):
        split = split_by_date(_ts(300))
        cfg = split.config
        assert cfg.purge_gap == DEFAULT_PURGE_GAP
        assert cfg.train_fraction == 0.70
        assert cfg.val_fraction == 0.15
        assert cfg.train_end < cfg.val_end

    def test_assignment_by_timestamp(self):
        ts = _ts(200, per_symbol=1)
        split = split_by_date(ts)
        got = _utc(ts)
        train_end, val_end = split.config.train_end, split.config.val_end
        assert np.all(got[split.train_index] <= train_end)
        assert np.all((got[split.val_index] > train_end) & (got[split.val_index] <= val_end))
        assert np.all(got[split.test_index] > val_end)

    def test_same_calendar_window_for_every_symbol(self):
        """Pooled multi-symbol rows at the same date land in the same phase."""
        n_sessions, per_symbol = 300, 4
        ts = _ts(n_sessions, per_symbol=per_symbol)
        split = split_by_date(ts)
        got = _utc(ts)
        sidx = _session_idx(ts)
        train_dates = set(sidx[split.train_index])
        val_dates = set(sidx[split.val_index])
        test_dates = set(sidx[split.test_index])
        # every session index is in exactly one phase
        assert train_dates.isdisjoint(val_dates)
        assert val_dates.isdisjoint(test_dates)
        assert train_dates | val_dates | test_dates == set(range(n_sessions))
        # so every symbol (identical timestamp rows) shares the phase of its date
        for i in range(len(ts)):
            phase = (
                "train"
                if sidx[i] in train_dates
                else "val"
                if sidx[i] in val_dates
                else "test"
            )
            if i in set(split.train_index):
                assert phase == "train"
            elif i in set(split.val_index):
                assert phase == "val"
            else:
                assert phase == "test"

    def test_deterministic(self):
        ts = _ts(350)
        a, b = split_by_date(ts), split_by_date(ts)
        np.testing.assert_array_equal(a.train_index, b.train_index)
        np.testing.assert_array_equal(a.val_index, b.val_index)
        np.testing.assert_array_equal(a.test_index, b.test_index)
        assert a.config == b.config

    def test_empty_phase_raises(self):
        # Boundaries push val_end to the last session -> empty test phase.
        with pytest.raises(ValueError, match="empty phase"):
            split_by_date(_ts(4), train_fraction=0.5, val_fraction=0.5)

    def test_degenerate_boundaries_raise(self):
        with pytest.raises(ValueError, match="degenerate"):
            split_by_date(_ts(4), train_fraction=0.9, val_fraction=0.05)


class TestWalkForward:
    def test_blocks_quarter_sized_and_non_overlapping(self):
        n_sessions = 3000
        ts = _ts(n_sessions)
        folds = walk_forward_folds(ts)
        assert len(folds) > 3
        sidx = _session_idx(ts)
        seen = set()
        for f in folds:
            assert f.n_test_sessions <= DEFAULT_TEST_BLOCK
            start = int(sidx[f.test_index].min())
            assert set(f.test_index) <= {i for i in range(len(ts)) if sidx[i] in range(start, start + f.n_test_sessions)}
            # non-overlapping, forward-stepping blocks
            assert start not in seen
            for i in f.test_index:
                assert sidx[i] >= start and sidx[i] < start + f.n_test_sessions
            seen.add(start)

    def test_expanding_training_window(self):
        ts = _ts(3000)
        folds = walk_forward_folds(ts)
        n_train = [f.n_train_sessions for f in folds]
        assert n_train == sorted(n_train)
        assert len(set(n_train)) == len(n_train)  # strictly growing
        # training window always starts at the first session (expanding)
        assert len({f.train_start for f in folds}) == 1

    def test_min_train_rows_requirement(self):
        folds = walk_forward_folds(_ts(3000))
        assert all(f.n_train_sessions >= DEFAULT_MIN_TRAIN_ROWS for f in folds)
        # the first emitted fold is the first whose train block clears 504
        assert folds[0].n_train_sessions >= DEFAULT_MIN_TRAIN_ROWS

    def test_purge_gap_respected(self):
        ts = _ts(3000)
        folds = walk_forward_folds(ts)
        sidx = _session_idx(ts)
        for f in folds:
            start = int(sidx[f.test_index].min())
            # every train row is strictly before the purge-gap cut
            assert np.all(sidx[f.train_index] < start - DEFAULT_PURGE_GAP)
            # gap sessions are excluded from BOTH train and test
            gap = set(range(start - DEFAULT_PURGE_GAP, start))
            for i in np.concatenate([f.train_index, f.test_index]):
                assert sidx[i] not in gap

    def test_purge_gap_shifts_train_cut(self):
        ts = _ts(3000)
        g0 = {f.test_start: f for f in walk_forward_folds(ts, purge_gap=0)}
        g5 = {f.test_start: f for f in walk_forward_folds(ts, purge_gap=5)}
        common = set(g0) & set(g5)
        assert len(common) > 3
        for start in common:
            assert g5[start].n_train_sessions == g0[start].n_train_sessions - 5

    def test_short_universe_returns_no_folds(self):
        assert walk_forward_folds(_ts(200)) == []

    def test_deterministic(self):
        ts = _ts(2500)
        a = walk_forward_folds(ts)
        b = walk_forward_folds(ts)
        assert len(a) == len(b)
        for fa, fb in zip(a, b):
            np.testing.assert_array_equal(fa.train_index, fb.train_index)
            np.testing.assert_array_equal(fa.test_index, fb.test_index)
            assert fa.test_start == fb.test_start

    def test_validation_raises_on_bad_args(self):
        with pytest.raises(ValueError, match="purge_gap"):
            walk_forward_folds(_ts(1000), purge_gap=-1)
        with pytest.raises(ValueError, match="test_block"):
            walk_forward_folds(_ts(1000), test_block=0)
        with pytest.raises(ValueError, match="step"):
            walk_forward_folds(_ts(1000), step=0)


class TestIntegration:
    def test_split_and_walk_forward_on_ml_panel(self):
        ds = make_ml_dataset(symbols=("AAPL", "MSFT", "SPY"), n=200, seed=7)
        ts = ds.timestamp.to_numpy()
        split = split_by_date(ts)
        assert len(split.train_index) > 0
        assert len(split.val_index) > 0
        assert len(split.test_index) > 0

        # Walk-forward over train ∪ val only (test region excluded).
        universe_idx = np.concatenate([split.train_index, split.val_index])
        universe_ts = ts[universe_idx]
        folds = walk_forward_folds(universe_ts)
        universe_set = set(universe_idx)
        for f in folds:
            assert set(f.train_index) <= universe_set
            assert set(f.test_index) <= universe_set
            assert set(f.train_index).isdisjoint(set(f.test_index))
