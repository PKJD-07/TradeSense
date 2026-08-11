"""Tests for temporal (chronological and walk-forward) data splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.split import chronological_split, walk_forward_windows


@pytest.fixture
def index() -> pd.DatetimeIndex:
    return pd.date_range("2021-01-04", periods=100, freq="B")


def test_chronological_split_sizes_and_order(index):
    train, val, test = chronological_split(index, ratios=(0.7, 0.15, 0.15))
    assert len(train) + len(val) + len(test) == 100
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    assert train[-1] < val[0] < test[0]


def test_chronological_split_gap_drops_rows(index):
    gap = 5
    train, val, test = chronological_split(index, ratios=(0.7, 0.15, 0.15), gap=gap)
    assert len(train) + len(val) + len(test) == 100 - 2 * gap
    # The gap rows sit between the segments
    assert index[len(train)] not in train and index[len(train)] not in val
    assert val[0] == index[len(train) + gap]
    assert test[0] == index[len(train) + gap + len(val) + gap]


def test_chronological_split_two_tuple(index):
    train, val, test = chronological_split(index, ratios=(0.8, 0.2))
    assert len(val) == 0
    assert len(train) + len(test) == 100


def test_chronological_split_invalid_ratios(index):
    with pytest.raises(ValueError):
        chronological_split(index, ratios=(0.5, 0.4))  # sums to 0.9
    with pytest.raises(ValueError):
        chronological_split(index, ratios=(0.7, 0.2, 0.1, 0.0))


def test_chronological_split_requires_rows():
    with pytest.raises(ValueError):
        chronological_split(pd.Index([]), ratios=(0.7, 0.15, 0.15))


def test_walk_forward_first_window(index):
    train_size, test_size, step, gap = 40, 10, 10, 5
    windows = walk_forward_windows(index, train_size, test_size, step, gap)
    train, test = windows[0]
    assert len(train) == train_size
    assert len(test) == test_size
    assert train[-1] < test[0]
    # gap honored: test starts after train_end + gap
    assert test[0] == index[train_size + gap]


def test_walk_forward_count_and_non_overlap(index):
    train_size, test_size, step = 40, 10, 10
    windows = walk_forward_windows(index, train_size, test_size, step)
    # starts 0,10,20,30,40,50 -> 6 folds for n=100 (start + train + test <= n)
    assert len(windows) == 6
    for train, test in windows:
        assert len(set(train) & set(test)) == 0
        assert train.is_monotonic_increasing and test.is_monotonic_increasing


def test_walk_forward_expanding(index):
    train_size, test_size, step = 40, 10, 10
    windows = walk_forward_windows(index, train_size, test_size, step, expanding=True)
    assert len(windows) == 6
    assert len(windows[0][0]) == train_size
    # last fold: start=50, train uses idx[:90] (expanding from 0)
    assert len(windows[-1][0]) == train_size + 5 * step
    # expanding train always starts at the beginning of the data
    assert windows[-1][0][0] == index[0]


def test_walk_forward_step_defaults_to_test_size(index):
    windows = walk_forward_windows(index, train_size=40, test_size=10)
    assert len(windows) == 6  # starts 0,10,20,30,40,50


def test_walk_forward_invalid_args(index):
    with pytest.raises(ValueError):
        walk_forward_windows(index, train_size=0, test_size=10)
    with pytest.raises(ValueError):
        walk_forward_windows(index, train_size=40, test_size=10, gap=-1)
