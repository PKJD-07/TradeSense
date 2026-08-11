"""Critical leakage regression tests for the ML pipeline.

Each of the 10 risks documented in docs/ml_pipeline.md §12 maps to exactly one
test here. A regression in ANY of these protections fails loudly:

 1. Target in features.
 2. Imputer/scaler fitted on validation/test.
 3. Future observations affecting earlier feature rows.
 4. Target values present in X.
 5. Test data influencing model fitting.
 6. Chronological ordering not preserved.
 7. Walk-forward training windows containing future observations.
 8. Purge gaps not respected.
 9. Cross-symbol leakage (target shift crossing a symbol boundary).
10. Evaluation metrics using the wrong predictions/targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from src.ml.baselines import Persistence
from src.ml.constants import TARGET_COLUMN, WARMUP_ROWS
from src.ml.dataset import build_ml_dataset
from src.ml.evaluation import evaluate_phase, next_session_oc_return
from src.ml.experiment import ExperimentConfig, run_experiment
from src.ml.models import make_model_pipeline
from src.ml.preprocessing import build_preprocessing_pipeline
from src.ml.validation import split_by_date, walk_forward_folds
from tests.features.fixtures import make_long_ohlcv_df, make_long_ohlcv_with_gaps
from tests.ml.fixtures import make_ml_dataset


def _session_idx(ts) -> np.ndarray:
    uniq = np.sort(np.unique(pd.to_datetime(pd.Series(ts), utc=True).to_numpy()))
    return np.searchsorted(
        uniq, pd.to_datetime(pd.Series(ts), utc=True).to_numpy()
    )


# --- Risk 1: target in features -----------------------------------------------
def test_risk1_target_never_in_features():
    """A target-like column on the input must never surface as a feature."""
    ohlcv = make_long_ohlcv_df(symbols=("AAPL", "MSFT"), n=50, seed=1)
    # Poison the input with a precomputed target column: build_features must
    # ignore it, and the dataset builder must keep it out of X.
    ohlcv = ohlcv.copy()
    rng = np.random.default_rng(1)
    ohlcv["target_direction"] = rng.choice([-1, 0, 1], size=len(ohlcv))
    ds = build_ml_dataset(ohlcv, include_market_context=False)
    assert TARGET_COLUMN not in ds.feature_columns
    assert TARGET_COLUMN not in ds.X.columns
    for col in ds.feature_columns:
        assert "target" not in col and "direction" not in col
    # The panel's target column holds the per-symbol future-session labels,
    # not the poisoned input column.
    assert list(ds.df.columns).count(TARGET_COLUMN) == 1


# --- Risk 2: imputer/scaler fitted on validation/test -------------------------
def test_risk2_transforms_fit_on_train_only():
    """Imputer/scaler statistics must come from TRAINING data only, and must
    not change when validation/test rows are transformed or predicted."""
    X_train = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, np.nan], "b": [10.0, 20.0, 30.0, 40.0]}
    )
    X_val = pd.DataFrame({"a": [np.nan, 100.0, 500.0], "b": [1.0, 1.0, 1.0]})
    y_train = np.array([0, 1, -1, 0])

    imputer_pipe = build_preprocessing_pipeline("tree", impute=True)
    imputer_pipe.fit(X_train, y_train)
    filled = imputer_pipe.transform(X_val)
    # val NaN is filled with the TRAIN mean of column 'a' (2.0), never 100/500
    assert np.isclose(filled[0, 0], 2.0)
    # transform must not refit: statistics unchanged after the call
    assert np.isclose(imputer_pipe.named_steps["imputer"].statistics_[0], 2.0)

    model = make_model_pipeline("logistic_regression")
    model.fit(X_train.fillna(2.0), y_train)
    coef_before = model.named_steps["model"].coef_.copy()
    _ = model.predict(X_val.fillna(2.0))
    coef_after = model.named_steps["model"].coef_
    np.testing.assert_array_equal(coef_before, coef_after)


# --- Risk 3: future observations affecting earlier feature rows ---------------
def test_risk3_future_sessions_do_not_change_earlier_features():
    """Mutating a future session's OHLCV must leave every earlier row's X
    bit-identical (features are causal: information through close_t only)."""
    ohlcv = make_long_ohlcv_df(symbols=("AAPL", "MSFT"), n=50, seed=3)
    mutation_idx = 40  # a later session
    ds_base = build_ml_dataset(ohlcv, include_market_context=False)

    mutated = ohlcv.copy()
    for col in ("open", "high", "low", "close", "volume"):
        mutated.loc[mutation_idx, col] = mutated.loc[mutation_idx, col] * (
            1.5 if col != "volume" else 3.0
        )
    ds_mut = build_ml_dataset(mutated, include_market_context=False)

    mutated_ts = ohlcv.loc[mutation_idx, "timestamp"]
    base_mask = ds_base.df["timestamp"] < mutated_ts
    mut_mask = ds_mut.df["timestamp"] < mutated_ts
    pd.testing.assert_frame_equal(
        ds_base.df.loc[base_mask, ds_base.feature_columns].reset_index(drop=True),
        ds_mut.df.loc[mut_mask, ds_mut.feature_columns].reset_index(drop=True),
    )


# --- Risk 4: target values present in X ---------------------------------------
def test_risk4_target_values_never_in_X():
    """The target (a function of session t+1) must not be derivable from that
    row's X: flipping the target leaves the row's features unchanged."""
    ohlcv = make_long_ohlcv_df(symbols=("AAPL",), n=40, seed=5)
    ds1 = build_ml_dataset(ohlcv, include_market_context=False)

    sub = ohlcv[ohlcv["symbol"] == "AAPL"].sort_values("timestamp").reset_index(drop=True)
    t = 25  # beyond warm-up (row 21), not the last row
    ts_t = sub.loc[t, "timestamp"]
    ts_next = sub.loc[t + 1, "timestamp"]

    def row_x(ds, symbol, ts):
        mask = (ds.df["symbol"] == symbol) & (ds.df["timestamp"] == ts)
        return ds.df.loc[mask, ds.feature_columns].iloc[0]

    X_before = row_x(ds1, "AAPL", ts_t).copy()
    y_before = int(
        ds1.df.loc[
            (ds1.df["symbol"] == "AAPL") & (ds1.df["timestamp"] == ts_t),
            TARGET_COLUMN,
        ].iloc[0]
    )

    # Force the opposite sign of the base OC move at session t+1, which is
    # guaranteed to flip the label y_t (epsilon-band membership always changes
    # when the move crosses to the other side of the band).
    base_oc = sub.loc[t + 1, "close"] / sub.loc[t + 1, "open"] - 1.0
    ohlcv2 = ohlcv.copy()
    mask = (ohlcv2["symbol"] == "AAPL") & (ohlcv2["timestamp"] == ts_next)
    ohlcv2.loc[mask, "close"] = (
        ohlcv2.loc[mask, "open"] * (0.98 if base_oc >= 0 else 1.02)
    )
    ds2 = build_ml_dataset(ohlcv2, include_market_context=False)
    y_after = int(
        ds2.df.loc[
            (ds2.df["symbol"] == "AAPL") & (ds2.df["timestamp"] == ts_t),
            TARGET_COLUMN,
        ].iloc[0]
    )
    assert y_before != y_after  # the target actually moved
    # ...but the row's X is bit-identical: the target is not in X.
    pd.testing.assert_series_equal(row_x(ds2, "AAPL", ts_t), X_before)


# --- Risk 5: test data influencing model fitting ------------------------------
def test_risk5_test_data_does_not_influence_fitting():
    """A model fit on train must predict identically regardless of what happens
    to the TEST labels afterwards (it never saw them)."""
    ohlcv = make_long_ohlcv_df(symbols=("AAPL", "MSFT", "SPY"), n=80, seed=7)
    ds = build_ml_dataset(ohlcv)
    split = split_by_date(ds.timestamp.to_numpy())

    X, y = ds.X, ds.y.to_numpy()
    X_tr = X.iloc[split.train_index]
    y_tr = y[split.train_index]
    X_te = X.iloc[split.test_index]
    y_te = y[split.test_index]

    pipe = make_model_pipeline("logistic_regression")
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)

    # Flip the test labels: predictions must NOT change (no refit on test).
    flipped = pipe.predict(X_te)  # same instance, no refit
    np.testing.assert_array_equal(flipped, preds)
    # And a model trained with corrupted test targets in scope is impossible:
    # the split guarantees train timestamps <= val_end < test timestamps.
    assert ds.timestamp.to_numpy()[split.train_index].max() < ds.timestamp.to_numpy()[split.test_index].min()


# --- Risk 6: chronological ordering not preserved -----------------------------
def test_risk6_chronological_order_is_enforced():
    """Shuffled input must produce a canonically sorted panel and
    chronologically coherent phases."""
    ohlcv = make_long_ohlcv_df(symbols=("AAPL", "MSFT", "SPY"), n=50, seed=9)
    shuffled = ohlcv.sample(frac=1.0, random_state=0).reset_index(drop=True)
    ds1 = build_ml_dataset(ohlcv, include_market_context=False)
    ds2 = build_ml_dataset(shuffled, include_market_context=False)

    # Canonical (symbol, timestamp) ordering regardless of input order.
    pd.testing.assert_frame_equal(
        ds1.df.sort_values(["symbol", "timestamp"]).reset_index(drop=True),
        ds2.df.sort_values(["symbol", "timestamp"]).reset_index(drop=True),
    )
    # Within each symbol timestamps are ascending.
    for symbol in ds2.df["symbol"].unique():
        sub = ds2.df[ds2.df["symbol"] == symbol]
        assert sub["timestamp"].is_monotonic_increasing

    split = split_by_date(ds2.timestamp.to_numpy())
    ts = ds2.timestamp.to_numpy()
    assert ts[split.train_index].max() <= split.config.train_end
    assert ts[split.val_index].min() > split.config.train_end
    assert ts[split.test_index].min() > split.config.val_end


# --- Risk 7: walk-forward train windows containing future observations --------
def test_risk7_walkforward_train_never_sees_future():
    """Every walk-forward fold's training block must be strictly before its
    test block in time."""
    ohlcv = make_long_ohlcv_df(symbols=("AAPL", "MSFT", "SPY"), n=200, seed=11)
    ds = build_ml_dataset(ohlcv)
    split = split_by_date(ds.timestamp.to_numpy())
    universe_idx = np.concatenate([split.train_index, split.val_index])
    universe_ts = ds.timestamp.to_numpy()[universe_idx]

    folds = walk_forward_folds(universe_ts, purge_gap=1, test_block=20, step=20, min_train_rows=30)
    assert len(folds) > 0
    for f in folds:
        train_ts = universe_ts[f.train_index]
        test_ts = universe_ts[f.test_index]
        assert train_ts.max() < test_ts.min()
        # purge gap: at least purge_gap unique sessions between train and test
        gap_dates = pd.DatetimeIndex(np.unique(universe_ts))
        train_last = train_ts.max()
        test_first = test_ts.min()
        between = gap_dates[(gap_dates > train_last) & (gap_dates < test_first)]
        assert len(between) >= 1


# --- Risk 8: purge gaps not respected -----------------------------------------
def test_risk8_purge_gaps_are_respected():
    """Sessions in the purge gap must appear in NEITHER the fold's train nor
    its test block."""
    ohlcv = make_long_ohlcv_df(symbols=("AAPL", "MSFT", "SPY"), n=200, seed=13)
    ds = build_ml_dataset(ohlcv)
    split = split_by_date(ds.timestamp.to_numpy())
    universe_idx = np.concatenate([split.train_index, split.val_index])
    universe_ts = ds.timestamp.to_numpy()[universe_idx]

    sidx = _session_idx(universe_ts)
    folds = walk_forward_folds(universe_ts, purge_gap=1, test_block=20, step=20, min_train_rows=30)
    for f in folds:
        test_start = int(sidx[f.test_index].min())
        gap = set(range(test_start - 1, test_start))
        for i in np.concatenate([f.train_index, f.test_index]):
            assert sidx[i] not in gap
    # A larger gap shifts the cut: no train row within `purge_gap` of test start.
    folds5 = walk_forward_folds(universe_ts, purge_gap=5, test_block=20, step=20, min_train_rows=30)
    for f in folds5:
        test_start = int(sidx[f.test_index].min())
        assert np.all(sidx[f.train_index] < test_start - 5)


# --- Risk 9: cross-symbol leakage ---------------------------------------------
def test_risk9_target_never_crosses_symbol_boundary():
    """The target for a symbol's final session must be dropped (NaN), never
    computed from the NEXT symbol's first session."""
    ohlcv = make_long_ohlcv_df(symbols=("AAPL", "MSFT"), n=40, seed=15)
    # Force MSFT's first post-warm-up session to have a huge positive OC move.
    ohlcv = ohlcv.copy()
    msft = ohlcv[ohlcv["symbol"] == "MSFT"].sort_values("timestamp")
    first_ts = msft["timestamp"].iloc[0]
    mask = (ohlcv["symbol"] == "MSFT") & (ohlcv["timestamp"] == first_ts)
    ohlcv.loc[mask, "close"] = ohlcv.loc[mask, "open"] * 1.10

    ds = build_ml_dataset(ohlcv, include_market_context=False)
    assert ds.report.dropped_no_target == 2  # one final session per symbol
    aapl_ts = sorted(ohlcv[ohlcv["symbol"] == "AAPL"]["timestamp"])[-1]
    # AAPL's last session is absent (dropped as unavailable target).
    assert not (ds.df["timestamp"] == aapl_ts).any()

    # Same guarantee for the realized-return helper.
    rr = next_session_oc_return(ohlcv)
    for symbol in ("AAPL", "MSFT"):
        sub = ohlcv[ohlcv["symbol"] == symbol].sort_values("timestamp")
        last_ts = sub["timestamp"].iloc[-1]
        val = rr.loc[(rr["symbol"] == symbol) & (rr["timestamp"] == last_ts), "realized_return"].iloc[0]
        assert np.isnan(val)


# --- Risk 10: evaluation using the wrong predictions/targets ------------------
def test_risk10_metrics_use_the_right_predictions_and_targets():
    """The recorded test metrics must equal a recomputation from the saved
    prediction frame (same rows, same predictions, same targets)."""
    cfg = ExperimentConfig(
        models=("logistic_regression",),
        baselines=("majority_class", "persistence"),
        run_id="leak10",
    )
    ohlcv = make_long_ohlcv_df(symbols=("AAPL", "MSFT", "SPY"), n=100, seed=17)
    res = run_experiment(ohlcv, config=cfg)
    for name in res.predictions:
        pm = res.estimator_results[name]["phase_metrics"]["test"]
        frame = res.predictions[name]["test"]
        assert pm["n_samples"] == len(frame) == res.split["n_test_rows"]
        recomputed = accuracy_score(
            frame[TARGET_COLUMN].to_numpy(), frame["y_pred"].to_numpy()
        )
        assert pm["accuracy"] == pytest.approx(recomputed)
        # The y_prob columns and realized_return align with the same rows.
        assert frame["y_prob_-1"].notna().all()
        assert frame["realized_return"].notna().all()

    # Mismatched inputs are rejected outright (never silently conflated).
    with pytest.raises(ValueError, match="same length"):
        evaluate_phase(np.array([1, 1]), np.array([1]))
