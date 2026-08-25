"""Tests for src.ml.evaluation: per-phase metrics and financial diagnostics.

Verifies classification/probability metrics against hand-computed and direct
sklearn values, the "None when not mathematically defined" contract for ROC-AUC,
per-phase isolation, per-symbol realized-return computation, and JSON
serializability of :class:`PhaseResult`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.ml.evaluation import (
    LABELS,
    N_BINS,
    PhaseResult,
    evaluate_phase,
    next_session_oc_return,
)
from src.ml.constants import TARGET_COLUMN
from tests.features.fixtures import make_long_ohlcv_df
from tests.ml.fixtures import make_ml_dataset


def _hand_xy():
    y_true = np.array([-1, -1, 0, 0, 1, 1])
    y_pred = np.array([-1, 0, 0, 1, 1, 1])
    y_prob = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.5, 0.4],
            [0.1, 0.2, 0.7],
            [0.2, 0.2, 0.6],
        ]
    )
    return y_true, y_pred, y_prob


class TestClassificationBlock:
    def test_accuracy_and_balanced_accuracy(self):
        y_true, y_pred, _ = _hand_xy()
        res = evaluate_phase(y_true, y_pred, phase="train")
        assert res.n_samples == 6
        assert res.accuracy == pytest.approx(4 / 6)
        assert res.balanced_accuracy == pytest.approx(
            balanced_accuracy_score(y_true, y_pred)
        )

    def test_per_class_and_macro_metrics(self):
        y_true, y_pred, _ = _hand_xy()
        res = evaluate_phase(y_true, y_pred)
        p, r, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=list(LABELS), zero_division=0
        )
        for i, c in enumerate(LABELS):
            assert res.per_class[int(c)]["precision"] == pytest.approx(p[i])
            assert res.per_class[int(c)]["recall"] == pytest.approx(r[i])
            assert res.per_class[int(c)]["f1"] == pytest.approx(f1[i])
            assert res.per_class[int(c)]["support"] == int(sup[i])
        assert res.macro_precision == pytest.approx(p.mean())
        assert res.macro_recall == pytest.approx(r.mean())
        assert res.macro_f1 == pytest.approx(f1.mean())

    def test_confusion_matrix_order(self):
        y_true, y_pred, _ = _hand_xy()
        res = evaluate_phase(y_true, y_pred)
        expected = confusion_matrix(y_true, y_pred, labels=list(LABELS))
        assert res.confusion_matrix == expected.tolist()
        assert np.asarray(res.confusion_matrix).shape == (3, 3)

    def test_class_distribution(self):
        y_true, y_pred, _ = _hand_xy()
        res = evaluate_phase(y_true, y_pred)
        assert res.class_distribution == {-1: 2, 0: 2, 1: 2}

    def test_phase_label_stamped(self):
        res = evaluate_phase(*_hand_xy()[:2], phase="validation")
        assert res.phase == "validation"


class TestRocAuc:
    def test_macro_ovr_matches_sklearn(self):
        y_true, _, y_prob = _hand_xy()
        res = evaluate_phase(y_true, y_pred=y_true, y_prob=y_prob)
        expected = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro", labels=list(LABELS)
        )
        assert res.roc_auc_macro_ovr == pytest.approx(expected)

    def test_macro_ovr_none_when_class_missing(self):
        y_true = np.array([-1, -1, 1, 1])  # no neutral class
        y_prob = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.2, 0.1, 0.7]])
        res = evaluate_phase(y_true, y_pred=y_true, y_prob=y_prob)
        assert res.roc_auc_macro_ovr is None

    def test_binary_lens_auc(self):
        y_true = np.array([-1, -1, 1, 1, 0, 0])
        y_pred = np.array([-1, 1, 1, 1, 0, 0])
        y_prob = np.array(
            [[0.7, 0.2, 0.1], [0.3, 0.3, 0.4], [0.2, 0.2, 0.6], [0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.3, 0.4, 0.3]]
        )
        res = evaluate_phase(y_true, y_pred, y_prob=y_prob)
        lens = y_true != 0
        expected = roc_auc_score(y_true[lens] == 1, y_prob[lens, 2])
        assert res.roc_auc_binary_lens == pytest.approx(expected)

    def test_binary_lens_none_when_single_class(self):
        y_true = np.array([-1, -1, 0, 0])
        y_pred = np.array([-1, -1, 0, 0])
        y_prob = np.array([[0.8, 0.1, 0.1]] * 4)
        res = evaluate_phase(y_true, y_pred, y_prob=y_prob)
        assert res.roc_auc_binary_lens is None


class TestProbabilityMetrics:
    def test_log_loss_and_brier(self):
        y_true, _, y_prob = _hand_xy()
        res = evaluate_phase(y_true, y_pred=y_true, y_prob=y_prob)
        assert res.log_loss == pytest.approx(log_loss(y_true, y_prob, labels=list(LABELS)))
        onehot = np.zeros((len(y_true), 3))
        for i, c in enumerate(LABELS):
            onehot[:, i] = y_true == c
        expected_brier = float(np.mean(np.sum((onehot - y_prob) ** 2, axis=1)))
        assert res.brier_multi == pytest.approx(expected_brier)

    def test_metrics_none_without_proba(self):
        res = evaluate_phase(*_hand_xy()[:2])
        assert res.log_loss is None
        assert res.brier_multi is None
        assert res.roc_auc_macro_ovr is None

    def test_calibration_only_when_requested(self):
        y_true, _, y_prob = _hand_xy()
        off = evaluate_phase(y_true, y_pred=y_true, y_prob=y_prob)
        assert off.calibration is None
        on = evaluate_phase(y_true, y_pred=y_true, y_prob=y_prob, compute_calibration=True)
        assert on.calibration is not None
        for cls in ("-1", "0", "1"):
            entry = on.calibration[cls]
            if entry is not None:
                assert len(entry["mean_predicted"]) == len(entry["fraction_positive"])
                assert entry["n_bins"] > 0


class TestFinancialDiagnostics:
    def test_mean_return_by_predicted_class(self):
        y_pred = np.array([1, 1, -1, -1, 0, 0])
        rr = np.array([0.01, 0.02, -0.01, 0.005, 0.0, np.nan])
        res = evaluate_phase(y_pred, y_pred, realized_return=rr)
        fin = res.financial
        assert fin["mean_realized_return_by_predicted_class"]["1"] == pytest.approx(0.015)
        assert fin["mean_realized_return_by_predicted_class"]["-1"] == pytest.approx(-0.0025)
        assert fin["mean_realized_return_by_predicted_class"]["0"] == pytest.approx(0.0)
        assert fin["mean_realized_return_predicted_bullish"] == pytest.approx(0.015)
        assert fin["mean_realized_return_predicted_bearish"] == pytest.approx(-0.0025)
        assert fin["long_short_proxy_spread"] == pytest.approx(0.0175)

    def test_financial_none_without_realized(self):
        res = evaluate_phase(*_hand_xy()[:2])
        assert res.financial is None

    def test_confidence_deciles(self):
        y_pred = np.array([1, 1, -1, -1, 0, 0, 1, 1, -1, -1])
        rr = np.linspace(-0.01, 0.01, 10)
        y_prob = np.zeros((10, 3))
        y_prob[:, 2] = np.linspace(0.4, 0.9, 10)  # confidence varies
        y_prob[:, 0] = (1.0 - y_prob[:, 2]) / 2
        y_prob[:, 1] = y_prob[:, 0]
        res = evaluate_phase(y_pred, y_pred, y_prob=y_prob, realized_return=rr)
        deciles = res.financial["confidence_deciles"]
        assert len(deciles) == N_BINS
        assert sum(d["n"] for d in deciles) == 10
        assert all(d["mean_confidence"] <= 1.0 for d in deciles)


class TestValidation:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            evaluate_phase(np.array([1, 1]), np.array([1]))

    def test_labels_outside_domain_raise(self):
        with pytest.raises(ValueError, match="y_true"):
            evaluate_phase(np.array([2, 1]), np.array([1, 1]))
        with pytest.raises(ValueError, match="y_pred"):
            evaluate_phase(np.array([1, 1]), np.array([5, 1]))

    def test_prob_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="y_prob"):
            evaluate_phase(
                np.array([1, 1]), np.array([1, 1]), y_prob=np.ones((2, 2))
            )

    def test_realized_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="align"):
            evaluate_phase(
                np.array([-1, 1]), np.array([-1, 1]), realized_return=np.array([0.1])
            )

    def test_to_dict_is_json_serializable(self):
        y_true, y_pred, y_prob = _hand_xy()
        res = evaluate_phase(
            y_true, y_pred, y_prob=y_prob, realized_return=np.ones(6) * 0.001,
            compute_calibration=True,
        )
        d = res.to_dict()
        json.dumps(d)  # must not raise
        assert isinstance(d["accuracy"], float)
        assert isinstance(d["confusion_matrix"][0][0], int)
        assert isinstance(d["per_class"]["-1"]["precision"], float)


class TestPhaseIsolation:
    def test_phases_never_conflated(self):
        """The same estimator must get different results per phase by design —
        metrics depend only on the arrays passed for that phase."""
        y1, p1 = np.array([1, 1, 1, 0]), np.array([1, 1, 1, 1])
        y2, p2 = np.array([-1, -1, -1, 0]), np.array([-1, -1, -1, -1])
        r1 = evaluate_phase(y1, p1, phase="train")
        r2 = evaluate_phase(y2, p2, phase="test")
        assert r1.phase == "train" and r2.phase == "test"
        # metrics are per-phase: train accuracy 1.0, test accuracy 1.0 but
        # distributions differ; recomputing train must not see test data.
        r1b = evaluate_phase(y1, p1, phase="train")
        assert r1b.class_distribution == r1.class_distribution
        assert r1b.accuracy == r1.accuracy


class TestRealizedReturn:
    def test_per_symbol_next_session_oc(self):
        ohlcv = make_long_ohlcv_df(symbols=("AAPL", "MSFT"), n=10, seed=1)
        rr = next_session_oc_return(ohlcv)
        assert list(rr.columns) == ["symbol", "timestamp", "realized_return"]
        for symbol in ("AAPL", "MSFT"):
            sub = ohlcv[ohlcv["symbol"] == symbol].sort_values("timestamp")
            expected = (sub["close"] / sub["open"] - 1.0).shift(-1).to_numpy()
            got = (
                rr[rr["symbol"] == symbol]
                .sort_values("timestamp")["realized_return"]
                .to_numpy()
            )
            np.testing.assert_allclose(got, expected, equal_nan=True)

    def test_final_session_per_symbol_is_nan(self):
        ohlcv = make_long_ohlcv_df(symbols=("AAPL", "MSFT"), n=8, seed=2)
        rr = next_session_oc_return(ohlcv)
        for symbol in ("AAPL", "MSFT"):
            sub = ohlcv[ohlcv["symbol"] == symbol].sort_values("timestamp")
            last_ts = sub["timestamp"].iloc[-1]
            val = rr.loc[
                (rr["symbol"] == symbol) & (rr["timestamp"] == last_ts),
                "realized_return",
            ].iloc[0]
            assert np.isnan(val)

    def test_aligned_on_ml_panel(self):
        """realized_return merged into the panel has no NaNs (all rows have a
        future session after the warm-up / no-target drops)."""
        ds = make_ml_dataset(symbols=("AAPL", "MSFT", "SPY"), n=40, seed=3)
        merged = ds.df.merge(
            next_session_oc_return(
                make_long_ohlcv_df(symbols=("AAPL", "MSFT", "SPY"), n=40, seed=3)
            ),
            on=["symbol", "timestamp"],
            how="left",
        )
        assert merged["realized_return"].isna().sum() == 0
        # Guarantee every predicted class is present so all financial means
        # are defined (np.resize tiles [-1, 0, 1] over the panel length).
        y_pred = np.resize(np.array([-1, 0, 1]), len(ds.df))
        res = evaluate_phase(
            ds.y.to_numpy(),
            y_pred,
            y_prob=np.full((len(ds.df), 3), 1 / 3),
            realized_return=merged["realized_return"].to_numpy(),
        )
        assert res.financial is not None
        for cls in ("-1", "0", "1"):
            assert res.financial["mean_realized_return_by_predicted_class"][cls] is not None
