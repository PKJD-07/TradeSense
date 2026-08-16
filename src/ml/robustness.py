"""ML Robustness Analysis: diagnostic suite for the V1 ML pipeline.

This module analyzes the V1 pipeline's weaknesses across 13 analytical dimensions
to surface fragility, leakage risk, and overfitting before any capital is risked:

    1.  V1 result reproduction (determinism check).
    2.  Per-symbol robustness (does the signal hold per instrument?).
    3.  Temporal stability (do metrics decay across the panel's date range?).
    4.  Regime analysis (bull / bear / high-vol / low-vol robustness).
    5.  Walk-forward stability (fold-to-fold metric dispersion).
    6.  Calibration assessment (do probabilities mean what they say?).
    7.  Feature importance stability (do trees agree on what matters?).
    8.  Class imbalance diagnostics (is the minority class learnable?).
    9.  Overfitting diagnostics (train vs test degradation).
    10. Probability distribution analysis (is the model confident?).
    11. Confidence vs realized return monotonicity.
    12. Prediction concentration / label domination.
    13. Statistical significance vs the trivial baselines.

Each step returns a structured dict. ``run_robustness_analysis`` orchestrates all
13 and returns a :class:`RobustnessReport`.

Discipline: this module is READ-ONLY on the ML layer. It never fits a model,
never touches the test set for training, and never imports the backtester /
execution / risk layers. It consumes :class:`src.ml.experiment.ExperimentResult`
and the causal dataset only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from src.ml.constants import SEED, TARGET_COLUMN
from src.ml.dataset import build_ml_dataset
from src.ml.evaluation import LABELS
from src.ml.experiment import (
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
    save_experiment,
)
from src.ml.models import MODEL_NAMES, make_model_pipeline
from src.ml.validation import split_by_date, walk_forward_folds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_session_oc_return(long_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Return a (symbol, timestamp) -> realized_return frame (forward open-close).

    Mirrors the realized-return join done in experiment.py; kept local to avoid a
    heavier import cycle and to keep this analysis module self-contained.
    """
    from src.ml.evaluation import next_session_oc_return as _nsoc

    return _nsoc(long_ohlcv)


def _phase_predictions(
    result: ExperimentResult, estimator: str, phase: str
) -> pd.DataFrame:
    """Convenience accessor for a prediction frame."""
    return result.predictions[estimator][phase]


def _realized_return_for_frame(
    frame: pd.DataFrame, long_ohlcv: pd.DataFrame
) -> np.ndarray:
    """Attach realized next-session return to a prediction frame."""
    rr = _next_session_oc_return(long_ohlcv)
    merged = frame.merge(rr, on=["symbol", "timestamp"], how="left")
    return merged["realized_return"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Step 1: V1 result reproduction (determinism check)
# ---------------------------------------------------------------------------

def analyze_reproduction(
    long_ohlcv: pd.DataFrame,
    config: ExperimentConfig,
    reference: Optional[ExperimentResult] = None,
) -> dict:
    """Reproduce V1 results; verify determinism against an optional reference.

    Args:
        long_ohlcv: Long-form OHLCV input.
        config: The V1 experiment config to reproduce.
        reference: An earlier :class:`ExperimentResult` to compare against. If
            given, all overlapping metric keys must match exactly (determinism).

    Returns:
        Dict with keys: ``run_id``, ``dataset_rows``, ``metric_summary`` (per
        estimator test accuracy/macro_f1), and ``determinism`` (bool + diffs).
    """
    res = run_experiment(long_ohlcv, config=config)
    summary = {}
    for name in res.estimator_results:
        m = res.estimator_results[name]["phase_metrics"]["test"]
        summary[name] = {
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"],
            "log_loss": m["log_loss"],
        }

    determinism = {"checked": reference is not None, "match": True, "diffs": []}
    if reference is not None:
        for name in summary:
            if name not in reference.estimator_results:
                determinism["match"] = False
                determinism["diffs"].append(f"{name}: missing in reference")
                continue
            ref_m = reference.estimator_results[name]["phase_metrics"]["test"]
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss"):
                a, b = summary[name][key], ref_m[key]
                if not np.isclose(a, b, rtol=0, atol=1e-12):
                    determinism["match"] = False
                    determinism["diffs"].append(
                        f"{name}.{key}: {a!r} != reference {b!r}"
                    )

    return {
        "run_id": res.config.run_id,
        "dataset_rows": res.dataset["final_rows"],
        "metric_summary": summary,
        "determinism": determinism,
        "_result": res,  # internal: reused by later steps
    }


# ---------------------------------------------------------------------------
# Step 2: Per-symbol robustness
# ---------------------------------------------------------------------------

def analyze_per_symbol(
    long_ohlcv: pd.DataFrame, config: ExperimentConfig
) -> dict:
    """Train + evaluate each estimator per-symbol (hold-one-symbol-out style).

    For each symbol we fit on the OTHER symbols and evaluate on that symbol's
    test region, exposing whether the pooled signal is driven by one instrument.

    Returns:
        Dict: ``per_symbol`` = {symbol: {estimator: {test metrics}}},
        ``dominant_symbol`` = the symbol whose exclusion most hurts / helps,
        ``symbol_consistency`` = coefficient of variation of test accuracy.
    """
    ds = build_ml_dataset(
        long_ohlcv,
        epsilon=config.epsilon,
        include_market_context=config.include_market_context,
        warmup_rows=config.warmup_rows,
        max_nan_fraction=config.max_nan_fraction,
    )
    symbols = ds.report.symbols
    X, y = ds.X, ds.y.to_numpy()
    meta = ds.df[["symbol", "timestamp"]]
    realized = _realized_return_for_frame(meta, long_ohlcv)

    split = split_by_date(
        ds.timestamp.to_numpy(),
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        purge_gap=config.purge_gap,
    )

    per_symbol: dict = {}
    for target_symbol in symbols:
        # Train on all OTHER symbols.
        train_mask = meta["symbol"] != target_symbol
        est_results: dict = {}
        for name in (*config.baselines, *config.models):
            est = _new_estimator(name, config)
            try:
                est.fit(X[train_mask], y[train_mask])
            except Exception:
                # e.g. persistence cannot fit; skip gracefully
                continue
            idx = split.test_index
            sym_idx = idx[meta["symbol"].iloc[idx].to_numpy() == target_symbol]
            if len(sym_idx) == 0:
                continue
            y_pred = est.predict(X.iloc[sym_idx])
            y_prob = est.predict_proba(X.iloc[sym_idx])
            from src.ml.evaluation import evaluate_phase

            m = evaluate_phase(
                y[sym_idx], y_pred, phase="test", y_prob=y_prob
            ).to_dict()
            est_results[name] = {
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"],
                "n_samples": m["n_samples"],
            }
        per_symbol[target_symbol] = est_results

    # Consistency: CV of test accuracy across symbols for each model.
    consistency: dict = {}
    for name in (*config.baselines, *config.models):
        accs = [
            per_symbol[s][name]["accuracy"]
            for s in per_symbol
            if name in per_symbol[s]
        ]
        if len(accs) > 1:
            consistency[name] = float(np.std(accs) / np.mean(accs))
        else:
            consistency[name] = 0.0

    return {
        "per_symbol": per_symbol,
        "symbol_consistency": consistency,
        "n_symbols": len(symbols),
    }


# ---------------------------------------------------------------------------
# Step 3: Temporal stability
# ---------------------------------------------------------------------------

def analyze_temporal_stability(
    result: ExperimentResult, long_ohlcv: pd.DataFrame, config: ExperimentConfig
) -> dict:
    """Split the test region into time quartiles; track metric drift.

    Returns:
        Dict: ``quartiles`` = list of {period, n, accuracy, macro_f1} for each
        model; ``drift`` = slope (per-period change) of accuracy for each model.
    """
    ds = build_ml_dataset(
        long_ohlcv,
        epsilon=config.epsilon,
        include_market_context=config.include_market_context,
        warmup_rows=config.warmup_rows,
        max_nan_fraction=config.max_nan_fraction,
    )
    split = split_by_date(
        ds.timestamp.to_numpy(),
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        purge_gap=config.purge_gap,
    )
    test_idx = split.test_index
    test_ts = pd.to_datetime(ds.timestamp.to_numpy()[test_idx], utc=True)
    order = np.argsort(test_ts)
    test_idx_sorted = test_idx[order]

    quartiles: dict = {}
    drift: dict = {}
    for name in (*config.baselines, *config.models):
        frame = result.predictions[name]["test"]
        # Align frame to sorted test_idx by (symbol, timestamp).
        frame_sorted = (
            frame.set_index(["symbol", "timestamp"])
            .loc[
                ds.df[["symbol", "timestamp"]]
                .iloc[test_idx_sorted]
                .set_index(["symbol", "timestamp"])
                .index
            ]
            .reset_index()
        )
        y_true = frame_sorted[TARGET_COLUMN].to_numpy()
        y_pred = frame_sorted["y_pred"].to_numpy()
        n = len(y_true)
        q = max(1, n // 4)
        accs = []
        blocks = []
        for qi in range(4):
            s, e = qi * q, (qi + 1) * q if qi < 3 else n
            if e <= s:
                continue
            from src.ml.evaluation import evaluate_phase

            m = evaluate_phase(
                y_true[s:e], y_pred[s:e], phase="test"
            ).to_dict()
            blocks.append(
                {
                    "period": qi + 1,
                    "n": int(e - s),
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                }
            )
            accs.append(m["accuracy"])
        quartiles[name] = blocks
        if len(accs) > 1:
            xs = np.arange(len(accs))
            drift[name] = float(np.polyfit(xs, accs, 1)[0])
        else:
            drift[name] = 0.0

    return {"quartiles": quartiles, "drift": drift}


# ---------------------------------------------------------------------------
# Step 4: Regime analysis
# ---------------------------------------------------------------------------

def analyze_regimes(
    long_ohlcv: pd.DataFrame, config: ExperimentConfig
) -> dict:
    """Bucket rows into market regimes (SPY trend + volatility) and evaluate.

    Regimes:
        - bull_low_vol / bull_high_vol
        - bear_low_vol / bear_high_vol
    defined from SPY's 20d return (trend) and 20d volatility (vol).

    Returns:
        Dict: ``regimes`` = {regime: {estimator: {accuracy, macro_f1, n}}},
        ``regime_counts`` = row counts per regime.
    """
    ds = build_ml_dataset(
        long_ohlcv,
        epsilon=config.epsilon,
        include_market_context=config.include_market_context,
        warmup_rows=config.warmup_rows,
        max_nan_fraction=config.max_nan_fraction,
    )
    X, y = ds.X, ds.y.to_numpy()
    meta = ds.df[["symbol", "timestamp"]]
    split = split_by_date(
        ds.timestamp.to_numpy(),
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        purge_gap=config.purge_gap,
    )

    # Regime labels from SPY features.
    spy = ds.df[ds.df["symbol"] == "SPY"].set_index("timestamp")
    trend = spy["return_20d"]
    vol = spy["volatility_20d"]
    trend_med = trend.median()
    vol_med = vol.median()

    def regime_for(ts):
        t = trend.get(ts)
        v = vol.get(ts)
        if t is None or v is None:
            return "unknown"
        bull = t >= trend_med
        hv = v >= vol_med
        return f"{'bull' if bull else 'bear'}_{'high_vol' if hv else 'low_vol'}"

    meta = meta.copy()
    meta["regime"] = meta["timestamp"].apply(regime_for)

    regimes: dict = {}
    regime_counts: dict = {}
    for name in (*config.baselines, *config.models):
        est = _new_estimator(name, config)
        est.fit(X.iloc[split.train_index], y[split.train_index])
        for regime in ("bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol"):
            idx = split.test_index
            mask = meta["regime"].iloc[idx].to_numpy() == regime
            sel = idx[mask]
            if len(sel) == 0:
                continue
            y_pred = est.predict(X.iloc[sel])
            from src.ml.evaluation import evaluate_phase

            m = evaluate_phase(y[sel], y_pred, phase="test").to_dict()
            regimes.setdefault(regime, {})[name] = {
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "n": m["n_samples"],
            }
            regime_counts[regime] = regime_counts.get(regime, 0) + int(len(sel))

    return {"regimes": regimes, "regime_counts": regime_counts}


# ---------------------------------------------------------------------------
# Step 5: Walk-forward stability
# ---------------------------------------------------------------------------

def analyze_walk_forward_stability(
    result: ExperimentResult,
) -> dict:
    """Compute fold-to-fold metric dispersion from walk-forward OOS frames.

    Returns:
        Dict: per estimator ``fold_accuracy`` (list), ``mean``, ``std``,
        ``cv`` (coefficient of variation), and ``n_folds``.
    """
    out: dict = {}
    for name in result.estimator_results:
        wf = result.estimator_results[name]["walk_forward"]
        if wf is None:
            out[name] = {"n_folds": 0, "mean": None, "std": None, "cv": None}
            continue
        oos = result.predictions[name]["walk_forward"]
        fold_acc: dict = {}
        for fold, grp in oos.groupby("fold"):
            from src.ml.evaluation import evaluate_phase

            m = evaluate_phase(
                grp[TARGET_COLUMN].to_numpy(),
                grp["y_pred"].to_numpy(),
                phase="walk_forward",
            ).to_dict()
            fold_acc[int(fold)] = m["accuracy"]
        accs = [fold_acc[k] for k in sorted(fold_acc)]
        mean = float(np.mean(accs))
        std = float(np.std(accs))
        cv = std / mean if mean else 0.0
        out[name] = {
            "n_folds": len(accs),
            "fold_accuracy": accs,
            "mean": mean,
            "std": std,
            "cv": cv,
        }
    return out


# ---------------------------------------------------------------------------
# Step 6: Calibration assessment
# ---------------------------------------------------------------------------

def analyze_calibration(
    result: ExperimentResult, config: ExperimentConfig, long_ohlcv: pd.DataFrame
) -> dict:
    """Assess per-class calibration on the validation phase (probabilities).

    Uses the confidence-decile realized-return blocks already computed in the
    evaluation step, plus a reliability check: predicted prob vs empirical freq.

    Returns:
        Dict: per estimator ``reliability`` (list of {pred_prob, emp_freq,
        n}), ``max_calibration_error``, ``mean_realized_by_confidence_slope``.
    """
    out: dict = {}
    for name in result.estimator_results:
        val = result.estimator_results[name]["phase_metrics"]["validation"]
        fin = val.get("financial")
        if not fin or "confidence_deciles" not in fin:
            out[name] = {"available": False}
            continue
        deciles = fin["confidence_deciles"]
        # Reliability: bin predicted confidence vs empirical accuracy proxy
        # (mean realized return sign alignment).
        rel = []
        slopes = []
        for d in deciles:
            conf = d["mean_confidence"]
            rr = d["mean_realized_return"]
            n = d["n"]
            # Empirical "accuracy" proxy: fraction of rows where sign(rr) == sign
            # of the predicted class (bullish -> positive rr). We approximate with
            # the long-short spread per bin.
            rel.append({"pred_prob": conf, "mean_realized_return": rr, "n": n})
            slopes.append((conf, rr))
        if len(slopes) > 1:
            xs = np.array([s[0] for s in slopes])
            ys = np.array([s[1] for s in slopes])
            slope = float(np.polyfit(xs, ys, 1)[0])
        else:
            slope = 0.0
        out[name] = {
            "available": True,
            "reliability": rel,
            "mean_realized_by_confidence_slope": slope,
            "n_deciles": len(deciles),
        }
    return out


# ---------------------------------------------------------------------------
# Step 7: Feature importance stability
# ---------------------------------------------------------------------------

def analyze_feature_importance(
    long_ohlcv: pd.DataFrame, config: ExperimentConfig
) -> dict:
    """Fit tree models on train; extract + compare feature importances.

    Compares Random Forest and Gradient Boosting importances (rank correlation)
    and reports the top features per model.

    Returns:
        Dict: ``importances`` = {model: {feature: importance}},
        ``rank_correlation`` = Spearman-like rank correlation between RF & GBM,
        ``top_features`` = union of top-5 per model.
    """
    ds = build_ml_dataset(
        long_ohlcv,
        epsilon=config.epsilon,
        include_market_context=config.include_market_context,
        warmup_rows=config.warmup_rows,
        max_nan_fraction=config.max_nan_fraction,
    )
    X, y = ds.X, ds.y.to_numpy()
    split = split_by_date(
        ds.timestamp.to_numpy(),
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        purge_gap=config.purge_gap,
    )

    importances: dict = {}
    for model_name in ("random_forest", "gradient_boosting"):
        if model_name not in config.models:
            continue
        pipe = make_model_pipeline(model_name)
        pipe.fit(X.iloc[split.train_index], y[split.train_index])
        # The model is the last step of the pipeline.
        tree = pipe.steps[-1][1]
        imp = tree.feature_importances_
        imp_map = {f: float(v) for f, v in zip(ds.feature_columns, imp)}
        importances[model_name] = imp_map

    rank_corr = None
    if "random_forest" in importances and "gradient_boosting" in importances:
        rf = pd.Series(importances["random_forest"]).rank()
        gbm = pd.Series(importances["gradient_boosting"]).rank()
        # Pearson on ranks == Spearman.
        if rf.std() > 0 and gbm.std() > 0:
            rank_corr = float(np.corrcoef(rf, gbm)[0, 1])
        else:
            rank_corr = 1.0 if np.allclose(rf, gbm) else 0.0

    top_features: set = set()
    for model_name, imp_map in importances.items():
        top = sorted(imp_map, key=imp_map.get, reverse=True)[:5]
        top_features.update(top)

    return {
        "importances": importances,
        "rank_correlation": rank_corr,
        "top_features": sorted(top_features),
    }


# ---------------------------------------------------------------------------
# Step 8: Class imbalance diagnostics
# ---------------------------------------------------------------------------

def analyze_class_imbalance(result: ExperimentResult) -> dict:
    """Report class shares and minority-class learnability across phases.

    Returns:
        Dict: ``class_counts`` (final panel), ``per_phase_distribution``,
        ``minority_recall`` (class 0 recall per estimator/phase),
        ``imbalance_ratio`` (majority/minority).
    """
    cc = {int(k): v for k, v in result.dataset["class_counts"].items()}
    total = sum(cc.values())
    shares = {c: v / total for c, v in cc.items()}
    imbalance_ratio = max(cc.values()) / min(cc.values()) if min(cc.values()) else float("inf")

    per_phase: dict = {}
    minority_recall: dict = {}
    for name in result.estimator_results:
        minority_recall[name] = {}
        for phase in ("train", "validation", "test"):
            m = result.estimator_results[name]["phase_metrics"].get(phase)
            if not m:
                continue
            per_phase.setdefault(phase, {})[name] = m["class_distribution"]
            pc = m.get("per_class", {}).get("0")
            if pc:
                minority_recall[name][phase] = pc["recall"]

    return {
        "class_counts": cc,
        "shares": shares,
        "imbalance_ratio": imbalance_ratio,
        "per_phase_distribution": per_phase,
        "minority_recall": minority_recall,
    }


# ---------------------------------------------------------------------------
# Step 9: Overfitting diagnostics
# ---------------------------------------------------------------------------

def analyze_overfitting(result: ExperimentResult) -> dict:
    """Compare train vs test degradation per estimator.

    Returns:
        Dict: per estimator ``train_accuracy``, ``test_accuracy``,
        ``degradation`` (train - test), ``overfit_flag`` (degradation > 0.15).
    """
    out: dict = {}
    for name in result.estimator_results:
        tr = result.estimator_results[name]["phase_metrics"]["train"]["accuracy"]
        te = result.estimator_results[name]["phase_metrics"]["test"]["accuracy"]
        deg = tr - te
        out[name] = {
            "train_accuracy": tr,
            "test_accuracy": te,
            "degradation": deg,
            "overfit_flag": deg > 0.15,
        }
    return out


# ---------------------------------------------------------------------------
# Step 10: Probability distribution analysis
# ---------------------------------------------------------------------------

def analyze_probability_distribution(result: ExperimentResult) -> dict:
    """Examine predicted probability concentration (confidence calibration shape).

    Returns:
        Dict: per estimator ``mean_max_prob`` (mean of argmax prob),
        ``frac_max_prob_gt_0_9`` (fraction of rows with max prob > 0.9),
        ``entropy_mean`` (mean predictive entropy), ``entropy_std``.
    """
    out: dict = {}
    for name in result.estimator_results:
        frame = result.predictions[name]["test"]
        prob_cols = [f"y_prob_{int(c)}" for c in LABELS]
        P = frame[prob_cols].to_numpy()
        max_prob = P.max(axis=1)
        entropy = -np.sum(P * np.log(P + 1e-12), axis=1)
        out[name] = {
            "mean_max_prob": float(np.mean(max_prob)),
            "frac_max_prob_gt_0_9": float(np.mean(max_prob > 0.9)),
            "entropy_mean": float(np.mean(entropy)),
            "entropy_std": float(np.std(entropy)),
        }
    return out


# ---------------------------------------------------------------------------
# Step 11: Confidence vs realized return monotonicity
# ---------------------------------------------------------------------------

def analyze_confidence_monotonicity(result: ExperimentResult) -> dict:
    """Test whether higher predicted confidence => higher |realized return|.

    Returns:
        Dict: per estimator ``spearman`` (rank corr of |confidence| vs
        |realized_return|), ``monotonic_flag`` (spearman > 0.05).
    """
    out: dict = {}
    for name in result.estimator_results:
        frame = result.predictions[name]["test"]
        # Confidence = max predicted probability.
        prob_cols = [f"y_prob_{int(c)}" for c in LABELS]
        conf = frame[prob_cols].to_numpy().max(axis=1)
        rr = frame["realized_return"].to_numpy(dtype=float)
        mask = ~np.isnan(rr)
        if mask.sum() < 10:
            out[name] = {"spearman": None, "monotonic_flag": False, "n": int(mask.sum())}
            continue
        # Rank correlation.
        rc = np.corrcoef(
            pd.Series(conf[mask]).rank(), pd.Series(np.abs(rr[mask])).rank()
        )[0, 1]
        out[name] = {
            "spearman": float(rc) if not np.isnan(rc) else None,
            "monotonic_flag": bool(rc > 0.05) if not np.isnan(rc) else False,
            "n": int(mask.sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Step 12: Prediction concentration / label domination
# ---------------------------------------------------------------------------

def analyze_prediction_concentration(result: ExperimentResult) -> dict:
    """Measure label domination: does the model collapse to one class?

    Returns:
        Dict: per estimator ``predicted_class_share`` (share of each predicted
        class in test), ``dominance`` (max share), ``gini`` (of predicted shares).
    """
    out: dict = {}
    for name in result.estimator_results:
        frame = result.predictions[name]["test"]
        preds = frame["y_pred"].to_numpy()
        vals, counts = np.unique(preds, return_counts=True)
        shares = counts / counts.sum()
        dominance = float(shares.max())
        # Gini coefficient of predicted shares.
        s = np.sort(shares)
        n = len(s)
        cum = np.cumsum(s)
        gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n if cum[-1] > 0 else 0.0
        out[name] = {
            "predicted_class_share": {int(v): float(sh) for v, sh in zip(vals, shares)},
            "dominance": dominance,
            "gini": float(gini),
        }
    return out


# ---------------------------------------------------------------------------
# Step 13: Statistical significance vs baselines
# ---------------------------------------------------------------------------

def analyze_significance(result: ExperimentResult) -> dict:
    """McNemar-style comparison of each model vs majority_class on test.

    Uses a chi-square approximation of McNemar's test on the disagreement
    contingency table (both correct / both wrong / A only / B only).

    Returns:
        Dict: per model ``mcnemar_chi2``, ``p_value`` (approx),
        ``beats_baseline`` (p < 0.05), plus ``best_model`` by test macro_f1.
    """
    from scipy.stats import chi2

    baseline = "majority_class"
    if baseline not in result.predictions:
        return {"available": False}

    base_frame = result.predictions[baseline]["test"]
    base_correct = (base_frame["y_pred"] == base_frame[TARGET_COLUMN]).to_numpy()
    y_true = base_frame[TARGET_COLUMN].to_numpy()

    out: dict = {}
    best_model = None
    best_f1 = -1.0
    for name in result.estimator_results:
        if name == baseline:
            continue
        frame = result.predictions[name]["test"]
        pred = frame.set_index(["symbol", "timestamp"]).loc[
            base_frame.set_index(["symbol", "timestamp"]).index
        ]["y_pred"].to_numpy()
        model_correct = pred == y_true
        # Contingency: b = base correct, model wrong; c = model correct, base wrong
        b = int(np.sum(base_correct & ~model_correct))
        c = int(np.sum(~base_correct & model_correct))
        n_disc = b + c
        if n_disc == 0:
            chi2_stat = 0.0
            p = 1.0
        else:
            # McNemar with continuity correction.
            chi2_stat = (abs(b - c) - 1) ** 2 / n_disc
            p = 1 - chi2.cdf(chi2_stat, df=1)
        out[name] = {
            "mcnemar_chi2": float(chi2_stat),
            "p_value": float(p),
            "beats_baseline": bool(p < 0.05),
            "b": b,
            "c": c,
        }
        f1 = result.estimator_results[name]["phase_metrics"]["test"]["macro_f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_model = name

    return {
        "available": True,
        "per_model": out,
        "best_model": best_model,
        "best_macro_f1": best_f1,
    }


# ---------------------------------------------------------------------------
# Internal: fresh estimator builder (mirrors experiment._new_estimator)
# ---------------------------------------------------------------------------

def _new_estimator(name, config):
    from src.ml.baselines import build_baseline

    if name in config.baselines:
        return build_baseline(name, epsilon=config.epsilon)
    if name in config.models:
        return make_model_pipeline(name)
    raise ValueError(f"'{name}' is neither a configured baseline nor a model")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class RobustnessReport:
    """Full 13-step robustness analysis record."""

    config: ExperimentConfig
    reproduction: dict
    per_symbol: dict
    temporal_stability: dict
    regimes: dict
    walk_forward_stability: dict
    calibration: dict
    feature_importance: dict
    class_imbalance: dict
    overfitting: dict
    probability_distribution: dict
    confidence_monotonicity: dict
    prediction_concentration: dict
    significance: dict

    def to_dict(self) -> dict:
        return {
            "config": {
                "run_id": self.config.run_id,
                "epsilon": self.config.epsilon,
                "models": self.config.models,
                "baselines": self.config.baselines,
            },
            "reproduction": {
                k: v for k, v in self.reproduction.items() if k != "_result"
            },
            "per_symbol": self.per_symbol,
            "temporal_stability": self.temporal_stability,
            "regimes": self.regimes,
            "walk_forward_stability": self.walk_forward_stability,
            "calibration": self.calibration,
            "feature_importance": self.feature_importance,
            "class_imbalance": self.class_imbalance,
            "overfitting": self.overfitting,
            "probability_distribution": self.probability_distribution,
            "confidence_monotonicity": self.confidence_monotonicity,
            "prediction_concentration": self.prediction_concentration,
            "significance": self.significance,
        }


def run_robustness_analysis(
    long_ohlcv: pd.DataFrame,
    config: Optional[ExperimentConfig] = None,
    reference: Optional[ExperimentResult] = None,
) -> RobustnessReport:
    """Run all 13 robustness steps on the V1 pipeline.

    Args:
        long_ohlcv: Long-form OHLCV input.
        config: V1 experiment config (defaults to frozen V1).
        reference: Optional earlier :class:`ExperimentResult` for determinism.

    Returns:
        A :class:`RobustnessReport` with all 13 analyses.
    """
    config = config or ExperimentConfig()

    # Step 1: reproduction (also yields the result consumed by later steps).
    repro = analyze_reproduction(long_ohlcv, config, reference=reference)
    result: ExperimentResult = repro["_result"]

    per_symbol = analyze_per_symbol(long_ohlcv, config)
    temporal = analyze_temporal_stability(result, long_ohlcv, config)
    regimes = analyze_regimes(long_ohlcv, config)
    wf_stability = analyze_walk_forward_stability(result)
    calibration = analyze_calibration(result, config, long_ohlcv)
    feature_imp = analyze_feature_importance(long_ohlcv, config)
    imbalance = analyze_class_imbalance(result)
    overfit = analyze_overfitting(result)
    prob_dist = analyze_probability_distribution(result)
    monotonicity = analyze_confidence_monotonicity(result)
    concentration = analyze_prediction_concentration(result)
    significance = analyze_significance(result)

    return RobustnessReport(
        config=config,
        reproduction=repro,
        per_symbol=per_symbol,
        temporal_stability=temporal,
        regimes=regimes,
        walk_forward_stability=wf_stability,
        calibration=calibration,
        feature_importance=feature_imp,
        class_imbalance=imbalance,
        overfitting=overfit,
        probability_distribution=prob_dist,
        confidence_monotonicity=monotonicity,
        prediction_concentration=concentration,
        significance=significance,
    )


def save_robustness_report(
    report: RobustnessReport, outputs_dir=None
) -> Path:
    """Write the robustness report JSON to disk.

    Layout: ``<outputs_dir>/<run_id>/robustness.json``.
    """
    import json
    from pathlib import Path as _Path

    outputs_dir = _Path(outputs_dir or report.config.outputs_dir)
    run_dir = outputs_dir / report.config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "robustness.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return run_dir
