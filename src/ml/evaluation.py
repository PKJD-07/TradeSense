"""Per-phase evaluation metrics for the ML pipeline.

Metrics are computed per **phase** (``train`` / ``validation`` / ``test``) and
never conflated: every call to :func:`evaluate_phase` receives the true labels,
predictions, probabilities, and (optionally) realized returns for ONE phase
only. The phase name is stamped onto the result.

Blocks (docs/ml_pipeline.md, §8):
    - Classification: accuracy, balanced accuracy, per-class precision / recall
      / F1 (``zero_division=0``), macro precision / recall / F1, 3x3 confusion
      matrix, class distribution.
    - ROC-AUC where mathematically appropriate: macro one-vs-rest AUC for the
      3-class problem (secondary diagnostic; ``None`` if any class is absent)
      and standard binary ROC-AUC for the ``y != 0`` lens.
    - Probability metrics: multi-class Log Loss, multi-class Brier, and
      per-class calibration curves (validation only — requested via
      ``compute_calibration``).

Financial-interpretation diagnostics (clearly labeled **not** strategy returns):
    - Mean realized next-session return conditional on predicted class.
    - Mean return of predicted-bullish / predicted-bearish subsets and the
      long-short proxy spread.
    - Confidence vs realized return: deciles of the max-class probability vs
      mean realized return per bin.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

#: Canonical 3-class order for the target {-1, 0, +1}.
LABELS = (-1, 0, 1)

#: Number of bins for per-class calibration curves and the confidence deciles.
N_BINS = 10


@dataclass
class PhaseResult:
    """All metrics for one phase.

    Attributes:
        phase: "train", "validation", or "test".
        n_samples: Number of samples evaluated.
        accuracy: Standard accuracy.
        balanced_accuracy: Average recall over classes (imbalance-robust).
        per_class: ``{class: {"precision", "recall", "f1", "support"}}``.
        macro_precision / macro_recall / macro_f1: Unweighted means over the
            3 classes.
        confusion_matrix: 3x3 with rows/cols ordered ``LABELS``.
        class_distribution: ``{class: count}`` of true labels.
        roc_auc_macro_ovr: Macro one-vs-rest ROC-AUC, or ``None`` when not
            mathematically defined (any class absent from ``y_true``).
        roc_auc_binary_lens: Binary ROC-AUC on the ``y != 0`` lens, or ``None``
            when the lens does not contain both ``-1`` and ``+1``.
        log_loss: Multi-class log loss, or ``None`` when probabilities were not
            provided.
        brier_multi: Multi-class Brier score, or ``None`` similarly.
        calibration: Per-class calibration curves (``None`` unless requested).
        financial: Financial-interpretation diagnostics, or ``None`` when no
            realized returns were provided.
    """

    phase: str
    n_samples: int
    accuracy: float
    balanced_accuracy: float
    per_class: dict
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion_matrix: list
    class_distribution: dict
    roc_auc_macro_ovr: Optional[float]
    roc_auc_binary_lens: Optional[float]
    log_loss: Optional[float]
    brier_multi: Optional[float]
    calibration: Optional[dict]
    financial: Optional[dict]

    def to_dict(self) -> dict:
        """JSON-serializable dict with native Python types throughout."""
        return _jsonify(asdict(self))


def _jsonify(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def next_session_oc_return(long_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Per-(symbol, timestamp) realized next-session OC return ``r^OC_{t+1}``.

    ``r^OC_{t+1} = close_{t+1}/open_{t+1} - 1`` is the tradable open-to-close
    move of the session AFTER sample ``t`` — the continuous value underlying the
    target label. Computed PER SYMBOL (``shift`` never crosses a symbol
    boundary). Used only for financial-interpretation diagnostics, never as a
    strategy return.

    Returns:
        DataFrame with columns [symbol, timestamp, realized_return], aligned to
        ``long_ohlcv`` rows (final session of each symbol is NaN).
    """
    frames = []
    for symbol, group in long_ohlcv.groupby("symbol"):
        g = group.sort_values("timestamp").copy()
        g["realized_return"] = (g["close"] / g["open"] - 1.0).shift(-1)
        frames.append(g[["symbol", "timestamp", "realized_return"]])
    if not frames:
        return pd.DataFrame(columns=["symbol", "timestamp", "realized_return"])
    return pd.concat(frames, ignore_index=True)


def evaluate_phase(
    y_true,
    y_pred,
    *,
    phase: str = "evaluation",
    y_prob=None,
    realized_return=None,
    compute_calibration: bool = False,
    labels: tuple[int, ...] = LABELS,
) -> PhaseResult:
    """Compute the full per-phase metric block.

    Args:
        y_true: True labels in ``{-1, 0, +1}``.
        y_pred: Predicted labels in ``{-1, 0, +1}``, same length as ``y_true``.
        phase: Phase name stamped onto the result ("train", "validation", ...).
        y_prob: Optional probability matrix (n, 3) with columns ordered
            ``labels``. Enables log loss, Brier, AUC, calibration, and the
            confidence deciles.
        realized_return: Optional array of next-session realized OC returns
            aligned to ``y_true``. Enables the financial-interpretation
            diagnostics.
        compute_calibration: Whether to compute per-class calibration curves
            (validation only by convention).
        labels: Class order used by every metric.

    Returns:
        A :class:`PhaseResult` for this phase only.

    Raises:
        ValueError: On length mismatch, or if labels outside ``labels`` appear
            in ``y_true`` / ``y_pred``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")

    allowed = set(labels)
    if not set(np.unique(y_true)) <= allowed:
        raise ValueError(f"y_true contains classes outside {list(labels)}")
    if not set(np.unique(y_pred)) <= allowed:
        raise ValueError(f"y_pred contains classes outside {list(labels)}")

    proba = np.asarray(y_prob, dtype=float) if y_prob is not None else None
    if proba is not None:
        if proba.shape != (len(y_true), len(labels)):
            raise ValueError(
                f"y_prob must have shape ({len(y_true)}, {len(labels)})"
            )

    # --- Classification block -------------------------------------------------
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )

    per_class: dict = {}
    for i, c in enumerate(labels):
        per_class[int(c)] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    class_distribution = {int(c): int((y_true == c).sum()) for c in labels}

    # --- ROC-AUC where mathematically appropriate ----------------------------
    roc_auc_macro_ovr: Optional[float] = None
    roc_auc_binary_lens: Optional[float] = None
    if proba is not None:
        if set(int(c) for c in np.unique(y_true)) == set(labels):
            try:
                roc_auc_macro_ovr = float(
                    roc_auc_score(
                        y_true,
                        proba,
                        multi_class="ovr",
                        average="macro",
                        labels=list(labels),
                    )
                )
            except ValueError:
                roc_auc_macro_ovr = None
        lens_mask = y_true != 0
        lens_true = y_true[lens_mask]
        if len(np.unique(lens_true)) == 2:
            pos_idx = list(labels).index(1)
            try:
                roc_auc_binary_lens = float(
                    roc_auc_score(lens_true == 1, proba[lens_mask, pos_idx])
                )
            except ValueError:
                roc_auc_binary_lens = None

    # --- Probability metrics --------------------------------------------------
    logloss: Optional[float] = None
    brier: Optional[float] = None
    calibration: Optional[dict] = None
    if proba is not None:
        logloss = float(log_loss(y_true, proba, labels=list(labels)))
        onehot = label_binarize(y_true, classes=list(labels))
        brier = float(np.mean(np.sum((onehot - proba) ** 2, axis=1)))
        if compute_calibration:
            calibration = _per_class_calibration(y_true, proba, labels)

    # --- Financial-interpretation diagnostics --------------------------------
    financial: Optional[dict] = None
    if realized_return is not None:
        financial = _financial_diagnostics(y_pred, realized_return, proba)

    return PhaseResult(
        phase=phase,
        n_samples=len(y_true),
        accuracy=acc,
        balanced_accuracy=bal_acc,
        per_class=per_class,
        macro_precision=float(np.mean(precision)),
        macro_recall=float(np.mean(recall)),
        macro_f1=float(np.mean(f1)),
        confusion_matrix=cm.tolist(),
        class_distribution=class_distribution,
        roc_auc_macro_ovr=roc_auc_macro_ovr,
        roc_auc_binary_lens=roc_auc_binary_lens,
        log_loss=logloss,
        brier_multi=brier,
        calibration=calibration,
        financial=financial,
    )


def _per_class_calibration(
    y_true: np.ndarray, proba: np.ndarray, labels: tuple[int, ...]
) -> dict:
    """One-vs-rest calibration curve per class (uniform-width bins)."""
    result: dict = {}
    for i, c in enumerate(labels):
        try:
            frac_pos, mean_pred = calibration_curve(
                (y_true == c).astype(int),
                proba[:, i],
                n_bins=N_BINS,
                strategy="uniform",
            )
            result[str(int(c))] = {
                "n_bins": int(len(frac_pos)),
                "mean_predicted": [float(v) for v in mean_pred],
                "fraction_positive": [float(v) for v in frac_pos],
            }
        except ValueError:
            # Degenerate input (e.g. constant probabilities) -> not measurable.
            result[str(int(c))] = None
    return result


def _financial_diagnostics(
    y_pred: np.ndarray,
    realized_return,
    proba: Optional[np.ndarray],
) -> dict:
    """Financial-interpretation diagnostics (NOT strategy returns).

    Drops rows with a NaN realized return (final session per symbol) so the
    means are well-defined.
    """
    rr = np.asarray(realized_return, dtype=float)
    if rr.shape != y_pred.shape:
        raise ValueError("realized_return must align with y_pred")

    valid = ~np.isnan(rr)
    yp = y_pred[valid]
    rrv = rr[valid]
    prb = proba[valid] if proba is not None else None

    mean_by_pred_class: dict = {}
    for c in LABELS:
        sub = rrv[yp == c]
        mean_by_pred_class[str(int(c))] = (
            float(sub.mean()) if len(sub) else None
        )

    bull = rrv[yp == 1]
    bear = rrv[yp == -1]
    mean_bullish = float(bull.mean()) if len(bull) else None
    mean_bearish = float(bear.mean()) if len(bear) else None
    long_short_spread = (
        mean_bullish - mean_bearish
        if mean_bullish is not None and mean_bearish is not None
        else None
    )

    return {
        "mean_realized_return_by_predicted_class": mean_by_pred_class,
        "mean_realized_return_predicted_bullish": mean_bullish,
        "mean_realized_return_predicted_bearish": mean_bearish,
        "long_short_proxy_spread": long_short_spread,
        "confidence_deciles": _confidence_deciles(prb, rrv),
    }


def _confidence_deciles(
    proba: Optional[np.ndarray], realized: np.ndarray
) -> Optional[list]:
    """Deciles of max-class confidence vs mean realized return per bin.

    Rows are ranked by confidence and split into 10 equal-size bins (stable
    ranking, deterministic). Returns ``None`` when probabilities are missing.
    """
    if proba is None:
        return None
    confidence = proba.max(axis=1)
    n = len(confidence)
    order = np.argsort(confidence, kind="stable")
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    bin_idx = np.minimum((ranks * N_BINS) // n, N_BINS - 1)

    rows = []
    for b in range(N_BINS):
        mask = bin_idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "n": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()),
                "mean_realized_return": float(realized[mask].mean()),
            }
        )
    return rows
