"""Leakage-safe preprocessing for the ML pipeline.

All fitted transforms live inside an sklearn ``Pipeline`` and are learned ONLY
through ``.fit(X_train, y_train)`` — never on validation or test data. There is
no code path that fits an imputer, scaler, selector, or any other transform
outside a pipeline fitted on training data.

Policy (V1, docs/ml_pipeline.md):
    - NaN handling happens at dataset construction (warm-up + remaining-NaN
      rows are dropped; see ``dataset.py``). An optional train-fitted
      ``SimpleImputer`` exists for completeness but is OFF by default: missing
      SPY market features are a data gap, not missing-at-random, and global
      statistics would mislead.
    - Scaling: ``StandardScaler`` for linear models only. Tree-based models are
      split-based and scale-invariant, so they get NO scaler.
"""

from __future__ import annotations

from typing import Union

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

#: Supported model families. "linear" models are scale-sensitive; "tree"
#: models are split-based and scale-invariant.
MODEL_FAMILIES = ("linear", "tree")

#: Human-readable preprocessing policy labels recorded in experiment results.
PREPROCESSING_SCALER = "scaler"
PREPROCESSING_NONE = "none"


def requires_scaling(model_family: str) -> bool:
    """Whether the model family is scale-sensitive (needs a StandardScaler)."""
    if model_family not in MODEL_FAMILIES:
        raise ValueError(
            f"Unknown model family '{model_family}'; expected one of {MODEL_FAMILIES}"
        )
    return model_family == "linear"


def preprocessing_policy(model_family: str) -> str:
    """Policy label for the result record."""
    if model_family not in MODEL_FAMILIES:
        raise ValueError(
            f"Unknown model family '{model_family}'; expected one of {MODEL_FAMILIES}"
        )
    return PREPROCESSING_SCALER if requires_scaling(model_family) else PREPROCESSING_NONE


def build_preprocessing_pipeline(
    model_family: str,
    *,
    impute: bool = False,
    strategy: str = "mean",
) -> Pipeline:
    """Build the preprocessing-only pipeline for a model family.

    Args:
        model_family: "linear" (scaled) or "tree" (no scaling).
        impute: Whether to include a train-fitted ``SimpleImputer``
            (default off; dataset construction already drops NaN rows).
        strategy: ``SimpleImputer`` strategy when ``impute=True``.

    Returns:
        A sklearn ``Pipeline``. Fit it on training data only, then
        ``transform``/``predict`` on validation or test data. An empty
        pipeline (tree family, no imputation) is a valid identity transform.
    """
    if model_family not in MODEL_FAMILIES:
        raise ValueError(
            f"Unknown model family '{model_family}'; expected one of {MODEL_FAMILIES}"
        )

    steps: list[tuple[str, Union[SimpleImputer, StandardScaler, FunctionTransformer]]] = []
    if impute:
        steps.append(("imputer", SimpleImputer(strategy=strategy)))
    if requires_scaling(model_family):
        steps.append(("scaler", StandardScaler()))
    elif not impute:
        # sklearn 1.9 forbids empty pipelines; an identity passthrough keeps
        # the same compose-into-a-larger-pipeline pattern for model fitting.
        steps.append(("passthrough", FunctionTransformer()))
    return Pipeline(steps)
