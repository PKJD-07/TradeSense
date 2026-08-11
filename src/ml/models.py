"""Candidate ML models: a fixed, documented registry.

Fixed, documented hyperparameters. **No hyperparameter tuning in V1.** No
XGBoost/LightGBM. Each model carries its model family ("linear" | "tree"),
which drives the preprocessing policy (StandardScaler for linear models only;
tree models are split-based and scale-invariant).

Fixed configurations (docs/ml_pipeline.md, §7):

| Name | Config | Family | Notes |
|---|---|---|---|
| logistic_regression | `C=1.0`, `max_iter=1000`, `random_state=SEED` | linear | Interpretable linear baseline; well-behaved probabilities |
| random_forest | `max_depth=6`, `min_samples_leaf=20`, `n_estimators=100`, `random_state=SEED` | tree | Nonlinear; depth/leaf bounds limit overfit |
| gradient_boosting | `n_estimators=200`, `max_depth=3`, `learning_rate=0.1`, `n_iter_no_change=None`, `random_state=SEED` | tree | No early stopping (design `early_stopping=False`; sklearn 1.9 replaced the flag with `n_iter_no_change`). sklearn's internal early-stopping split is random and violates temporal discipline |

``make_model_pipeline`` composes the train-fitted preprocessing with the model
into a single sklearn ``Pipeline`` — the only way a model is fit in
``experiment.py``.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.ml.constants import SEED
from src.ml.preprocessing import MODEL_FAMILIES, build_preprocessing_pipeline

#: Registry of model names accepted by ``build_model``.
MODEL_NAMES = ("logistic_regression", "random_forest", "gradient_boosting")

#: Fixed, documented V1 hyperparameters per model.
MODEL_CONFIGS: dict[str, dict] = {
    "logistic_regression": {
        "C": 1.0,
        # Convergence safety: scaled features converge fast; this removes
        # non-convergence warnings while leaving C=1.0 as the only tuned knob.
        "max_iter": 1000,
        "random_state": SEED,
    },
    "random_forest": {
        "max_depth": 6,
        "min_samples_leaf": 20,
        "n_estimators": 100,
        "random_state": SEED,
    },
    "gradient_boosting": {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.1,
        # No early stopping (design: early_stopping=False). sklearn 1.9 dropped
        # the early_stopping flag; the equivalent is n_iter_no_change=None.
        # sklearn's internal early-stopping split is random and would violate
        # temporal discipline.
        "n_iter_no_change": None,
        "random_state": SEED,
    },
}

#: Model family per model name; drives the preprocessing policy.
MODEL_FAMILY: dict[str, str] = {
    "logistic_regression": "linear",
    "random_forest": "tree",
    "gradient_boosting": "tree",
}

#: Models with a ``class_weight`` parameter (GBM uses per-row ``sample_weight``
#: instead, handled in the experiment layer).
CLASS_WEIGHT_SUPPORTED = ("logistic_regression", "random_forest")


def model_family(name: str) -> str:
    """Preprocessing family ("linear" | "tree") for a model name.

    Raises:
        ValueError: If ``name`` is not a known model.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{name}'; expected one of {MODEL_NAMES}")
    return MODEL_FAMILY[name]


def build_model(name: str, *, class_weight=None):
    """Build a fresh, unfitted model with the fixed V1 config.

    Args:
        name: One of ``MODEL_NAMES``.
        class_weight: Optional ``class_weight`` override for
            ``logistic_regression`` / ``random_forest`` (``None`` = unweighted,
            the V1 default). Gradient boosting has no ``class_weight``
            parameter; per-class imbalance there is handled via per-row
            ``sample_weight`` in ``experiment.py``.

    Returns:
        An unfitted sklearn estimator.

    Raises:
        ValueError: If ``name`` is unknown, or ``class_weight`` is passed for
            a model that does not support it.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{name}'; expected one of {MODEL_NAMES}")
    if class_weight is not None and name not in CLASS_WEIGHT_SUPPORTED:
        raise ValueError(
            f"'{name}' has no class_weight parameter; pass per-row "
            "sample_weight instead"
        )

    config = dict(MODEL_CONFIGS[name])
    if name == "logistic_regression":
        return LogisticRegression(class_weight=class_weight, **config)
    if name == "random_forest":
        return RandomForestClassifier(class_weight=class_weight, **config)
    return GradientBoostingClassifier(**config)


def make_model_pipeline(
    name: str,
    *,
    impute: bool = False,
    strategy: str = "mean",
    class_weight=None,
) -> Pipeline:
    """Compose train-fitted preprocessing with a model into one Pipeline.

    The returned ``Pipeline`` is fit ONLY through ``.fit(X_train, y_train)``;
    validation/test data never contributes a fitted parameter. For the tree
    family the preprocessing is an identity passthrough, so the pipeline is
    just the model step plus a no-op.

    Args:
        name: Model name (one of ``MODEL_NAMES``).
        impute: Whether to include a train-fitted ``SimpleImputer`` (off by
            default; dataset construction already drops NaN rows).
        strategy: ``SimpleImputer`` strategy when ``impute=True``.
        class_weight: Passed through to ``build_model``.

    Returns:
        A single sklearn ``Pipeline`` ending in the model step.
    """
    model = build_model(name, class_weight=class_weight)
    pre = build_preprocessing_pipeline(
        model_family(name), impute=impute, strategy=strategy
    )
    return Pipeline([*pre.steps, ("model", model)])
