"""Shared constants for the TradeSense ML pipeline.

These values are modeling constants, not tuned hyperparameters. They are
documented in the V1 design (docs/ml_pipeline.md) and must not be optimized
against the final test set.
"""

# Max feature lookback is 20 sessions, but volatility_20d needs 20 log returns,
# i.e. 21 price observations -> the first feature-complete row per symbol is at
# row index 21. The first WARMUP_ROWS rows of each symbol are dropped at dataset
# construction (a fixed structural property, causally safe).
WARMUP_ROWS = 21

# Primary target epsilon (neutral-zone half-width). Default 0.001 = 0.10%.
DEFAULT_EPSILON = 0.001

# Secondary target horizon (sessions) for forward_return / realized vol.
DEFAULT_HORIZON = 5

# Global reproducibility seed, passed as random_state to every estimator.
SEED = 42

# Post-warm-up NaN guard: fail loudly if any feature's NaN fraction exceeds this.
DEFAULT_MAX_NAN_FRACTION = 0.02

# Name of the target column in the ML panel.
TARGET_COLUMN = "target_direction"
