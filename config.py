from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────


TRAIN_FEATURES_FILE = "train.csv"
TRAIN_LABELS_FILE   = "train_churn_labels.csv"

# ── Data Preparation ──────────────────────────────────────────
MISSING_THRESHOLD      = 0.80   # drop columns with > 80% missing
CARDINALITY_THRESHOLD  = 50     # drop categoricals with >= 50 unique values
TEST_SIZE              = 0.20
RANDOM_STATE           = 42

# ── Model Hyperparameters ─────────────────────────────────────
LOGISTIC_REGRESSION_PARAMS = {
    "max_iter"     : 1000,
    "C"            : 0.1,
    "class_weight" : "balanced",
    "random_state" : RANDOM_STATE,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators" : 200,
    "max_depth"    : 8,
    "class_weight" : "balanced",
    "random_state" : RANDOM_STATE,
    "n_jobs"       : -1,
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators"  : 150,
    "max_depth"     : 4,
    "learning_rate" : 0.05,
    "random_state"  : RANDOM_STATE,
}

RF_BALANCED_SUBSAMPLE_PARAMS = {
    "n_estimators" : 200,
    "max_depth"    : 10,
    "class_weight" : "balanced_subsample",
    "random_state" : 1,
    "n_jobs"       : -1,
}

# ── Threshold Tuning ──────────────────────────────────────────
THRESHOLD_MIN   = 0.10
THRESHOLD_MAX   = 0.90
THRESHOLD_STEPS = 80

# ── Visualization ─────────────────────────────────────────────
FIGURE_DPI    = 150
PALETTE       = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]