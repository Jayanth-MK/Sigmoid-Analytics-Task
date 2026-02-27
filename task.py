#!/usr/bin/env python
# coding: utf-8

# In[3]:


# ============================================================
# End-to-end implementation for the interview task
# - Loads train features + labels (50k rows)
# - Aligns/merges safely
# - Cleans columns (drops 100% missing + constant cols)
# - Builds preprocessing for numeric + categorical
# - Trains model (tries LightGBM -> XGBoost -> HistGB fallback)
# - Evaluates with PR-AUC, ROC-AUC, LogLoss
# - Finds best threshold on validation (max F1 by default)
# - Retrains on full data and saves artifacts
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss,
    precision_recall_curve,
    f1_score,
    classification_report,
    confusion_matrix
)

RANDOM_STATE = 42

# -----------------------------
# Paths (edit if needed)
# -----------------------------
FEATURES_PATH = "train (6).csv"
LABELS_PATH   = "train_churn_labels.csv"

OUT_DIR = "model_out"
os.makedirs(OUT_DIR, exist_ok=True)


# In[4]:


# -----------------------------
# Helper classes
# -----------------------------
class ToDense(BaseEstimator, TransformerMixin):
    """Convert sparse matrix to dense (required for HistGradientBoostingClassifier)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


# -----------------------------
# Helper functions
# -----------------------------
def load_and_align(features_path: str, labels_path: str):
    X = pd.read_csv(features_path)
    y_df = pd.read_csv(labels_path)

    possible_label_cols = ["label", "Label", "target", "Target", "y", "Y"]
    label_col = None
    for c in possible_label_cols:
        if c in y_df.columns:
            label_col = c
            break
    if label_col is None:
        label_col = y_df.columns[0]

    y_raw = y_df[label_col].copy()

    if len(X) != len(y_raw):
        raise ValueError(f"Row mismatch: X={len(X)} but y={len(y_raw)}. Need ID-based join.")

    y = y_raw.replace({-1: 0, 1: 1}).astype(int)
    return X, y


def drop_bad_columns(df: pd.DataFrame):
    df2 = df.copy()
    all_missing = [c for c in df2.columns if df2[c].isna().all()]
    df2.drop(columns=all_missing, inplace=True)

    constant = [c for c in df2.columns if df2[c].nunique(dropna=True) <= 1]
    df2.drop(columns=constant, inplace=True)

    dropped = {"all_missing": all_missing, "constant": constant}
    return df2, dropped


def add_missingness_features(df: pd.DataFrame):
    df2 = df.copy()
    miss_count = df2.isna().sum(axis=1)
    miss_ratio = miss_count / max(df2.shape[1], 1)
    df2["__missing_count__"] = miss_count
    df2["__missing_ratio__"] = miss_ratio
    return df2


def get_feature_types(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]
    return num_cols, cat_cols


def choose_model():
    # --- LightGBM ---
    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            objective="binary",
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        return "lightgbm", model
    except Exception:
        pass

    # --- XGBoost ---
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        return "xgboost", model
    except Exception:
        pass

    # --- Fallback: HistGradientBoosting ---
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_depth=6,
        max_iter=600,
        random_state=RANDOM_STATE
    )
    return "hist_gb", model


def build_pipeline(X: pd.DataFrame, base_model, model_name: str):
    """
    Key fix:
    - HistGradientBoostingClassifier requires dense input.
    - So we add ToDense() right after preprocessing only for hist_gb.
    """
    num_cols, cat_cols = get_feature_types(X)

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop"
    )

    steps = [("preprocess", preprocessor)]

    if model_name == "hist_gb":
        steps.append(("todense", ToDense()))  # ✅ critical fix

    steps.append(("model", base_model))

    return Pipeline(steps=steps)


def find_best_threshold(y_true, y_proba, method="max_f1"):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return 0.5

    if method == "max_f1":
        f1s = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
        best_idx = int(np.argmax(f1s))
        return float(thresholds[best_idx])

    return 0.5


# -----------------------------
# Main training flow
# -----------------------------
def main():
    print("1) Loading data...")
    X_raw, y = load_and_align(FEATURES_PATH, LABELS_PATH)

    print("2) Cleaning columns...")
    X_clean, dropped = drop_bad_columns(X_raw)

    print("3) Adding missingness features...")
    X_feat = add_missingness_features(X_clean)

    print("4) Train/validation split (stratified)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_feat, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pos_rate = y_train.mean()
    print(f"   Positive rate (train): {pos_rate:.4f} (imbalance expected)")

    print("5) Model selection...")
    model_name, base_model = choose_model()
    print(f"   Using model: {model_name}")

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    if pos == 0:
        raise ValueError("No positive samples in training split.")
    scale_pos_weight = neg / pos
    print(f"   scale_pos_weight ~ {scale_pos_weight:.2f}")

    pipe = build_pipeline(X_train, base_model, model_name)

    print("6) Hyperparameter search (lightweight)...")
    if model_name == "lightgbm":
        param_dist = {
            "model__num_leaves": [31, 63, 127],
            "model__learning_rate": [0.01, 0.03, 0.06],
            "model__n_estimators": [800, 1500, 2500],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.7, 0.8, 0.9],
            "model__reg_lambda": [0.0, 1.0, 5.0],
        }
    elif model_name == "xgboost":
        param_dist = {
            "model__max_depth": [4, 6, 8],
            "model__learning_rate": [0.01, 0.03, 0.06],
            "model__n_estimators": [800, 1500, 2500],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.7, 0.8, 0.9],
            "model__reg_lambda": [0.0, 1.0, 5.0],
        }
    else:
        param_dist = {
            "model__learning_rate": [0.03, 0.06, 0.1],
            "model__max_depth": [4, 6, 8],
            "model__max_iter": [300, 600, 900],
        }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=min(15, sum(len(v) for v in param_dist.values())),
        scoring="average_precision",
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        error_score="raise"   # ✅ so you see the REAL root error immediately
    )

    # sample weights (try to pass them; if estimator doesn't support, fall back)
    sample_weight = np.where(y_train.values == 1, scale_pos_weight, 1.0)

    try:
        search.fit(X_train, y_train, model__sample_weight=sample_weight)
    except TypeError:
        search.fit(X_train, y_train)

    best_pipe = search.best_estimator_
    print("Best CV PR-AUC:", search.best_score_)
    print("Best params:", search.best_params_)

    print("7) Validation evaluation...")
    if hasattr(best_pipe, "predict_proba"):
        y_val_proba = best_pipe.predict_proba(X_val)[:, 1]
    else:
        # fallback; most should have predict_proba
        y_val_proba = best_pipe.predict(X_val)

    pr_auc = average_precision_score(y_val, y_val_proba)
    roc_auc = roc_auc_score(y_val, y_val_proba)
    ll = log_loss(y_val, y_val_proba)

    print(f"   PR-AUC:  {pr_auc:.6f}")
    print(f"   ROC-AUC: {roc_auc:.6f}")
    print(f"   LogLoss: {ll:.6f}")

    thr = find_best_threshold(y_val, y_val_proba, method="max_f1")
    print(f"   Best threshold (max F1): {thr:.4f}")

    y_val_pred = (y_val_proba >= thr).astype(int)

    print("\nClassification report (val):")
    print(classification_report(y_val, y_val_pred, digits=4))
    print("Confusion matrix (val):")
    print(confusion_matrix(y_val, y_val_pred))

    print("8) Retrain best model on full data and save artifacts...")
    X_full = X_feat
    y_full = y

    neg_full = int((y_full == 0).sum())
    pos_full = int((y_full == 1).sum())
    spw_full = neg_full / max(pos_full, 1)
    sample_weight_full = np.where(y_full.values == 1, spw_full, 1.0)

    try:
        best_pipe.fit(X_full, y_full, model__sample_weight=sample_weight_full)
    except TypeError:
        best_pipe.fit(X_full, y_full)

    model_path = os.path.join(OUT_DIR, f"final_model_{model_name}.joblib")
    joblib.dump(best_pipe, model_path)

    meta = {
        "model_name": model_name,
        "dropped_columns": dropped,
        "best_params": search.best_params_,
        "val_metrics": {
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "log_loss": float(ll),
            "threshold_max_f1": float(thr)
        }
    }

    meta_path = os.path.join(OUT_DIR, "training_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nDONE ✅")
    print("Saved model:", model_path)
    print("Saved metadata:", meta_path)


if __name__ == "__main__":
    main()

