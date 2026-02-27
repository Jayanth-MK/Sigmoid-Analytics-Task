"""
Phase 4 — Modeling
Builds, trains, and returns all model pipelines.
"""

import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score
)

from config import (
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    GRADIENT_BOOSTING_PARAMS,
    RF_BALANCED_SUBSAMPLE_PARAMS,
)


def get_model_definitions(preprocessor) -> dict:
    """
    Return a dictionary of {model_name: sklearn Pipeline}.
    Add or remove models here without touching any other file.
    """
    def wrap(clf):
        return Pipeline([("prep", preprocessor), ("clf", clf)])

    return {
        "Baseline (Majority)"   : DummyClassifier(strategy="most_frequent"),
        "Logistic Regression"   : wrap(LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)),
        "Random Forest"         : wrap(RandomForestClassifier(**RANDOM_FOREST_PARAMS)),
        "Gradient Boosting"     : wrap(GradientBoostingClassifier(**GRADIENT_BOOSTING_PARAMS)),
        "RF Balanced Subsample" : wrap(RandomForestClassifier(**RF_BALANCED_SUBSAMPLE_PARAMS)),
    }


def train_all_models(models: dict,
                     X_train, y_train,
                     X_val,   y_val) -> dict:
    """
    Train every model, evaluate on validation set, and return results dict.

    Returns
    -------
    results : {
        model_name: {
            "model"    : fitted pipeline,
            "y_pred"   : np.ndarray,
            "y_prob"   : np.ndarray,
            "auc"      : float,
            "ap"       : float,
            "f1_churn" : float,
            "time"     : float (seconds),
        }
    }
    """
    print("\n[Modeling] Training models ...")
    results = {}

    for name, model in models.items():
        t0 = time.time()
        print(f"  Training: {name} ...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if name == "Baseline (Majority)":
            y_prob = np.zeros(len(y_val))
        else:
            y_prob = model.predict_proba(X_val)[:, 1]

        auc     = roc_auc_score(y_val, y_prob)      if y_prob.sum() > 0 else 0.5
        ap      = average_precision_score(y_val, y_prob) if y_prob.sum() > 0 else 0.0
        f1c     = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        elapsed = time.time() - t0

        results[name] = {
            "model"    : model,
            "y_pred"   : y_pred,
            "y_prob"   : y_prob,
            "auc"      : auc,
            "ap"       : ap,
            "f1_churn" : f1c,
            "time"     : elapsed,
        }
        print(f"    ROC-AUC={auc:.4f}  AP={ap:.4f}  F1-Churn={f1c:.4f}  [{elapsed:.1f}s]")

    return results


def select_best_model(results: dict,
                      metric: str = "auc") -> tuple[str, dict]:
    """
    Return the name and result dict of the best non-baseline model.

    Parameters
    ----------
    metric : key in each result dict to rank by (default "auc")
    """
    candidates = {k: v for k, v in results.items() if k != "Baseline (Majority)"}
    best_name  = max(candidates, key=lambda k: candidates[k][metric])
    print(f"\n  Best model: {best_name}  "
          f"(ROC-AUC={results[best_name]['auc']:.4f})")
    return best_name, results[best_name]