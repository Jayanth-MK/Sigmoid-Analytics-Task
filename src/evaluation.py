"""
Phase 5 — Evaluation
Threshold tuning, classification report, and deployment artifact saving.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    f1_score, precision_score, recall_score,
)

from config import (
    THRESHOLD_MIN,
    THRESHOLD_MAX,
    THRESHOLD_STEPS
)


def tune_threshold(y_val: np.ndarray,
                   y_prob: np.ndarray) -> tuple[float, dict]:
    """
    Sweep decision thresholds and find the one that maximises F1-churn.

    Returns
    -------
    best_threshold : float
    curves         : dict with keys "thresholds", "f1", "recall", "precision"
    """
    thresholds   = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS)
    f1s, recs, precs = [], [], []

    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_val, yp, pos_label=1, zero_division=0))
        recs.append(recall_score(y_val, yp, pos_label=1, zero_division=0))
        precs.append(precision_score(y_val, yp, pos_label=1, zero_division=0))

    best_idx       = int(np.argmax(f1s))
    best_threshold = float(thresholds[best_idx])

    print(f"  [Evaluation] Optimal threshold (max F1): {best_threshold:.2f}")

    curves = {
        "thresholds" : thresholds,
        "f1"         : np.array(f1s),
        "recall"     : np.array(recs),
        "precision"  : np.array(precs),
    }
    return best_threshold, curves


def print_classification_report(y_val: np.ndarray,
                                 y_pred: np.ndarray,
                                 model_name: str) -> str:
    """Print and return the sklearn classification report string."""
    report = classification_report(
        y_val, y_pred, target_names=["Retained", "Churned"]
    )
    print(f"\n[Evaluation] Classification Report — {model_name}")
    print(report)
    return report


def save_artifacts(best_name: str,
                   best_result: dict,
                   best_threshold: float,
                   feature_cols: list,
                   y: np.ndarray,
                   X_train, X_val,
                   y_val: np.ndarray,
                   report_lines: list,
                   output_dir: Path = Path("C:/Users/Jayanth.MK/OneDrive - Workplace Options/sigmoid task/task/case study-2024/")) -> None:
    """
    Save all deployment artifacts to output_dir:
      - best_model.pkl
      - model_metadata.json
      - crisp_dm_report.txt
      - validation_predictions.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Trained model
    model_path = "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_result["model"], f)
    print(f"  Saved: {model_path}")

    # 2. Metadata JSON
    metadata = {
        "best_model"        : best_name,
        "roc_auc"           : round(float(best_result["auc"]), 4),
        "avg_precision"     : round(float(best_result["ap"]), 4),
        "f1_churn"          : round(float(best_result["f1_churn"]), 4),
        "optimal_threshold" : round(best_threshold, 4),
        "features_used"     : feature_cols,
        "n_features"        : len(feature_cols),
        "train_size"        : int(len(X_train)),
        "val_size"          : int(len(X_val)),
        "class_distribution": {
            "retained": int((y == 0).sum()),
            "churned" : int((y == 1).sum()),
        },
        "churn_rate_pct"    : round(float(y.mean() * 100), 2),
    }
    meta_path =  "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {meta_path}")

    # 3. Text report
    report_path =  "crisp_dm_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    # 4. Validation predictions CSV
    pred_df = pd.DataFrame({
        "churn_probability" : best_result["y_prob"],
        "predicted_label"   : (best_result["y_prob"] >= best_threshold).astype(int),
        "actual_label"      : y_val,
    })
    pred_path =  "validation_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  Saved: {pred_path}")