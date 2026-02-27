"""
Phase 2 — Data Understanding
Loads raw CSVs, maps labels, and prints a basic data profile.
"""

import pandas as pd
import numpy as np
from config import TRAIN_FEATURES_FILE, TRAIN_LABELS_FILE


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load features and labels from disk.

    Returns
    -------
    df : pd.DataFrame   — raw feature matrix (50000 x 230)
    y  : np.ndarray     — binary labels  0=retained  1=churned
    """
    print("\n[DataLoader] Loading data ...")
    df     = pd.read_csv(TRAIN_FEATURES_FILE, low_memory=False)
    labels = pd.read_csv(TRAIN_LABELS_FILE)

    # Map -1 -> 0 (retained), 1 -> 1 (churned)
    y = labels["Label"].map({-1: 0, 1: 1}).values

    print(f"  Features shape : {df.shape}")
    print(f"  Labels shape   : {labels.shape}")
    print(f"  Churn rate     : {y.mean()*100:.2f}%  "
          f"(Retained={int((y==0).sum())}  Churned={int((y==1).sum())})")
    return df, y


def profile_data(df: pd.DataFrame) -> dict:
    """
    Return a dictionary summarising key data quality metrics.
    Used by the report and visualization modules.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = df.select_dtypes(include=["object"]).columns.tolist()
    miss_pct     = df.isnull().mean() * 100

    profile = {
        "n_rows"          : df.shape[0],
        "n_cols"          : df.shape[1],
        "numeric_cols"    : numeric_cols,
        "cat_cols"        : cat_cols,
        "missing_pct_col" : miss_pct,
        "overall_missing" : miss_pct.mean(),
    }

    print(f"  Numeric features  : {len(numeric_cols)}")
    print(f"  Categorical feats : {len(cat_cols)}")
    print(f"  Overall missing % : {profile['overall_missing']:.1f}%")
    return profile