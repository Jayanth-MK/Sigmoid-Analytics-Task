"""
Phase 3 — Data Preparation
Cleans raw data, encodes categoricals, builds sklearn preprocessing
pipeline, and returns train/validation splits.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from config import (
    MISSING_THRESHOLD,
    CARDINALITY_THRESHOLD,
    TEST_SIZE,
    RANDOM_STATE,
)


def drop_high_missing(df: pd.DataFrame, threshold: float = MISSING_THRESHOLD) -> pd.DataFrame:
    """Drop columns where the fraction of missing values exceeds threshold."""
    miss_frac = df.isnull().mean()
    keep_cols = miss_frac[miss_frac <= threshold].index.tolist()
    dropped   = df.shape[1] - len(keep_cols)
    print(f"  [drop_high_missing] Dropped {dropped} cols (>{threshold*100:.0f}% missing). "
          f"Remaining: {len(keep_cols)}")
    return df[keep_cols].copy()


def drop_high_cardinality_cats(df: pd.DataFrame,
                                threshold: int = CARDINALITY_THRESHOLD) -> pd.DataFrame:
    """Drop object columns whose unique-value count is >= threshold."""
    cat_cols  = df.select_dtypes(include=["object"]).columns.tolist()
    drop_cols = [c for c in cat_cols if df[c].nunique() >= threshold]
    df        = df.drop(columns=drop_cols)
    print(f"  [drop_high_cardinality_cats] Dropped {len(drop_cols)} high-cardinality cols. "
          f"Remaining: {df.shape[1]}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all remaining object columns (fills NaN with '__missing__')."""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()
    for c in cat_cols:
        df[c] = df[c].fillna("__missing__")
        df[c] = le.fit_transform(df[c].astype(str))
    print(f"  [encode_categoricals] Encoded {len(cat_cols)} categorical columns.")
    return df


def build_preprocessor(numeric_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Return a fitted-ready ColumnTransformer:
      - numeric  : median imputation  + standard scaling
      - categorical: mode imputation
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe,                               numeric_cols),
        ("cat", SimpleImputer(strategy="most_frequent"), cat_cols),
    ])
    return preprocessor


def prepare_data(df: pd.DataFrame,
                 y: np.ndarray) -> tuple:
    """
    Full preparation pipeline.

    Returns
    -------
    X_train, X_val, y_train, y_val : splits
    preprocessor                   : unfitted ColumnTransformer (fitted inside model pipelines)
    feature_cols                   : list of column names used
    numeric_cols, cat_cols         : for downstream use (e.g. feature importance labelling)
    """
    print("\n[DataPreparation] Cleaning data ...")

    df = drop_high_missing(df)
    df = drop_high_cardinality_cats(df)
    df = encode_categoricals(df)

    numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols      = df.select_dtypes(include=["object"]).columns.tolist()
    feature_cols  = numeric_cols + cat_cols

    print(f"  Final feature count — numeric: {len(numeric_cols)}, "
          f"categorical: {len(cat_cols)}, total: {len(feature_cols)}")

    preprocessor = build_preprocessor(numeric_cols, cat_cols)

    X = df[feature_cols]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train size: {len(X_train)} | Val size: {len(X_val)}")

    return X_train, X_val, y_train, y_val, preprocessor, feature_cols, numeric_cols, cat_cols