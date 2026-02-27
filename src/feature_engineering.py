"""
Feature Engineering (extendable module).
Currently a pass-through — add domain-specific transformations here
without touching data_preparation.py or modeling.py.

Examples of what to add:
  - ratio features  (e.g. calls_per_day = total_calls / tenure)
  - log transforms  for heavy-tailed numeric columns
  - interaction terms
  - time-based aggregations
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering transformations to a cleaned DataFrame.
    Called BEFORE build_preprocessor in the pipeline.

    Parameters
    ----------
    df : cleaned DataFrame (after missing / cardinality drops, before encoding)

    Returns
    -------
    df : DataFrame with new / transformed columns added
    """
    # ── Example: log-transform skewed numeric columns ─────────
    # numeric_cols = df.select_dtypes(include=[np.number]).columns
    # for col in numeric_cols:
    #     if df[col].skew() > 2:
    #         df[col] = np.log1p(df[col].clip(lower=0))

    # ── Example: ratio feature ─────────────────────────────────
    # if "Var6" in df.columns and "Var7" in df.columns:
    #     df["ratio_v6_v7"] = df["Var6"] / (df["Var7"] + 1)

    print("  [FeatureEngineering] No custom features applied (pass-through).")
    return df