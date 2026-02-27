"""
CRISP-DM Churn Prediction — Entry Point
========================================
Run with:
    python main.py

All six CRISP-DM phases are orchestrated here.
Each phase delegates to a dedicated module in src/.
"""

from src.data_loader       import load_data, profile_data
from src.data_preparation  import prepare_data
from src.feature_engineering import engineer_features
from src.modeling          import get_model_definitions, train_all_models, select_best_model
from src.evaluation        import tune_threshold, print_classification_report, save_artifacts
from src.visualization     import plot_all

REPORT_LINES = []

def log(msg=""):
    print(msg)
    REPORT_LINES.append(str(msg))


def main():
    # ── Phase 1: Business Understanding ───────────────────────
    log("=" * 65)
    log("PHASE 1 — BUSINESS UNDERSTANDING")
    log("=" * 65)
    log("Objective  : Predict customer churn (Label 1=churned, -1=retained)")
    log("Metric     : ROC-AUC (primary) + F1-Churn (secondary)")
    log("Target     : ROC-AUC > 0.75 | F1-Churn > 0.40")

    # ── Phase 2: Data Understanding ───────────────────────────
    log("\n" + "=" * 65)
    log("PHASE 2 — DATA UNDERSTANDING")
    log("=" * 65)
    df, y        = load_data()
    orig_df      = df.copy()         # keep raw copy for visualisation
    data_profile = profile_data(df)

    # ── Phase 3: Data Preparation ─────────────────────────────
    log("\n" + "=" * 65)
    log("PHASE 3 — DATA PREPARATION")
    log("=" * 65)
    df = engineer_features(df)       # extendable hook
    (X_train, X_val, y_train, y_val,
     preprocessor, feature_cols,
     numeric_cols, cat_cols)         = prepare_data(df, y)

    # ── Phase 4: Modeling ─────────────────────────────────────
    log("\n" + "=" * 65)
    log("PHASE 4 — MODELING")
    log("=" * 65)
    models    = get_model_definitions(preprocessor)
    results   = train_all_models(models, X_train, y_train, X_val, y_val)
    best_name, best_result = select_best_model(results, metric="auc")

    # ── Phase 5: Evaluation ───────────────────────────────────
    log("\n" + "=" * 65)
    log("PHASE 5 — EVALUATION")
    log("=" * 65)
    report = print_classification_report(y_val, best_result["y_pred"], best_name)
    REPORT_LINES.append(report)

    best_threshold, threshold_curves = tune_threshold(y_val, best_result["y_prob"])

    plot_all(
        results          = results,
        best_name        = best_name,
        best_threshold   = best_threshold,
        threshold_curves = threshold_curves,
        y                = y,
        y_val            = y_val,
        orig_df          = orig_df,
        numeric_cols     = numeric_cols,
        cat_cols         = cat_cols,
    )

    # ── Phase 6: Deployment ───────────────────────────────────
    log("\n" + "=" * 65)
    log("PHASE 6 — DEPLOYMENT")
    log("=" * 65)
    save_artifacts(
        best_name        = best_name,
        best_result      = best_result,
        best_threshold   = best_threshold,
        feature_cols     = feature_cols,
        y                = y,
        X_train          = X_train,
        X_val            = X_val,
        y_val            = y_val,
        report_lines     = REPORT_LINES,
    )

    # ── Summary ───────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("PIPELINE COMPLETE")
    log(f"  Best Model    : {best_name}")
    log(f"  ROC-AUC       : {best_result['auc']:.4f}")
    log(f"  F1-Churn      : {best_result['f1_churn']:.4f}")
    log(f"  Opt Threshold : {best_threshold:.2f}")
    log(f"  Outputs in    : {('C:/Users/Jayanth.MK/OneDrive - Workplace Options/sigmoid task/task/case study-2024/')}")
    log("=" * 65)


if __name__ == "__main__":
    main()