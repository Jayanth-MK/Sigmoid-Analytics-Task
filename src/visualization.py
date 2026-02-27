"""
Phase 5 — Visualization
Produces the 10-panel results figure and saves it to outputs/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix
)

from config import PALETTE, FIGURE_DPI


def plot_all(results       : dict,
             best_name     : str,
             best_threshold: float,
             threshold_curves: dict,
             y             : np.ndarray,
             y_val         : np.ndarray,
             orig_df       : pd.DataFrame,
             numeric_cols  : list,
             cat_cols      : list,
             output_dir    : Path = Path("C:/Users/Jayanth.MK/OneDrive - Workplace Options/sigmoid task/task/case study-2024/")) -> None:
    """
    Render and save the full 10-panel CRISP-DM results figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    best  = results[best_name]
    probs = best["y_prob"]
    sns.set_style("whitegrid")

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("CRISP-DM Churn Prediction Pipeline — Results",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Class Distribution ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    class_counts = pd.Series(y).value_counts().sort_index()
    bars = ax1.bar(["Retained (0)", "Churned (1)"], class_counts.values,
                   color=["#4CAF50", "#E91E63"], edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, class_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()+300,
                 f"{v:,}\n({v/len(y)*100:.1f}%)",
                 ha="center", fontsize=10, fontweight="bold")
    ax1.set_title("Class Distribution", fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.set_ylim(0, class_counts.max() * 1.2)

    # ── Panel 2: Missingness Distribution ─────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    miss_bins   = pd.cut(orig_df.isnull().mean()*100, bins=[0,20,40,60,80,100],
                         labels=["0-20%","20-40%","40-60%","60-80%","80-100%"])
    miss_counts = miss_bins.value_counts().sort_index()
    ax2.bar(miss_counts.index, miss_counts.values, color=PALETTE, edgecolor="white")
    ax2.set_title("Feature Missingness Distribution", fontweight="bold")
    ax2.set_xlabel("Missing %"); ax2.set_ylabel("# Features")

    # ── Panel 3: Model Comparison ─────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    mnames = list(results.keys())
    aucs   = [results[m]["auc"]      for m in mnames]
    f1s    = [results[m]["f1_churn"] for m in mnames]
    x, w   = np.arange(len(mnames)), 0.35
    ax3.bar(x-w/2, aucs, w, label="ROC-AUC",  color="#2196F3", alpha=0.85)
    ax3.bar(x+w/2, f1s,  w, label="F1-Churn", color="#E91E63", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace(" ","\n") for m in mnames], fontsize=7)
    ax3.axhline(0.5, ls="--", color="gray", alpha=0.5)
    ax3.set_ylim(0, 1.05)
    ax3.set_title("Model Comparison", fontweight="bold")
    ax3.legend(fontsize=8); ax3.set_ylabel("Score")

    # ── Panel 4: ROC Curves ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    for i, (name, res) in enumerate(results.items()):
        if name == "Baseline (Majority)": continue
        fpr, tpr, _ = roc_curve(y_val, res["y_prob"])
        ax4.plot(fpr, tpr, lw=2, color=PALETTE[i%len(PALETTE)],
                 label=f"{name} (AUC={res['auc']:.3f})")
    ax4.plot([0,1],[0,1],"k--",alpha=0.4)
    fpr_b, tpr_b, _ = roc_curve(y_val, probs)
    ax4.fill_between(fpr_b, tpr_b, alpha=0.08, color=PALETTE[0])
    ax4.set_xlabel("False Positive Rate"); ax4.set_ylabel("True Positive Rate")
    ax4.set_title("ROC Curves — All Models", fontweight="bold")
    ax4.legend(fontsize=8, loc="lower right")
    ax4.set_xlim([0,1]); ax4.set_ylim([0,1])

    # ── Panel 5: Precision-Recall ─────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    for i, (name, res) in enumerate(results.items()):
        if name == "Baseline (Majority)": continue
        prec, rec, _ = precision_recall_curve(y_val, res["y_prob"])
        ax5.plot(rec, prec, lw=1.8, color=PALETTE[i%len(PALETTE)],
                 label=f"{name[:12]} (AP={res['ap']:.3f})")
    ax5.axhline(y.mean(), ls="--", color="gray", alpha=0.5, label="Baseline")
    ax5.set_xlabel("Recall"); ax5.set_ylabel("Precision")
    ax5.set_title("Precision-Recall Curves", fontweight="bold")
    ax5.legend(fontsize=6.5, loc="upper right")

    # ── Panel 6: Confusion Matrix ─────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    cm = confusion_matrix(y_val, best["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Retained","Churned"],
                yticklabels=["Retained","Churned"], ax=ax6,
                annot_kws={"size":13,"weight":"bold"})
    ax6.set_title(f"Confusion Matrix\n{best_name}", fontweight="bold")
    ax6.set_ylabel("Actual"); ax6.set_xlabel("Predicted")

    # ── Panel 7: Feature Importances ─────────────────────────
    ax7 = fig.add_subplot(gs[2, 1:])
    try:
        clf_step = best["model"].named_steps["clf"]
        if hasattr(clf_step, "feature_importances_"):
            imp        = clf_step.feature_importances_
            feat_names = ([f"num_{c}" for c in numeric_cols] +
                          [f"cat_{c}" for c in cat_cols])
            if len(imp) == len(feat_names):
                fi_df = pd.DataFrame({"feature": feat_names, "importance": imp})
                fi_df = fi_df.nlargest(20, "importance")
                ax7.barh(fi_df["feature"], fi_df["importance"],
                         color="#2196F3", alpha=0.8)
                ax7.set_xlabel("Importance")
                ax7.set_title(f"Top-20 Feature Importances\n{best_name}", fontweight="bold")
                ax7.invert_yaxis()
            else:
                ax7.text(0.5, 0.5, "Feature count mismatch",
                         ha="center", transform=ax7.transAxes)
        else:
            ax7.text(0.5, 0.5, "Feature importance not available",
                     ha="center", va="center", transform=ax7.transAxes, fontsize=11)
            ax7.set_title("Feature Importances", fontweight="bold")
    except Exception as e:
        ax7.text(0.5, 0.5, f"Error:\n{e}",
                 ha="center", va="center", transform=ax7.transAxes, fontsize=9)

    # ── Panel 8: Score Distribution ───────────────────────────
    ax8 = fig.add_subplot(gs[3, 0])
    ax8.hist(probs[y_val==0], bins=40, alpha=0.6, color="#4CAF50",
             label="Retained", density=True)
    ax8.hist(probs[y_val==1], bins=40, alpha=0.6, color="#E91E63",
             label="Churned",  density=True)
    ax8.axvline(0.5, ls="--", color="black", alpha=0.6, label="Default threshold")
    ax8.set_xlabel("Predicted Churn Probability"); ax8.set_ylabel("Density")
    ax8.set_title("Score Distribution by Class", fontweight="bold")
    ax8.legend()

    # ── Panel 9: Threshold Tuning ─────────────────────────────
    ax9 = fig.add_subplot(gs[3, 1])
    tc = threshold_curves
    ax9.plot(tc["thresholds"], tc["f1"],        color="#2196F3", lw=2, label="F1")
    ax9.plot(tc["thresholds"], tc["recall"],    color="#4CAF50", lw=2, label="Recall")
    ax9.plot(tc["thresholds"], tc["precision"], color="#E91E63", lw=2, label="Precision")
    ax9.axvline(best_threshold, ls="--", color="black",
                label=f"Best={best_threshold:.2f}")
    ax9.set_xlabel("Decision Threshold"); ax9.set_ylabel("Score")
    ax9.set_title("Threshold Tuning", fontweight="bold")
    ax9.legend(fontsize=8)

    # ── Panel 10: Leaderboard Table ───────────────────────────
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.axis("off")
    table_data = [
        [name[:22], f"{res['auc']:.4f}", f"{res['ap']:.4f}",
         f"{res['f1_churn']:.4f}", f"{res['time']:.1f}s"]
        for name, res in results.items()
    ]
    tbl = ax10.table(cellText=table_data,
                     colLabels=["Model","ROC-AUC","Avg Prec","F1-Churn","Time"],
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1.1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2196F3")
            cell.set_text_props(color="white", fontweight="bold")
        elif r > 0 and table_data[r-1][0].strip() == best_name[:22].strip():
            cell.set_facecolor("#E8F5E9")
    ax10.set_title("Model Leaderboard", fontweight="bold", pad=12)

    out_path = "crisp_dm_results.png"
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")