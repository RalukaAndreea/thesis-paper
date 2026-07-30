#!/usr/bin/env python3
"""
Generate a combined ROC curve showing Baseline RF, XGBoost, and Extended RF
on the same plot, with their respective CV-optimal thresholds marked.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from pipeline import train_pathogenicity_model as train_baseline_rf
from pipeline_xgboost import load_and_split as load_xgb_data
from explainability_extended import train_refined_pathogenicity_model as train_extended_rf

MODELS_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "PLOTS")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Dark Theme Style ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#e0e0e0",
    "axes.labelcolor":  "#e0e0e0",
    "text.color":       "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "grid.color":       "#2a2a4a",
    "grid.alpha":       0.4,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.labelsize":   12,
})

COLORS = {
    "baseline_rf": "#7f5af0",  # Purple
    "xgboost":     "#f9c74f",  # Yellow
    "extended_rf": "#2cb67d",  # Green
}

def main():
    print("=" * 60)
    print(" Generating Combined ROC Curve")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(10, 8))

    # ─────────────────────────────────────────────────────────────
    # 1. Baseline Random Forest (8 Features)
    # ─────────────────────────────────────────────────────────────
    print("▶ Loading Baseline Random Forest...")
    # Retrain quickly to get exact test set and threshold, or load from disk.
    # Training is fast enough, let's just train to be safe and consistent.
    rf_model, _, rf_X_test, rf_y_test, rf_cv, _ = train_baseline_rf()
    rf_thresh = rf_cv.get("optimal_threshold", 0.5)
    
    rf_proba = rf_model.predict_proba(rf_X_test)[:, 1]
    fpr_rf, tpr_rf, th_rf = roc_curve(rf_y_test, rf_proba)
    auc_rf = auc(fpr_rf, tpr_rf)
    
    ax.plot(fpr_rf, tpr_rf, color=COLORS["baseline_rf"], linewidth=2.5, 
            label=f"Baseline RF (8 features) — AUC: {auc_rf:.3f}")
    
    # Mark threshold
    idx = np.argmin(np.abs(th_rf - rf_thresh))
    ax.scatter(fpr_rf[idx], tpr_rf[idx], color=COLORS["baseline_rf"], s=150, zorder=5, 
               edgecolors="#1a1a2e", linewidth=2, marker='o')


    # ─────────────────────────────────────────────────────────────
    # 2. XGBoost (8 Features)
    # ─────────────────────────────────────────────────────────────
    print("▶ Loading XGBoost...")
    xgb_model_path = os.path.join(MODELS_DIR, "xgb_pathogenicity_model.pkl")
    xgb_thresh_path = os.path.join(MODELS_DIR, "xgb_optimal_threshold.pkl")
    
    if os.path.exists(xgb_model_path):
        xgb_model = joblib.load(xgb_model_path)
        xgb_thresh = joblib.load(xgb_thresh_path) if os.path.exists(xgb_thresh_path) else 0.5
        _, xgb_X_test, _, xgb_y_test, _ = load_xgb_data(seed=42)
        
        xgb_proba = xgb_model.predict_proba(xgb_X_test)[:, 1]
        fpr_xgb, tpr_xgb, th_xgb = roc_curve(xgb_y_test, xgb_proba)
        auc_xgb = auc(fpr_xgb, tpr_xgb)
        
        ax.plot(fpr_xgb, tpr_xgb, color=COLORS["xgboost"], linewidth=2.5, 
                label=f"XGBoost (8 features) — AUC: {auc_xgb:.3f}")
                
        idx = np.argmin(np.abs(th_xgb - xgb_thresh))
        ax.scatter(fpr_xgb[idx], tpr_xgb[idx], color=COLORS["xgboost"], s=150, zorder=5, 
                   edgecolors="#1a1a2e", linewidth=2, marker='^')
    else:
        print("  Missing XGBoost model files, skipping...")


    # ─────────────────────────────────────────────────────────────
    # 3. Extended Random Forest (7 Refined Features)
    # ─────────────────────────────────────────────────────────────
    print("▶ Loading Extended Random Forest (7 Refined Features)...")
    ext_model, _, _, ext_X_test, _, ext_y_test, ext_cv, _ = train_extended_rf(seed=42)
    ext_thresh = ext_cv.get("optimal_threshold", 0.5)
    
    ext_proba = ext_model.predict_proba(ext_X_test)[:, 1]
    fpr_ext, tpr_ext, th_ext = roc_curve(ext_y_test, ext_proba)
    auc_ext = auc(fpr_ext, tpr_ext)
    
    ax.plot(fpr_ext, tpr_ext, color=COLORS["extended_rf"], linewidth=2.5, 
            label=f"Extended RF (7 features) — AUC: {auc_ext:.3f}")
            
    idx = np.argmin(np.abs(th_ext - ext_thresh))
    ax.scatter(fpr_ext[idx], tpr_ext[idx], color=COLORS["extended_rf"], s=180, zorder=5, 
               edgecolors="#1a1a2e", linewidth=2, marker='*')


    # ─────────────────────────────────────────────────────────────
    # Formatting the Plot
    # ─────────────────────────────────────────────────────────────
    ax.plot([0, 1], [0, 1], color="#666666", linewidth=1.2, linestyle="--", label="Random Classifier (AUC = 0.500)")
    
    # Custom legend for markers
    import matplotlib.lines as mlines
    marker_rf = mlines.Line2D([], [], color=COLORS["baseline_rf"], marker='o', linestyle='None',
                              markersize=10, label=f"RF Optimal Threshold (CV)")
    marker_xgb = mlines.Line2D([], [], color=COLORS["xgboost"], marker='^', linestyle='None',
                              markersize=10, label=f"XGB Optimal Threshold (CV)")
    marker_ext = mlines.Line2D([], [], color=COLORS["extended_rf"], marker='*', linestyle='None',
                              markersize=14, label=f"Ext RF Optimal Threshold (CV)")
    
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([marker_rf, marker_xgb, marker_ext])
    
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("Model Comparison — Combined ROC Curves", fontsize=16, fontweight="bold", pad=15)
    
    ax.legend(handles=handles, loc="lower right", frameon=True, facecolor="#1e2a4a", edgecolor="#444", fontsize=11)
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "combined_roc_curve.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\n  ✓ Combined ROC curve saved to: {output_path}")


if __name__ == "__main__":
    main()
