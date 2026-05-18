#!/usr/bin/env python3
"""Runner for the XGBoost (Optuna) pathogenicity pipeline.

Mirrors ``run_pipeline.py`` but trains, evaluates, and plots the
XGBoost model only.  Does NOT modify run_pipeline.py or any
Random-Forest artefacts.
"""

import os
import sys
import json

import numpy as np
import pandas as pd

# Ensure the project directory is on the Python path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from pipeline_xgboost import train_pathogenicity_model_xgb
from PLOTS.visualizations_xgboost import generate_all_xgb_plots


def main():

    print("=" * 65)
    print("  TP53 XGBoost (Optuna) Pathogenicity Pipeline")
    print("=" * 65)

    # ── Step 1: Train + CV + test evaluation ─────────────────────
    print("\n▶ Step 1: Training XGBoost classifier (Optuna, 50 trials)...")
    xgb_best, path_features, X_test, y_test, cv_results, meta_test = (
        train_pathogenicity_model_xgb()
    )

    # ── Step 2: Save CV comparison JSON ──────────────────────────
    cv_json = os.path.join(PROJECT_DIR, "cv_model_comparison_xgboost.json")
    cv_serializable = {}
    for k, v in cv_results.items():
        if isinstance(v, tuple):
            cv_serializable[k] = {"mean": round(v[0], 4), "std": round(v[1], 4)}
        else:
            cv_serializable[k] = v
    with open(cv_json, "w") as f:
        json.dump(cv_serializable, f, indent=2, default=str)
    print(f"\n  ✓ CV comparison saved to: {cv_json}")

    # ── Step 3: Save feature importances CSV ─────────────────────
    importances = xgb_best.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": path_features,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    fi_csv = os.path.join(PROJECT_DIR, "feature_importances_pathogenicity_xgboost.csv")
    fi_df.to_csv(fi_csv, index=False)
    print(f"  ✓ Feature importances saved to: {fi_csv}")

    # ── Step 4: Generate thesis plots ────────────────────────────
    optimal_threshold = cv_results["optimal_threshold"]
    print(f"\n▶ Step 2: Generating XGBoost thesis plots (threshold={optimal_threshold:.4f})...")
    plots = generate_all_xgb_plots(
        path_model=xgb_best,
        path_features=path_features,
        path_X_test=X_test,
        path_y_test=y_test,
        output_dir=PROJECT_DIR,
        optimal_threshold=optimal_threshold,
    )

    # ── Step 5: Side-by-side comparison table ────────────────────
    print("\n" + "=" * 65)
    print("  Side-by-Side Model Comparison (5-Fold CV)")
    print("=" * 65)

    rows = []

    # Load Random Forest results if available
    rf_json_path = os.path.join(PROJECT_DIR, "cv_model_comparison.json")
    if os.path.exists(rf_json_path):
        with open(rf_json_path, "r") as f:
            rf = json.load(f)
        rows.append({
            "Model": "Random Forest",
            "Accuracy": f"{rf['rf_accuracy']['mean']:.4f} ± {rf['rf_accuracy']['std']:.4f}",
            "F1": f"{rf['rf_f1']['mean']:.4f} ± {rf['rf_f1']['std']:.4f}",
            "AUC": f"{rf['rf_auc']['mean']:.4f} ± {rf['rf_auc']['std']:.4f}",
            "Sens": f"{rf['rf_sensitivity']['mean']:.4f} ± {rf['rf_sensitivity']['std']:.4f}" if "rf_sensitivity" in rf else "N/A",
            "Spec": f"{rf['rf_specificity']['mean']:.4f} ± {rf['rf_specificity']['std']:.4f}" if "rf_specificity" in rf else "N/A",
            "Threshold": f"{rf['optimal_threshold']:.4f}",
        })
    else:
        print("  (cv_model_comparison.json not found — run run_pipeline.py first for RF metrics)")

    # XGBoost results (just computed)
    xgb_ser = cv_serializable
    rows.append({
        "Model": "XGBoost (Optuna)",
        "Accuracy": f"{xgb_ser['xgb_accuracy']['mean']:.4f} ± {xgb_ser['xgb_accuracy']['std']:.4f}",
        "F1": f"{xgb_ser['xgb_f1']['mean']:.4f} ± {xgb_ser['xgb_f1']['std']:.4f}",
        "AUC": f"{xgb_ser['xgb_auc']['mean']:.4f} ± {xgb_ser['xgb_auc']['std']:.4f}",
        "Sens": f"{xgb_ser['xgb_sensitivity']['mean']:.4f} ± {xgb_ser['xgb_sensitivity']['std']:.4f}",
        "Spec": f"{xgb_ser['xgb_specificity']['mean']:.4f} ± {xgb_ser['xgb_specificity']['std']:.4f}",
        "Threshold": f"{xgb_ser['optimal_threshold']:.4f}",
    })

    # Print table
    header = f"  {'Model':<20s} {'Accuracy':<18s} {'F1':<18s} {'AUC':<18s} {'Sensitivity':<18s} {'Specificity':<18s} {'Threshold':<10s}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for r in rows:
        print(f"  {r['Model']:<20s} {r['Accuracy']:<18s} {r['F1']:<18s} {r['AUC']:<18s} {r['Sens']:<18s} {r['Spec']:<18s} {r['Threshold']:<10s}")
    print()

    # ── Summary ──────────────────────────────────────────────────
    print("=" * 65)
    print("  Pipeline complete!")
    print("=" * 65)

    print(f"\n  Output files:")
    print(f"    • cv_model_comparison_xgboost.json")
    print(f"    • feature_importances_pathogenicity_xgboost.csv")

    print(f"\n  Thesis plots ({len(plots)}):")
    for p in plots:
        print(f"    • {os.path.basename(p)}")
    print()


if __name__ == "__main__":
    main()
