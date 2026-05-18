#!/usr/bin/env python3
"""Pre-train and save the XGBoost pathogenicity model for fast loading in the Streamlit app."""

import os
import sys
import joblib

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from pipeline_xgboost import train_pathogenicity_model_xgb


def main():
    models_dir = os.path.join(PROJECT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("  Pre-training TP53 XGBoost (Optuna) Model")
    print("=" * 60)

    print("\n▶ Training XGBoost Model (Optuna, 50 trials)...")
    xgb_model, features, _, _, cv_results, _ = train_pathogenicity_model_xgb()

    joblib.dump(xgb_model, os.path.join(models_dir, "xgb_pathogenicity_model.pkl"))
    print("  ✓ Saved xgb_pathogenicity_model.pkl")

    # Save CV optimal threshold alongside the model
    optimal_threshold = cv_results.get("optimal_threshold", 0.5)
    joblib.dump(optimal_threshold, os.path.join(models_dir, "xgb_optimal_threshold.pkl"))
    print(f"  ✓ Saved xgb_optimal_threshold.pkl (threshold = {optimal_threshold:.4f})")

    print(f"\n  Best params: {cv_results['best_params']}")
    print(f"  CV Accuracy: {cv_results['xgb_accuracy'][0]:.4f} ± {cv_results['xgb_accuracy'][1]:.4f}")
    print(f"  CV F1:       {cv_results['xgb_f1'][0]:.4f} ± {cv_results['xgb_f1'][1]:.4f}")
    print(f"  CV AUC:      {cv_results['xgb_auc'][0]:.4f} ± {cv_results['xgb_auc'][1]:.4f}")
    print(f"  CV Threshold: {optimal_threshold:.4f}")

    print("\n" + "=" * 60)
    print("  Model saved to:", models_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
