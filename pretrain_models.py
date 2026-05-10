#!/usr/bin/env python3
"""Pre-train and save the TP53 pathogenicity model for fast loading in the Streamlit app."""

import os
import sys
import joblib

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from pipeline import train_pathogenicity_model


def main():
    models_dir = os.path.join(PROJECT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("  Pre-training TP53 Pathogenicity Model")
    print("=" * 60)

    print("\n▶ Training Pathogenicity Model (Random Forest + GridSearchCV)...")
    path_model, path_features, _, _, cv_results = train_pathogenicity_model()
    joblib.dump(path_model, os.path.join(models_dir, "pathogenicity_model.pkl"))
    print("  ✓ Saved pathogenicity_model.pkl")

    print(f"\n  Best params: {cv_results['best_params']}")
    print(f"  CV Accuracy: {cv_results['rf_accuracy'][0]:.4f} ± {cv_results['rf_accuracy'][1]:.4f}")
    print(f"  CV F1:       {cv_results['rf_f1'][0]:.4f} ± {cv_results['rf_f1'][1]:.4f}")
    print(f"  CV AUC:      {cv_results['rf_auc'][0]:.4f} ± {cv_results['rf_auc'][1]:.4f}")

    print("\n" + "=" * 60)
    print("  Model saved to:", models_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
