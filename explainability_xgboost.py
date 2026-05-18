#!/usr/bin/env python3
"""
SHAP Explainability & Case Study Extraction for the XGBoost (Optuna) model.

Mirrors explainability.py but uses the pretrained XGBoost pickle and writes
outputs to case_studies/xgb_pathogenicity/ so both models' SHAP results can
coexist and be compared in the Streamlit app.
"""

import os
import sys

import joblib

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from pipeline_xgboost import load_and_split, PATHOGENICITY_FEATURES

# Re-use the stage runner and helpers from the RF explainability module
from explainability import run_stage, generate_training_shap, CASE_STUDIES_DIR, _ensure_dir

MODELS_DIR = os.path.join(PROJECT_DIR, "models")


def main():
    print("=" * 60)
    print("  TP53 Explainability — XGBoost (Optuna)")
    print("  SHAP Analysis on 20% Hold-Out Test Set")
    print("=" * 60)

    _ensure_dir(CASE_STUDIES_DIR)

    # 1. Load pretrained XGBoost model + CV optimal threshold
    model_path = os.path.join(MODELS_DIR, "xgb_pathogenicity_model.pkl")
    thresh_path = os.path.join(MODELS_DIR, "xgb_optimal_threshold.pkl")

    if not os.path.exists(model_path):
        print("ERROR: xgb_pathogenicity_model.pkl not found.")
        print("       Run `python pretrain_models_xgboost.py` first.")
        sys.exit(1)

    xgb_model = joblib.load(model_path)
    optimal_threshold = joblib.load(thresh_path) if os.path.exists(thresh_path) else 0.5
    print(f"\n  ✓ Loaded pretrained XGBoost model")
    print(f"  ✓ CV optimal threshold: {optimal_threshold:.4f}")

    # 2. Reproduce the exact same train/test split (seed=42, identical to RF)
    print("\n▶ Reproducing train/test split...")
    X_train, X_test, y_train, y_test, meta_test = load_and_split(seed=42)
    print(f"  Test set: {len(y_test)} samples "
          f"(Functional={int((y_test == 0).sum())}, "
          f"Non-functional={int((y_test == 1).sum())})")

    # 3. Run the full explainability pipeline (test set)
    #    X_train is passed as background so XGBoost SHAP values are on the
    #    probability scale (not raw log-odds).
    run_stage(
        "xgb_pathogenicity", "XGBoost — Pathogenicity",
        ["Functional", "Non-functional"],
        xgb_model, X_test, y_test, list(PATHOGENICITY_FEATURES),
        meta_test=meta_test,
        optimal_threshold=optimal_threshold,
        X_background=X_train,
    )

    # 4. Training-set SHAP (beeswarm + bar + dependence)
    print("\n▶ Generating Training-Set SHAP for XGBoost...")
    stage_dir = os.path.join(CASE_STUDIES_DIR, "xgb_pathogenicity")
    generate_training_shap(
        xgb_model, X_train, list(PATHOGENICITY_FEATURES),
        stage_dir, "XGBoost — Pathogenicity",
        X_background=X_train,
    )

    print("\n" + "=" * 60)
    print("  ✓ XGBoost case studies & SHAP explanations generated!")
    print(f"    → {os.path.join(CASE_STUDIES_DIR, 'xgb_pathogenicity')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
