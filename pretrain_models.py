#!/usr/bin/env python3
"""Pre-train and save all TP53 models for fast loading in the Streamlit app."""

import os
import sys
import joblib

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from pipeline import (
    train_pathogenicity_model,
    train_origin_model,
    train_nonfunc_origin_model,
    PATHOGENICITY_FEATURES,
    ORIGIN_FEATURES,
)


def main():
    models_dir = os.path.join(PROJECT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("  Pre-training TP53 Models")
    print("=" * 60)

    # Stage 1: Pathogenicity
    print("\n▶ Training Stage 1 — Pathogenicity...")
    path_model, path_features, _, _, cv_results = train_pathogenicity_model()
    joblib.dump(path_model, os.path.join(models_dir, "pathogenicity_model.pkl"))
    print("  ✓ Saved pathogenicity_model.pkl")

    # Stage 2a: Origin (all variants)
    print("\n▶ Training Stage 2a — Origin (All)...")
    origin_model, origin_features, _, _ = train_origin_model()
    joblib.dump(origin_model, os.path.join(models_dir, "origin_model.pkl"))
    print("  ✓ Saved origin_model.pkl")

    # Stage 2b: Origin (non-functional only)
    print("\n▶ Training Stage 2b — Origin (Non-Functional)...")
    nf_origin_model, nf_origin_features, _, _ = train_nonfunc_origin_model()
    joblib.dump(nf_origin_model, os.path.join(models_dir, "nf_origin_model.pkl"))
    print("  ✓ Saved nf_origin_model.pkl")

    print("\n" + "=" * 60)
    print("  All models saved to:", models_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
