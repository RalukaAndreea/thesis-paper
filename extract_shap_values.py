#!/usr/bin/env python3
"""
Standalone script to extract exact SHAP value statistics per feature
from the RF Baseline (Training Set) and RF Baseline (Test Set) beeswarm plots.

Outputs JSON files with min, max, mean, std, median, P5, P95 per feature.
Does NOT modify any existing code or models.
"""
import os
import sys
import json
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from pipeline import train_pathogenicity_model, PATHOGENICITY_FEATURES
from pipeline_xgboost import load_and_split
from explainability import _compute_shap

OUTPUT_DIR = os.path.join(PROJECT_DIR, "shap_extracted_values")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42


def extract_stats(shap_values, feature_names):
    """Extract per-feature statistics from a SHAP values matrix."""
    stats = {}
    for i, fname in enumerate(feature_names):
        col = shap_values[:, i]
        stats[fname] = {
            "min": round(float(np.min(col)), 4),
            "max": round(float(np.max(col)), 4),
            "mean": round(float(np.mean(col)), 4),
            "std": round(float(np.std(col)), 4),
            "median": round(float(np.median(col)), 4),
            "P5": round(float(np.percentile(col, 5)), 4),
            "P95": round(float(np.percentile(col, 95)), 4),
            "mean_abs": round(float(np.mean(np.abs(col))), 4),
            "n_positive": int(np.sum(col > 0)),
            "n_negative": int(np.sum(col < 0)),
            "n_zero": int(np.sum(col == 0)),
            "range": round(float(np.max(col) - np.min(col)), 4),
        }
    return stats


def main():
    print("=" * 60)
    print("  SHAP Value Extraction — Exact Statistics")
    print("=" * 60)

    # 1. Train the model (reuses cached splits via pipeline)
    print("\n▶ Training Pathogenicity Model...")
    model, feats, X_test, y_test, cv_results, meta_test = train_pathogenicity_model()
    feature_names = list(feats)

    # 2. Get training data
    print("\n▶ Loading training split...")
    X_train, X_test_xgb, y_train, y_test_xgb, _ = load_and_split(seed=SEED)

    # ─── TRAINING SET SHAP ───
    print(f"\n▶ Computing SHAP on Training Set (n={len(X_train)})...")
    sv_train, base_train = _compute_shap(model, X_train, feature_names)
    stats_train = extract_stats(sv_train, feature_names)

    train_output = {
        "description": "SHAP values — RF Baseline, Training Set",
        "n_samples": int(len(X_train)),
        "base_value": round(float(base_train), 4),
        "features": stats_train,
    }

    train_path = os.path.join(OUTPUT_DIR, "shap_stats_rf_baseline_training.json")
    with open(train_path, "w") as f:
        json.dump(train_output, f, indent=2)
    print(f"  ✓ Saved: {train_path}")

    # ─── TEST SET SHAP ───
    print(f"\n▶ Computing SHAP on Test Set (n={len(X_test)})...")
    sv_test, base_test = _compute_shap(model, X_test, feature_names)
    stats_test = extract_stats(sv_test, feature_names)

    test_output = {
        "description": "SHAP values — RF Baseline, Test Set",
        "n_samples": int(len(X_test)),
        "base_value": round(float(base_test), 4),
        "features": stats_test,
    }

    test_path = os.path.join(OUTPUT_DIR, "shap_stats_rf_baseline_test.json")
    with open(test_path, "w") as f:
        json.dump(test_output, f, indent=2)
    print(f"  ✓ Saved: {test_path}")

    # ─── PRINT SUMMARY TABLE ───
    print("\n" + "=" * 60)
    print("  TRAINING SET — SHAP Value Ranges")
    print("=" * 60)
    print(f"  {'Feature':<18s} {'Min':>8s} {'Max':>8s} {'Mean':>8s} {'|Mean|':>8s} {'Range':>8s}")
    print("  " + "-" * 58)
    for fname in feature_names:
        s = stats_train[fname]
        print(f"  {fname:<18s} {s['min']:>8.4f} {s['max']:>8.4f} {s['mean']:>8.4f} {s['mean_abs']:>8.4f} {s['range']:>8.4f}")

    print("\n" + "=" * 60)
    print("  TEST SET — SHAP Value Ranges")
    print("=" * 60)
    print(f"  {'Feature':<18s} {'Min':>8s} {'Max':>8s} {'Mean':>8s} {'|Mean|':>8s} {'Range':>8s}")
    print("  " + "-" * 58)
    for fname in feature_names:
        s = stats_test[fname]
        print(f"  {fname:<18s} {s['min']:>8.4f} {s['max']:>8.4f} {s['mean']:>8.4f} {s['mean_abs']:>8.4f} {s['range']:>8.4f}")

    print(f"\n📁 JSON files saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
