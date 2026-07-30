#!/usr/bin/env python3
"""
Extract exact SHAP value statistics for ALL models:
  1. RF Baseline     — Training + Test
  2. XGBoost Baseline — Training + Test
  3. RF Extended-7   — Training + Test

Saves one JSON per model+set combination.
Does NOT modify any existing code or models.
"""
import os
import sys
import json
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import shap

# ── Imports from existing pipelines ──
from pipeline import train_pathogenicity_model, PATHOGENICITY_FEATURES
from pipeline_xgboost import load_and_split, tune_xgboost
from pipeline_extended import train_extended_pathogenicity_model, EXTENDED_PATHOGENICITY_FEATURES

OUTPUT_DIR = os.path.join(PROJECT_DIR, "shap_extracted_values")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42


def compute_shap(model, X_data, feature_names, X_background=None):
    """Compute SHAP values using TreeExplainer."""
    if X_background is not None:
        bg = X_background
        if len(bg) > 200:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(bg), 200, replace=False)
            bg = bg[idx]
        explainer = shap.TreeExplainer(model, data=bg, model_output="probability")
    else:
        explainer = shap.TreeExplainer(model)

    sv_raw = explainer.shap_values(X_data)

    if isinstance(sv_raw, list):
        sv = np.asarray(sv_raw[1])
        base = float(explainer.expected_value[1])
    elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
        sv = sv_raw[:, :, 1]
        base = float(explainer.expected_value[1])
    else:
        sv = np.asarray(sv_raw)
        ev = explainer.expected_value
        base = float(ev[1]) if hasattr(ev, "__len__") else float(ev)

    return sv, base


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


def save_and_print(description, n_samples, base_value, stats, feature_names, filename):
    """Save JSON and print summary table."""
    output = {
        "description": description,
        "n_samples": n_samples,
        "base_value": round(float(base_value), 4),
        "features": stats,
    }
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  {description}")
    print(f"  {'Feature':<20s} {'Min':>8s} {'Max':>8s} {'Mean':>8s} {'|Mean|':>8s} {'Range':>8s}")
    print("  " + "-" * 62)
    for fname in feature_names:
        s = stats[fname]
        print(f"  {fname:<20s} {s['min']:>8.4f} {s['max']:>8.4f} {s['mean']:>8.4f} {s['mean_abs']:>8.4f} {s['range']:>8.4f}")
    print(f"  ✓ Saved: {filename}")


def main():
    print("=" * 60)
    print("  SHAP Extraction — ALL MODELS (Train + Test)")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════
    #  1. RF BASELINE (already done but redo for completeness)
    # ═══════════════════════════════════════════════════════════
    print("\n▶ [1/3] RF Baseline...")
    rf_model, rf_feats, rf_Xtest, rf_ytest, rf_cv, _ = train_pathogenicity_model()
    rf_feat_names = list(rf_feats)

    # Get training data
    rf_Xtrain, _, rf_ytrain, _, _ = load_and_split(seed=SEED)

    # Training
    print(f"  Computing SHAP on Training Set (n={len(rf_Xtrain)})...")
    sv_train, base_train = compute_shap(rf_model, rf_Xtrain, rf_feat_names)
    stats_train = extract_stats(sv_train, rf_feat_names)
    save_and_print("RF Baseline — Training Set", len(rf_Xtrain), base_train,
                   stats_train, rf_feat_names, "shap_rf_baseline_train.json")

    # Test
    print(f"  Computing SHAP on Test Set (n={len(rf_Xtest)})...")
    sv_test, base_test = compute_shap(rf_model, rf_Xtest, rf_feat_names)
    stats_test = extract_stats(sv_test, rf_feat_names)
    save_and_print("RF Baseline — Test Set", len(rf_Xtest), base_test,
                   stats_test, rf_feat_names, "shap_rf_baseline_test.json")

    # ═══════════════════════════════════════════════════════════
    #  2. XGBOOST BASELINE
    # ═══════════════════════════════════════════════════════════
    print("\n\n▶ [2/3] XGBoost Baseline...")
    xgb_model, xgb_Xtrain, xgb_Xtest, xgb_ytrain, xgb_ytest, _, _, _ = \
        tune_xgboost(seed=SEED)

    # XGBoost needs background data for probability-scale SHAP
    # Training
    print(f"  Computing SHAP on Training Set (n={len(xgb_Xtrain)})...")
    sv_train_xgb, base_train_xgb = compute_shap(
        xgb_model, xgb_Xtrain, rf_feat_names, X_background=xgb_Xtrain
    )
    stats_train_xgb = extract_stats(sv_train_xgb, rf_feat_names)
    save_and_print("XGBoost Baseline — Training Set", len(xgb_Xtrain), base_train_xgb,
                   stats_train_xgb, rf_feat_names, "shap_xgb_baseline_train.json")

    # Test
    print(f"  Computing SHAP on Test Set (n={len(xgb_Xtest)})...")
    sv_test_xgb, base_test_xgb = compute_shap(
        xgb_model, xgb_Xtest, rf_feat_names, X_background=xgb_Xtrain
    )
    stats_test_xgb = extract_stats(sv_test_xgb, rf_feat_names)
    save_and_print("XGBoost Baseline — Test Set", len(xgb_Xtest), base_test_xgb,
                   stats_test_xgb, rf_feat_names, "shap_xgb_baseline_test.json")

    # ═══════════════════════════════════════════════════════════
    #  3. RF EXTENDED-7
    # ═══════════════════════════════════════════════════════════
    print("\n\n▶ [3/3] RF Extended-7...")
    ext_model, ext_feats, ext_Xtest, ext_ytest, ext_cv, ext_meta = \
        train_extended_pathogenicity_model(seed=SEED)
    ext_feat_names = list(ext_feats)

    # Need training data for extended model
    from pipeline_extended import _load_extended_pathogenicity_training_data
    ext_X, ext_y, _ = _load_extended_pathogenicity_training_data()
    ext_X_arr = ext_X.values.astype(float)
    ext_y_arr = ext_y.values

    indices = np.arange(len(ext_X_arr))
    from sklearn.model_selection import train_test_split
    idx_train, idx_test, _, _ = train_test_split(
        indices, ext_y_arr, test_size=0.2, random_state=SEED, stratify=ext_y_arr
    )
    ext_Xtrain = ext_X_arr[idx_train]

    # Training
    print(f"  Computing SHAP on Training Set (n={len(ext_Xtrain)})...")
    sv_train_ext, base_train_ext = compute_shap(ext_model, ext_Xtrain, ext_feat_names)
    stats_train_ext = extract_stats(sv_train_ext, ext_feat_names)
    save_and_print("RF Extended-7 — Training Set", len(ext_Xtrain), base_train_ext,
                   stats_train_ext, ext_feat_names, "shap_rf_extended7_train.json")

    # Test
    print(f"  Computing SHAP on Test Set (n={len(ext_Xtest)})...")
    sv_test_ext, base_test_ext = compute_shap(ext_model, ext_Xtest, ext_feat_names)
    stats_test_ext = extract_stats(sv_test_ext, ext_feat_names)
    save_and_print("RF Extended-7 — Test Set", len(ext_Xtest), base_test_ext,
                   stats_test_ext, ext_feat_names, "shap_rf_extended7_test.json")

    print(f"\n\n{'=' * 60}")
    print(f"  ✓ ALL DONE — 6 JSON files saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
