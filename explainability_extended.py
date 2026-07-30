#!/usr/bin/env python3
"""
SHAP Explainability & Case Study Extraction for the Extended TP53 Model.

Uses the refined 7-feature set selected by automated RFECV + permutation
importance filtering:
  Grantham_Score, REVEL, BAYESDEL, Is_Hotspot,
  SpliceAI_DS_AG, SpliceAI_DS_DG, Exon_Number

Generates:
  • Global SHAP plots (beeswarm + bar) on test set
  • SHAP dependence plots for top-3 features
  • 5 case studies (TP, TN, edge correct, edge incorrect, error)
    with per-case SHAP waterfall plots
  • Training-set SHAP analysis

All outputs are saved under case_studies/extended_pathogenicity/
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from pipeline_extended import (
    _load_extended_pathogenicity_training_data,
    EXTENDED_PATHOGENICITY_FEATURES,
)
from pipeline import (
    _load_iarc_csv, GERMLINE_CSV, SOMATIC_CSV,
    AA_3TO1, AGVGD_MAP, SIFT_MAP, PP2_MAP,
)
from grantham import get_grantham_score

# Reuse all SHAP and case-study logic from the baseline explainability module
from explainability import (
    extract_cases,
    generate_shap_global,
    generate_shap_dependence,
    generate_shap_case,
    generate_training_shap,
    run_stage,
    _ensure_dir,
    CASE_STUDIES_DIR,
)

# ───────────────────────────────────────────────────────────────
#  Refined 7-feature list (after permutation importance filtering)
# ───────────────────────────────────────────────────────────────
REFINED_FEATURES = [
    "Grantham_Score", "REVEL", "BAYESDEL",
    "Is_Hotspot",
    "SpliceAI_DS_AG", "SpliceAI_DS_DG",
    "Exon_Number",
]


# ───────────────────────────────────────────────────────────────
#  Training pipeline for the refined 7-feature model
# ───────────────────────────────────────────────────────────────

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
)
from sklearn.metrics import (
    classification_report, roc_curve, f1_score,
    accuracy_score, roc_auc_score, recall_score,
)

SEED = 42


def _load_refined_training_data():
    """Load the combined IARC data with only the refined 7 features."""
    # Reuse the full 12-feature loader, then select the 7 refined features
    X_full, y, meta = _load_extended_pathogenicity_training_data()
    X = X_full[REFINED_FEATURES].copy()
    print(f"  Refined to {len(REFINED_FEATURES)} features: {REFINED_FEATURES}")
    return X, y, meta


def train_refined_pathogenicity_model(seed: int = SEED):
    """Train the refined 7-feature Random Forest model.

    Mirrors pipeline.train_pathogenicity_model() exactly:
      1. 80/20 stratified train/test split
      2. GridSearchCV with 5-fold stratified CV
      3. Per-fold optimal threshold via Youden's J statistic
      4. Hold-out test evaluation

    Returns
    -------
    (rf_best, REFINED_FEATURES, X_test, y_test, cv_results, meta_test)
    """
    X, y, meta = _load_refined_training_data()
    X_arr = X.values.astype(float)
    y_arr = y.values
    meta_arr = meta.reset_index(drop=True)

    indices = np.arange(len(X_arr))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y_arr, test_size=0.2, random_state=seed, stratify=y_arr
    )
    X_train = X_arr[idx_train]
    X_test  = X_arr[idx_test]
    meta_test = meta_arr.iloc[idx_test].reset_index(drop=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 1. GridSearchCV
    print("\n  ── GridSearchCV: Tuning Refined Random Forest (7 features) ──")
    rf_param_grid = {
        "n_estimators":      [200, 400, 600],
        "max_depth":         [8, 12, 16, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
    }
    rf_base = RandomForestClassifier(
        class_weight="balanced", random_state=seed, n_jobs=-1
    )

    def _make_grid_search(n_jobs):
        return GridSearchCV(
            rf_base, rf_param_grid, cv=cv,
            scoring="f1_weighted", n_jobs=n_jobs, refit=True,
        )

    grid_search = _make_grid_search(n_jobs=-1)
    try:
        grid_search.fit(X_train, y_train)
    except PermissionError:
        print("    Parallel GridSearchCV unavailable; retrying with n_jobs=1...")
        grid_search = _make_grid_search(n_jobs=1)
        grid_search.fit(X_train, y_train)

    rf_best = grid_search.best_estimator_
    print(f"    Best params: {grid_search.best_params_}")
    print(f"    Best CV F1:  {grid_search.best_score_:.4f}")

    # 2. 5-Fold CV with per-fold optimal threshold
    print("\n  ── 5-Fold Cross-Validation: Refined Random Forest ──")
    fold_accuracies    = []
    fold_f1s           = []
    fold_aucs          = []
    fold_thresholds    = []
    fold_sensitivities = []
    fold_specificities = []

    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_tr_fold, X_val_fold = X_train[tr_idx], X_train[val_idx]
        y_tr_fold, y_val_fold = y_train[tr_idx], y_train[val_idx]

        fold_model = RandomForestClassifier(
            **grid_search.best_params_,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
        fold_model.fit(X_tr_fold, y_tr_fold)

        val_proba = fold_model.predict_proba(X_val_fold)[:, 1]

        fpr_fold, tpr_fold, thresholds_fold = roc_curve(y_val_fold, val_proba)
        j_scores = tpr_fold - fpr_fold
        best_j_idx = np.argmax(j_scores)
        fold_threshold = thresholds_fold[best_j_idx]
        fold_thresholds.append(fold_threshold)

        val_pred = (val_proba >= fold_threshold).astype(int)
        fold_accuracies.append(accuracy_score(y_val_fold, val_pred))
        fold_f1s.append(f1_score(y_val_fold, val_pred, average="weighted"))
        fold_aucs.append(roc_auc_score(y_val_fold, val_proba))
        fold_sensitivities.append(recall_score(y_val_fold, val_pred, pos_label=1, zero_division=0))
        fold_specificities.append(recall_score(y_val_fold, val_pred, pos_label=0, zero_division=0))

        print(f"    Fold {fold_idx}: threshold={fold_threshold:.4f}, "
              f"Acc={fold_accuracies[-1]:.4f}, AUC={fold_aucs[-1]:.4f}, "
              f"Sens={fold_sensitivities[-1]:.4f}, Spec={fold_specificities[-1]:.4f}")

    rf_acc_mean  = np.mean(fold_accuracies)
    rf_acc_std   = np.std(fold_accuracies)
    rf_f1_mean   = np.mean(fold_f1s)
    rf_f1_std    = np.std(fold_f1s)
    rf_auc_mean  = np.mean(fold_aucs)
    rf_auc_std   = np.std(fold_aucs)
    rf_sens_mean = np.mean(fold_sensitivities)
    rf_sens_std  = np.std(fold_sensitivities)
    rf_spec_mean = np.mean(fold_specificities)
    rf_spec_std  = np.std(fold_specificities)
    cv_optimal_threshold = float(np.mean(fold_thresholds))

    print(f"\n    Accuracy:     {rf_acc_mean:.4f} ± {rf_acc_std:.4f}")
    print(f"    F1 Score:     {rf_f1_mean:.4f} ± {rf_f1_std:.4f}")
    print(f"    AUC:          {rf_auc_mean:.4f} ± {rf_auc_std:.4f}")
    print(f"    Sensitivity:  {rf_sens_mean:.4f} ± {rf_sens_std:.4f}")
    print(f"    Specificity:  {rf_spec_mean:.4f} ± {rf_spec_std:.4f}")
    print(f"    CV Optimal Threshold: {cv_optimal_threshold:.4f} "
          f"(per-fold: {[round(t, 4) for t in fold_thresholds]})")

    # 3. Hold-out test evaluation
    y_test_proba = rf_best.predict_proba(X_test)[:, 1]
    y_pred = (y_test_proba >= cv_optimal_threshold).astype(int)

    print("\n  ── Refined Model: Random Forest (hold-out test evaluation) ──")
    print(f"    Using CV optimal threshold: {cv_optimal_threshold:.4f}")
    test_sens = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    test_spec = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    print(classification_report(
        y_test, y_pred,
        target_names=["Functional", "Non-functional"],
        zero_division=0,
    ))
    print(f"    Sensitivity: {test_sens:.4f}")
    print(f"    Specificity: {test_spec:.4f}")

    cv_results = {
        "rf_accuracy":        (rf_acc_mean,  rf_acc_std),
        "rf_f1":              (rf_f1_mean,   rf_f1_std),
        "rf_auc":             (rf_auc_mean,  rf_auc_std),
        "rf_sensitivity":     (rf_sens_mean, rf_sens_std),
        "rf_specificity":     (rf_spec_mean, rf_spec_std),
        "test_sensitivity":   float(test_sens),
        "test_specificity":   float(test_spec),
        "best_model":         "Random Forest",
        "best_params":        grid_search.best_params_,
        "optimal_threshold":  cv_optimal_threshold,
        "per_fold_thresholds": fold_thresholds,
    }

    return rf_best, REFINED_FEATURES, X_train, X_test, y_train, y_test, cv_results, meta_test


# ───────────────────────────────────────────────────────────────
#  MAIN
# ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TP53 Extended Model — Explainability & Case Studies")
    print("  SHAP Analysis (Refined 7-Feature Model)")
    print("=" * 60)

    _ensure_dir(CASE_STUDIES_DIR)

    # 1. Train the refined 7-feature model
    print("\n▶ Training Refined Pathogenicity Model (7 features)...")
    (model, features, X_train, X_test,
     y_train, y_test, cv_results, meta_test) = train_refined_pathogenicity_model()

    optimal_threshold = cv_results.get("optimal_threshold", 0.5)
    print(f"  Using CV optimal threshold: {optimal_threshold:.4f}")

    # 2. Run full explainability pipeline (cases + global SHAP + dependence + waterfalls)
    run_stage(
        "extended_pathogenicity",
        "Extended Pathogenicity (7 Refined Features)",
        ["Functional", "Non-functional"],
        model, X_test, y_test, features,
        meta_test=meta_test,
        optimal_threshold=optimal_threshold,
    )

    # 3. Training-set SHAP (beeswarm + bar + dependence)
    print("\n▶ Generating Training-Set SHAP for Extended RF...")
    stage_dir = os.path.join(CASE_STUDIES_DIR, "extended_pathogenicity")
    generate_training_shap(
        model, X_train, features, stage_dir,
        "Extended Pathogenicity (7 Refined Features)",
    )

    # 4. Print summary
    rf_f1  = cv_results["rf_f1"]
    rf_auc = cv_results["rf_auc"]
    rf_acc = cv_results["rf_accuracy"]

    print("\n" + "=" * 60)
    print("  ✓ Extended Explainability Complete!")
    print("=" * 60)
    print(f"\n  ── Refined Model — 5-Fold CV ──")
    print(f"    Features:     {features}")
    print(f"    Accuracy:     {rf_acc[0]:.4f} ± {rf_acc[1]:.4f}")
    print(f"    F1 Score:     {rf_f1[0]:.4f} ± {rf_f1[1]:.4f}")
    print(f"    AUC:          {rf_auc[0]:.4f} ± {rf_auc[1]:.4f}")
    print(f"    Threshold:    {optimal_threshold:.4f}")
    print(f"\n    → Case studies & SHAP saved to: {CASE_STUDIES_DIR}/extended_pathogenicity/")
    print("=" * 60)


if __name__ == "__main__":
    main()
