"""
pipeline_extended.py
────────────────────
Extended 12-feature Random Forest pathogenicity model for TP53 missense
variants.  Mirrors the training flow of ``pipeline.py`` but uses the
feature set selected by RFECV in ``automate_feature_selection_extended.py``:

  Baseline (7):  Grantham_Score, REVEL, BAYESDEL, AGVGDClass, SIFTClass,
                 Polyphen2, Is_Hotspot
  New      (5):  SpliceAI_DS_AG, SpliceAI_DS_AL, SpliceAI_DS_DG,
                 SpliceAI_DS_DL, Exon_Number

Is_CpG was dropped by RFECV → not included here.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import (
    classification_report,
    roc_curve,
    f1_score,
    accuracy_score,
    roc_auc_score,
    recall_score,
)

from pipeline import (
    _load_iarc_csv,
    GERMLINE_CSV,
    SOMATIC_CSV,
    AA_3TO1,
    AGVGD_MAP,
    SIFT_MAP,
    PP2_MAP,
)
from grantham import get_grantham_score

# ──────────────────────────────────────────────────────────────────────
#  Feature list — 12 features kept by RFECV (Is_CpG dropped)
# ──────────────────────────────────────────────────────────────────────

''''
EXTENDED_PATHOGENICITY_FEATURES = [
    "Grantham_Score", "REVEL", "BAYESDEL",
    "AGVGDClass", "SIFTClass", "Polyphen2",
    "Is_Hotspot",
    "SpliceAI_DS_AG", "SpliceAI_DS_AL", "SpliceAI_DS_DG", "SpliceAI_DS_DL",
    "Exon_Number",
]
'''


EXTENDED_PATHOGENICITY_FEATURES = [
    "Grantham_Score", "REVEL", "BAYESDEL",
    "Is_Hotspot",
    "SpliceAI_DS_AG", "SpliceAI_DS_DG",
    "Exon_Number",
]


# ──────────────────────────────────────────────────────────────────────
#  Data loading & feature engineering
# ──────────────────────────────────────────────────────────────────────

def _load_extended_pathogenicity_training_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load IARC missense data and engineer all 12 extended features.

    Returns
    -------
    X : DataFrame   – feature matrix  (n_samples × 12)
    y : Series      – binary labels   (0 = functional, 1 = non-functional)
    meta : DataFrame – metadata columns for case-study identification
    """
    germ = _load_iarc_csv(GERMLINE_CSV)
    soma = _load_iarc_csv(SOMATIC_CSV)

    # Filter to missense only and combine
    germ_miss = germ[germ["Effect"] == "missense"].copy()
    soma_miss = soma[soma["Effect"] == "missense"].copy()
    combined = pd.concat([germ_miss, soma_miss], ignore_index=True)

    # Deduplicate by unique mutation
    combined = combined.drop_duplicates(
        subset=["WT_AA", "Mutant_AA", "Codon_number"]
    ).copy()

    # ── Baseline features (7) ──────────────────────────────────────
    # Grantham Score
    combined["AA_REF_1"] = combined["WT_AA"].map(AA_3TO1)
    combined["AA_ALT_1"] = combined["Mutant_AA"].map(AA_3TO1)
    combined["Grantham_Score"] = combined.apply(
        lambda r: get_grantham_score(r["AA_REF_1"], r["AA_ALT_1"])
        if pd.notna(r["AA_REF_1"]) and pd.notna(r["AA_ALT_1"]) else 0,
        axis=1,
    ).fillna(0)

    # REVEL & BayesDel
    combined["REVEL"]    = pd.to_numeric(combined["REVEL"],    errors="coerce").fillna(0.5)
    combined["BAYESDEL"] = pd.to_numeric(combined["BayesDel"], errors="coerce").fillna(0.0)

    # Categorical predictors
    combined["AGVGDClass"] = (
        combined["AGVGDClass"].astype(str).str.strip()
        .map(AGVGD_MAP).fillna(3).astype(int)
    )
    combined["SIFTClass"] = (
        combined["SIFTClass"].astype(str).str.strip()
        .map(SIFT_MAP).fillna(0).astype(int)
    )
    combined["Polyphen2"] = (
        combined["Polyphen2"].astype(str).str.strip()
        .map(PP2_MAP).fillna(1).astype(int)
    )

    # Hotspot flag (Is_CpG intentionally omitted — dropped by RFECV)
    combined["Is_Hotspot"] = (
        combined["Hotspot"].str.lower() == "yes"
    ).astype(int)

    # ── New extended features (5) ──────────────────────────────────
    # SpliceAI delta scores (fill missing with 0.0 = no splicing impact)
    splice_cols = ["SpliceAI_DS_AG", "SpliceAI_DS_AL",
                   "SpliceAI_DS_DG", "SpliceAI_DS_DL"]
    for col in splice_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0)

    # Exon number extracted from ExonIntron column (e.g. "5-exon" → 5)
    combined["Exon_Number"] = (
        combined["ExonIntron"]
        .str.extract(r"(\d+)")[0]
        .astype(float)
        .fillna(0)
        .astype(int)
    )

    # ── Labels ─────────────────────────────────────────────────────
    label_map = {
        "functional":           0,
        "supertrans":           0,
        "non-functional":       1,
        "partially functional": 1,
    }
    combined["label"] = (
        combined["TransactivationClass"]
        .str.strip().str.lower()
        .map(label_map)
    )
    combined = combined.dropna(subset=["label"])
    combined["label"] = combined["label"].astype(int)

    X = combined[EXTENDED_PATHOGENICITY_FEATURES].copy()
    y = combined["label"]

    # ── Metadata for case-study identification ─────────────────────
    meta_cols = []
    if "MUT_ID" in combined.columns:
        meta_cols.append("MUT_ID")
    if "Individual_ID" in combined.columns:
        meta_cols.append("Individual_ID")
    if "ProtDescription" in combined.columns:
        meta_cols.append("ProtDescription")
    meta = combined[meta_cols].copy() if meta_cols else pd.DataFrame(index=combined.index)

    print(f"  Loaded {len(X)} real missense variants for extended pathogenicity training")
    print(f"  Functional: {(y == 0).sum()}, Non-functional: {(y == 1).sum()}")
    print(f"  Features ({len(EXTENDED_PATHOGENICITY_FEATURES)}): {EXTENDED_PATHOGENICITY_FEATURES}")
    return X, y, meta


# ──────────────────────────────────────────────────────────────────────
#  Training: GridSearchCV + 5-Fold CV + hold-out test evaluation
# ──────────────────────────────────────────────────────────────────────

def train_extended_pathogenicity_model(seed: int = 42) -> tuple:
    """Train the extended 12-feature Random Forest pathogenicity model.

    Mirrors ``pipeline.train_pathogenicity_model()`` exactly:
      1. 80/20 stratified train/test split
      2. GridSearchCV with 5-fold stratified CV
      3. Per-fold optimal threshold via Youden's J statistic
      4. Hold-out test evaluation with the CV-averaged threshold

    Returns
    -------
    rf_best : RandomForestClassifier – best estimator from GridSearchCV
    features : list[str] – EXTENDED_PATHOGENICITY_FEATURES
    X_test  : ndarray – held-out test features
    y_test  : ndarray – held-out test labels
    cv_results : dict – all CV and test metrics
    meta_test  : DataFrame – metadata for the test split
    """
    X, y, meta = _load_extended_pathogenicity_training_data()
    X_arr = X.values.astype(float)
    y_arr = y.values
    meta_arr = meta.reset_index(drop=True)

    # Train / test split (stratified, indices-based)
    indices = np.arange(len(X_arr))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y_arr, test_size=0.2, random_state=seed, stratify=y_arr
    )
    X_train = X_arr[idx_train]
    X_test  = X_arr[idx_test]
    meta_test = meta_arr.iloc[idx_test].reset_index(drop=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # ── 1. GridSearchCV ────────────────────────────────────────────
    print("\n  ── GridSearchCV: Tuning Extended Random Forest ──")
    rf_param_grid = {
        "n_estimators":      [200, 400, 600],
        "max_depth":         [8, 12, 16, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
    }
    rf_base = RandomForestClassifier(
        class_weight="balanced", random_state=seed, n_jobs=-1
    )

    def _make_grid_search(n_jobs: int) -> GridSearchCV:
        return GridSearchCV(
            rf_base,
            rf_param_grid,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=n_jobs,
            refit=True,
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

    # ── 2. 5-Fold CV with per-fold optimal threshold ───────────────
    print("\n  ── 5-Fold Cross-Validation: Extended Random Forest ──")
    fold_accuracies   = []
    fold_f1s          = []
    fold_aucs         = []
    fold_thresholds   = []
    fold_sensitivities = []
    fold_specificities = []

    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_tr_fold, X_val_fold = X_train[tr_idx], X_train[val_idx]
        y_tr_fold, y_val_fold = y_train[tr_idx], y_train[val_idx]

        # Clone best estimator for this fold
        fold_model = RandomForestClassifier(
            **grid_search.best_params_,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
        fold_model.fit(X_tr_fold, y_tr_fold)

        # Out-of-fold probabilities
        val_proba = fold_model.predict_proba(X_val_fold)[:, 1]

        # Optimal threshold via Youden's J statistic
        fpr_fold, tpr_fold, thresholds_fold = roc_curve(y_val_fold, val_proba)
        j_scores = tpr_fold - fpr_fold
        best_j_idx = np.argmax(j_scores)
        fold_threshold = thresholds_fold[best_j_idx]
        fold_thresholds.append(fold_threshold)

        # Per-fold metrics using the fold's optimal threshold
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

    # ── 3. Evaluate on hold-out test set ───────────────────────────
    y_test_proba = rf_best.predict_proba(X_test)[:, 1]
    y_pred = (y_test_proba >= cv_optimal_threshold).astype(int)

    print("\n  ── Extended Model: Random Forest (hold-out test evaluation) ──")
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

    # Package CV results
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

    return rf_best, EXTENDED_PATHOGENICITY_FEATURES, X_test, y_test, cv_results, meta_test


# ──────────────────────────────────────────────────────────────────────
#  Feature importance extraction (same as pipeline.py)
# ──────────────────────────────────────────────────────────────────────

def extract_feature_importances(model,
                                 feature_names: list[str],
                                 stage_name: str = "") -> pd.DataFrame:
    """Return a DataFrame of Gini importances, sorted descending."""
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    if stage_name:
        print(f"\n  ── Feature Importances: {stage_name} ──")
        for _, row in fi_df.iterrows():
            bar = "█" * int(row["Importance"] * 40)
            print(f"    {row['Feature']:<22s} {row['Importance']:.4f}  {bar}")

    return fi_df
