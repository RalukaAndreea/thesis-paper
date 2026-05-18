# pipeline_xgboost.py — XGBoost pathogenicity classifier (Stage 1)
# ─────────────────────────────────────────────────────────────────
# Requirements:  pip install xgboost optuna
# ─────────────────────────────────────────────────────────────────
#
# This module trains an XGBoost classifier on the same TP53 missense
# data used by the Random Forest in pipeline.py, tuned with Optuna
# instead of GridSearchCV.  It is intentionally kept in a separate
# file so that pipeline.py, run_pipeline.py, and visualizations.py
# remain untouched.

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    recall_score,
)
import xgboost as xgb
import optuna

# ── Re-use the exact same data loader & feature list from the RF pipeline ──
from pipeline import _load_pathogenicity_training_data, PATHOGENICITY_FEATURES

SEED = 42


# ──────────────────────────────────────────────────────────────────
#  Data loading & train/test split  (byte-identical to pipeline.py)
# ──────────────────────────────────────────────────────────────────

def load_and_split(seed: int = SEED):
    """Return X_train, X_test, y_train, y_test using the same
    index-based split as the Random Forest pipeline."""
    X, y, meta = _load_pathogenicity_training_data()
    X_arr = X.values.astype(float)
    y_arr = y.values

    indices = np.arange(len(X_arr))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y_arr, test_size=0.2, random_state=seed, stratify=y_arr
    )
    X_train = X_arr[idx_train]
    X_test  = X_arr[idx_test]
    meta_test = meta.reset_index(drop=True).iloc[idx_test].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, meta_test


# ──────────────────────────────────────────────────────────────────
#  Optuna objective  (mean weighted-F1 over 5 stratified folds)
# ──────────────────────────────────────────────────────────────────

def _make_objective(X_train, y_train, cv, seed: int = SEED):
    """Return a closure that Optuna can call as its objective."""

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        fold_f1s = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            # Per-fold scale_pos_weight (mirrors class_weight="balanced")
            n_neg = int((y_tr == 0).sum())
            n_pos = int((y_tr == 1).sum())
            spw = n_neg / n_pos if n_pos > 0 else 1.0

            model = xgb.XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=spw,
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr, verbose=False)
            y_pred = model.predict(X_val)
            fold_f1s.append(f1_score(y_val, y_pred, average="weighted"))

        return float(np.mean(fold_f1s))

    return objective


# ──────────────────────────────────────────────────────────────────
#  Optuna tuning  →  refit best model on full training set
# ──────────────────────────────────────────────────────────────────

def tune_xgboost(seed: int = SEED):
    """Run Optuna HPO and return (xgb_best, X_train, X_test,
    y_train, y_test, study, cv, meta_test)."""

    # 1. Data
    X_train, X_test, y_train, y_test, meta_test = load_and_split(seed)

    # 2. CV object (same as RF)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 3. Optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(_make_objective(X_train, y_train, cv, seed), n_trials=50)

    print("\n  ── Optuna Tuning Complete ──")
    print(f"    Best params: {study.best_params}")
    print(f"    Best CV F1 (weighted): {study.best_value:.4f}")

    # 4. Refit on full training set with best hyper-parameters
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw_full = n_neg / n_pos if n_pos > 0 else 1.0

    xgb_best = xgb.XGBClassifier(
        **study.best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=spw_full,
        random_state=seed,
        n_jobs=-1,
    )
    xgb_best.fit(X_train, y_train, verbose=False)

    return xgb_best, X_train, X_test, y_train, y_test, study, cv, meta_test


# ──────────────────────────────────────────────────────────────────
#  Full pipeline: tune → CV evaluate → test evaluate
#  Return signature mirrors pipeline.train_pathogenicity_model
# ──────────────────────────────────────────────────────────────────

def train_pathogenicity_model_xgb(seed: int = SEED) -> tuple:
    """End-to-end XGBoost training pipeline.

    Returns
    -------
    (xgb_best, PATHOGENICITY_FEATURES, X_test, y_test, cv_results, meta_test)
        Same shape as pipeline.train_pathogenicity_model so both can be
        consumed by the same downstream code.
    """

    # ── 1. Optuna HPO + refit ────────────────────────────────────
    (xgb_best, X_train, X_test,
     y_train, y_test, study, cv, meta_test) = tune_xgboost(seed)

    # ── 2. 5-Fold CV — XGBoost + per-fold optimal threshold ─────
    print("\n  ── 5-Fold Cross-Validation: XGBoost ──")
    fold_accuracies = []
    fold_f1s = []
    fold_aucs = []
    fold_thresholds = []
    fold_sensitivities = []
    fold_specificities = []

    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_tr_fold, X_val_fold = X_train[tr_idx], X_train[val_idx]
        y_tr_fold, y_val_fold = y_train[tr_idx], y_train[val_idx]

        # Per-fold scale_pos_weight
        n_neg = int((y_tr_fold == 0).sum())
        n_pos = int((y_tr_fold == 1).sum())
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        fold_model = xgb.XGBClassifier(
            **study.best_params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=spw,
            random_state=seed,
            n_jobs=-1,
        )
        fold_model.fit(X_tr_fold, y_tr_fold, verbose=False)

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
        # Sensitivity = TP/(TP+FN): recall of class 1 (Non-functional)
        fold_sensitivities.append(recall_score(y_val_fold, val_pred, pos_label=1, zero_division=0))
        # Specificity = TN/(TN+FP): recall of class 0 (Functional)
        fold_specificities.append(recall_score(y_val_fold, val_pred, pos_label=0, zero_division=0))

        print(f"    Fold {fold_idx}: threshold={fold_threshold:.4f}, "
              f"Acc={fold_accuracies[-1]:.4f}, AUC={fold_aucs[-1]:.4f}, "
              f"Sens={fold_sensitivities[-1]:.4f}, Spec={fold_specificities[-1]:.4f}")

    xgb_acc_mean = np.mean(fold_accuracies)
    xgb_acc_std  = np.std(fold_accuracies)
    xgb_f1_mean  = np.mean(fold_f1s)
    xgb_f1_std   = np.std(fold_f1s)
    xgb_auc_mean = np.mean(fold_aucs)
    xgb_auc_std  = np.std(fold_aucs)
    xgb_sens_mean = np.mean(fold_sensitivities)
    xgb_sens_std  = np.std(fold_sensitivities)
    xgb_spec_mean = np.mean(fold_specificities)
    xgb_spec_std  = np.std(fold_specificities)
    cv_optimal_threshold = float(np.mean(fold_thresholds))

    print(f"\n    Accuracy:     {xgb_acc_mean:.4f} ± {xgb_acc_std:.4f}")
    print(f"    F1 Score:     {xgb_f1_mean:.4f} ± {xgb_f1_std:.4f}")
    print(f"    AUC:          {xgb_auc_mean:.4f} ± {xgb_auc_std:.4f}")
    print(f"    Sensitivity:  {xgb_sens_mean:.4f} ± {xgb_sens_std:.4f}")
    print(f"    Specificity:  {xgb_spec_mean:.4f} ± {xgb_spec_std:.4f}")
    print(f"    CV Optimal Threshold: {cv_optimal_threshold:.4f} "
          f"(per-fold: {[round(t, 4) for t in fold_thresholds]})")

    # ── 3. Held-out test evaluation ──────────────────────────────
    y_test_proba = xgb_best.predict_proba(X_test)[:, 1]
    y_pred = (y_test_proba >= cv_optimal_threshold).astype(int)
    print("\n  ── Stage 1: XGBoost (hold-out test evaluation) ──")
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

    # ── 4. Package CV results ────────────────────────────────────
    cv_results = {
        "xgb_accuracy":        (xgb_acc_mean, xgb_acc_std),
        "xgb_f1":              (xgb_f1_mean,  xgb_f1_std),
        "xgb_auc":             (xgb_auc_mean, xgb_auc_std),
        "xgb_sensitivity":     (xgb_sens_mean, xgb_sens_std),
        "xgb_specificity":     (xgb_spec_mean, xgb_spec_std),
        "test_sensitivity":    float(test_sens),
        "test_specificity":    float(test_spec),
        "best_model":          "XGBoost (Optuna)",
        "best_params":         study.best_params,
        "optimal_threshold":   cv_optimal_threshold,
        "per_fold_thresholds": fold_thresholds,
        "n_trials":            50,
    }

    return xgb_best, PATHOGENICITY_FEATURES, X_test, y_test, cv_results, meta_test


# ──────────────────────────────────────────────────────────────────
#  Quick sanity check when run directly
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    xgb_best, feats, X_test, y_test, cv_results, meta_test = (
        train_pathogenicity_model_xgb()
    )
