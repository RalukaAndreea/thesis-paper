#!/usr/bin/env python3
"""
Compare RF Baseline vs XGBoost Baseline predictions on the hold-out test set.
Identify all variants where the two models DISAGREE and export details.
Does NOT modify any existing code.
"""
import os, sys, json
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from pipeline import train_pathogenicity_model, PATHOGENICITY_FEATURES, _load_pathogenicity_training_data
from pipeline_xgboost import tune_xgboost

SEED = 42
OUTPUT_DIR = os.path.join(PROJECT_DIR, "shap_extracted_values")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=" * 70)
    print("  RF vs XGBoost — Disagreement Analysis on Test Set")
    print("=" * 70)

    # ── 1. Train RF Baseline ──
    print("\n▶ Training RF Baseline...")
    rf_model, rf_feats, rf_Xtest, rf_ytest, rf_cv, rf_meta = train_pathogenicity_model(seed=SEED)
    rf_threshold = rf_cv["optimal_threshold"]
    print(f"  RF threshold: {rf_threshold:.4f}")

    # ── 2. Train XGBoost Baseline ──
    print("\n▶ Training XGBoost Baseline...")
    xgb_model, xgb_Xtrain, xgb_Xtest, xgb_ytrain, xgb_ytest, study, cv, xgb_meta = tune_xgboost(seed=SEED)

    # XGBoost needs its own CV threshold (replicate the logic from pipeline_xgboost.py)
    from sklearn.metrics import roc_curve, recall_score
    from sklearn.model_selection import StratifiedKFold
    import xgboost as xgb

    cv_obj = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_thresholds = []
    for tr_idx, val_idx in cv_obj.split(xgb_Xtrain, xgb_ytrain):
        X_tr, X_val = xgb_Xtrain[tr_idx], xgb_Xtrain[val_idx]
        y_tr, y_val = xgb_ytrain[tr_idx], xgb_ytrain[val_idx]
        n_neg = int((y_tr == 0).sum())
        n_pos = int((y_tr == 1).sum())
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        fold_model = xgb.XGBClassifier(
            **study.best_params,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", scale_pos_weight=spw,
            random_state=SEED, n_jobs=-1,
        )
        fold_model.fit(X_tr, y_tr, verbose=False)
        val_proba = fold_model.predict_proba(X_val)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val, val_proba)
        j_scores = tpr - fpr
        fold_thresholds.append(thresholds[np.argmax(j_scores)])

    xgb_threshold = float(np.mean(fold_thresholds))
    print(f"  XGBoost threshold: {xgb_threshold:.4f}")

    # ── 3. Get probabilities on test set ──
    rf_proba = rf_model.predict_proba(rf_Xtest)[:, 1]
    xgb_proba = xgb_model.predict_proba(xgb_Xtest)[:, 1]

    rf_pred = (rf_proba >= rf_threshold).astype(int)
    xgb_pred = (xgb_proba >= xgb_threshold).astype(int)

    # ── 4. Build comparison DataFrame ──
    feature_names = list(PATHOGENICITY_FEATURES)
    df = pd.DataFrame(rf_Xtest, columns=feature_names)

    # Add meta columns
    for col in rf_meta.columns:
        df[col] = rf_meta[col].values

    df["True_Label"] = rf_ytest
    df["True_Class"] = df["True_Label"].map({0: "Functional", 1: "Non-functional"})
    df["RF_Proba"] = np.round(rf_proba, 4)
    df["RF_Pred"] = rf_pred
    df["RF_Class"] = df["RF_Pred"].map({0: "Functional", 1: "Non-functional"})
    df["RF_Correct"] = (rf_pred == rf_ytest)
    df["XGB_Proba"] = np.round(xgb_proba, 4)
    df["XGB_Pred"] = xgb_pred
    df["XGB_Class"] = df["XGB_Pred"].map({0: "Functional", 1: "Non-functional"})
    df["XGB_Correct"] = (xgb_pred == rf_ytest)
    df["Agree"] = (rf_pred == xgb_pred)

    # ── 5. Summary statistics ──
    n_total = len(df)
    n_agree = df["Agree"].sum()
    n_disagree = n_total - n_agree

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {n_total} test variants")
    print(f"{'=' * 70}")
    print(f"  Agree:    {n_agree} ({100*n_agree/n_total:.1f}%)")
    print(f"  Disagree: {n_disagree} ({100*n_disagree/n_total:.1f}%)")

    # ── 6. Analyze disagreements ──
    disagree = df[~df["Agree"]].copy()

    if len(disagree) > 0:
        # Classify disagreement types
        disagree["Type"] = "unknown"
        # RF correct, XGB wrong
        mask_rf_wins = disagree["RF_Correct"] & ~disagree["XGB_Correct"]
        disagree.loc[mask_rf_wins, "Type"] = "RF_correct_XGB_wrong"
        # XGB correct, RF wrong
        mask_xgb_wins = ~disagree["RF_Correct"] & disagree["XGB_Correct"]
        disagree.loc[mask_xgb_wins, "Type"] = "XGB_correct_RF_wrong"
        # Both wrong (different wrong answers — shouldn't happen in binary, but just in case)
        mask_both_wrong = ~disagree["RF_Correct"] & ~disagree["XGB_Correct"]
        disagree.loc[mask_both_wrong, "Type"] = "Both_wrong_different"

        print(f"\n  Disagreement breakdown:")
        for dtype in ["RF_correct_XGB_wrong", "XGB_correct_RF_wrong", "Both_wrong_different"]:
            count = (disagree["Type"] == dtype).sum()
            if count > 0:
                print(f"    {dtype}: {count}")

        print(f"\n{'=' * 70}")
        print(f"  DETAILED DISAGREEMENTS")
        print(f"{'=' * 70}")

        cols_to_show = ["ProtDescription", "True_Class",
                        "RF_Proba", "RF_Class", "RF_Correct",
                        "XGB_Proba", "XGB_Class", "XGB_Correct",
                        "Type"] + feature_names

        available_cols = [c for c in cols_to_show if c in disagree.columns]

        for i, (idx, row) in enumerate(disagree.iterrows()):
            prot = row.get("ProtDescription", f"Variant_{idx}")
            print(f"\n  ── Variant {i+1}: {prot} ──")
            print(f"     True label:   {row['True_Class']}")
            print(f"     RF:           {row['RF_Class']} (proba={row['RF_Proba']:.4f}) {'✓' if row['RF_Correct'] else '✗'}")
            print(f"     XGBoost:      {row['XGB_Class']} (proba={row['XGB_Proba']:.4f}) {'✓' if row['XGB_Correct'] else '✗'}")
            print(f"     Winner:       {row['Type']}")
            print(f"     Features:")
            for fname in feature_names:
                print(f"       {fname:<18s} = {row[fname]:.4f}")

        # ── 7. Save full results ──
        # Save all test variants with predictions
        all_path = os.path.join(OUTPUT_DIR, "rf_vs_xgb_all_test_predictions.csv")
        df.to_csv(all_path, index=False)
        print(f"\n  ✓ All predictions saved: {all_path}")

        # Save disagreements only
        disagree_path = os.path.join(OUTPUT_DIR, "rf_vs_xgb_disagreements.csv")
        disagree.to_csv(disagree_path, index=False)
        print(f"  ✓ Disagreements saved: {disagree_path}")

        # Save summary JSON
        summary = {
            "n_total": int(n_total),
            "n_agree": int(n_agree),
            "n_disagree": int(n_disagree),
            "pct_agreement": round(100 * n_agree / n_total, 2),
            "rf_threshold": round(rf_threshold, 4),
            "xgb_threshold": round(xgb_threshold, 4),
            "rf_correct_xgb_wrong": int((disagree["Type"] == "RF_correct_XGB_wrong").sum()),
            "xgb_correct_rf_wrong": int((disagree["Type"] == "XGB_correct_RF_wrong").sum()),
            "disagreement_variants": [
                {
                    "variant": row.get("ProtDescription", f"idx_{idx}"),
                    "true_label": row["True_Class"],
                    "rf_proba": float(row["RF_Proba"]),
                    "rf_class": row["RF_Class"],
                    "xgb_proba": float(row["XGB_Proba"]),
                    "xgb_class": row["XGB_Class"],
                    "winner": row["Type"],
                }
                for idx, row in disagree.iterrows()
            ],
        }
        json_path = os.path.join(OUTPUT_DIR, "rf_vs_xgb_disagreement_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Summary JSON saved: {json_path}")

    else:
        print("\n  ⚠ No disagreements found! Both models predict identically.")

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
