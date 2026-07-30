import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score

# Import baseline functions and constants
from pipeline import (
    _load_iarc_csv, GERMLINE_CSV, SOMATIC_CSV,
    AA_3TO1, AGVGD_MAP, SIFT_MAP, PP2_MAP
)
from grantham import get_grantham_score

SEED = 42


def load_extended_training_data():
    germ = _load_iarc_csv(GERMLINE_CSV)
    soma = _load_iarc_csv(SOMATIC_CSV)

    # Filter to missense only and combine
    germ_miss = germ[germ["Effect"] == "missense"].copy()
    soma_miss = soma[soma["Effect"] == "missense"].copy()
    combined = pd.concat([germ_miss, soma_miss], ignore_index=True)

    # Deduplicate by unique mutation
    combined = combined.drop_duplicates(subset=["WT_AA", "Mutant_AA", "Codon_number"]).copy()

    # --- 1. BASELINE 8 FEATURES ---
    combined["AA_REF_1"] = combined["WT_AA"].map(AA_3TO1)
    combined["AA_ALT_1"] = combined["Mutant_AA"].map(AA_3TO1)
    combined["Grantham_Score"] = combined.apply(
        lambda r: get_grantham_score(r["AA_REF_1"], r["AA_ALT_1"])
        if pd.notna(r["AA_REF_1"]) and pd.notna(r["AA_ALT_1"]) else 0,
        axis=1
    ).fillna(0)
    combined["REVEL"] = pd.to_numeric(combined["REVEL"], errors="coerce").fillna(0.5)
    combined["BAYESDEL"] = pd.to_numeric(combined["BayesDel"], errors="coerce").fillna(0.0)
    combined["AGVGDClass"] = combined["AGVGDClass"].astype(str).str.strip().map(AGVGD_MAP).fillna(3).astype(int)
    combined["SIFTClass"]  = combined["SIFTClass"].astype(str).str.strip().map(SIFT_MAP).fillna(0).astype(int)
    combined["Polyphen2"]  = combined["Polyphen2"].astype(str).str.strip().map(PP2_MAP).fillna(1).astype(int)
    combined["Is_Hotspot"] = (combined["Hotspot"].str.lower() == "yes").astype(int)
    combined["Is_CpG"]     = (combined["CpG_site"].str.lower() == "yes").astype(int)

    splice_cols = ["SpliceAI_DS_AG", "SpliceAI_DS_AL", "SpliceAI_DS_DG", "SpliceAI_DS_DL"]
    for col in splice_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0)

    # ExonIntron (Values look like "5-exon", "6-intron". Extract the digit)
    combined["Exon_Number"] = combined["ExonIntron"].str.extract(r'(\d+)').astype(float).fillna(0).astype(int)

    # Labels: functional=0, non-functional=1
    label_map = {
        "functional": 0, "supertrans": 0,
        "non-functional": 1, "partially functional": 1,
    }
    combined["label"] = combined["TransactivationClass"].str.strip().str.lower().map(label_map)
    combined = combined.dropna(subset=["label"])
    combined["label"] = combined["label"].astype(int)

    # Final feature list
    feature_cols = [
        "Grantham_Score", "REVEL", "BAYESDEL", "AGVGDClass", "SIFTClass", "Polyphen2",
        "Is_Hotspot", "Is_CpG",
        "SpliceAI_DS_AG", "SpliceAI_DS_AL", "SpliceAI_DS_DG", "SpliceAI_DS_DL",
        "Exon_Number"
    ]

    X = combined[feature_cols].copy()
    y = combined["label"].copy()

    print(f" Loaded {len(X)} variants with {len(feature_cols)} features.")
    print(f" Functional: {(y == 0).sum()}, Non-functional: {(y == 1).sum()}")
    return X, y


def main():
    print("=" * 60)
    print("  EXTENDED AUTOMATED FEATURE SELECTION (with SpliceAI & Exon)")
    print("=" * 60)

    X, y = load_extended_training_data()
    out_dir = "Feature_importance"
    os.makedirs(out_dir, exist_ok=True)

    print("\n▶ Splitting data: 80% train / 20% held-out test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    print("\n Running Recursive Feature Elimination (RFECV) on training data...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rf_for_rfecv = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=4,
        min_samples_split=2, class_weight='balanced', random_state=SEED, n_jobs=-1
    )

    rfecv = RFECV(
        estimator=rf_for_rfecv, step=1, cv=cv,
        scoring='f1_weighted', min_features_to_select=1, n_jobs=-1
    )
    rfecv.fit(X_train, y_train)

    # RFECV Plot
    plt.figure(figsize=(10, 6))
    scores = rfecv.cv_results_['mean_test_score'] if hasattr(rfecv, 'cv_results_') else rfecv.grid_scores_
    plt.plot(range(1, len(scores) + 1), scores, marker='o', linewidth=2, color='#2196F3')
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Cross-Validation F1 Score (on training folds)")
    plt.title("RFECV — Optimal Feature Subset Selection")
    plt.grid(True, linestyle='--', alpha=0.7)

    opt_n = rfecv.n_features_
    opt_score = scores[opt_n - 1]
    plt.plot(opt_n, opt_score, 'r*', markersize=15, label=f'Optimal: {opt_n} features (F1={opt_score:.4f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/extended_rfecv.png", dpi=300)
    plt.close()

    # Extract selected / dropped features
    all_feats = list(X.columns)
    selected_features = [f for f, s in zip(all_feats, rfecv.support_) if s]
    dropped_features  = [f for f, s in zip(all_feats, rfecv.support_) if not s]

    print(f"\n Retraining Random Forest with the {opt_n} selected features...")
    X_train_sel = X_train[selected_features]
    X_test_sel  = X_test[selected_features]

    rf_final = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=4,
        min_samples_split=2, class_weight='balanced', random_state=SEED, n_jobs=-1
    )
    rf_final.fit(X_train_sel, y_train)

    print(" Calculating Permutation Importance on held-out test set...")
    perm_importance = permutation_importance(
        rf_final, X_test_sel, y_test,
        n_repeats=30, random_state=SEED, scoring='f1_weighted', n_jobs=-1
    )

    sorted_idx = perm_importance.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.boxplot(
        perm_importance.importances[sorted_idx].T,
        vert=False,
        tick_labels=X_test_sel.columns[sorted_idx]
    )
    ax.set_title("Permutation Importance on Held-Out Test Set (Unbiased)")
    ax.set_xlabel("Decrease in F1 Score when feature is shuffled")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/extended_permutation_importance.png", dpi=300)
    plt.close()

    y_pred = rf_final.predict(X_test_sel)
    y_proba = rf_final.predict_proba(X_test_sel)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    test_f1  = f1_score(y_test, y_pred, average='weighted')
    test_auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 60)
    print("  EXTENDED FEATURE JUSTIFICATION REPORT")
    print("=" * 60)

    print(f"\nRFECV determined the optimal number of features: {opt_n} (out of {len(all_feats)})")

    print(f"\n Features Kept ({len(selected_features)}):")
    for f in selected_features:
        print(f"  - {f}")

    print(f"\n Features Dropped by algorithm ({len(dropped_features)}):")
    if dropped_features:
        for f in dropped_features:
            print(f"  - {f}")
    else:
        print("  None! Every single feature was useful.")

    print("\n Feature Rankings — Permutation Importance (on held-out test set):")
    for idx in sorted_idx[::-1]:
        print(f"  {X_test_sel.columns[idx]:<18s}: "
              f"{perm_importance.importances_mean[idx]:.4f} ± "
              f"{perm_importance.importances_std[idx]:.4f}")

    print("\n Held-Out Test Set Performance (model never saw this data):")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  AUC:      {test_auc:.4f}")
    print("\n" + classification_report(
        y_test, y_pred,
        target_names=["Functional", "Non-functional"],
        zero_division=0
    ))

    print(f"📁 Plots saved to: {os.path.abspath(out_dir)}/")


if __name__ == '__main__':
    main()
