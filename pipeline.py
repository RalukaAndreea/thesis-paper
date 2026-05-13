import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
)
from sklearn.metrics import classification_report

from grantham import get_grantham_score, classify_grantham

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "IARC_TP53_DB")
GERMLINE_CSV = os.path.join(DATA_DIR, "GermlineDownload_r21.csv")
SOMATIC_CSV = os.path.join(DATA_DIR, "TumorVariantDownload_r21-2.csv")

AA_3TO1 = {
    "Ala": "A", "Cys": "C", "Asp": "D", "Glu": "E", "Phe": "F",
    "Gly": "G", "His": "H", "Ile": "I", "Lys": "K", "Leu": "L",
    "Met": "M", "Asn": "N", "Pro": "P", "Gln": "Q", "Arg": "R",
    "Ser": "S", "Thr": "T", "Val": "V", "Trp": "W", "Tyr": "Y",
    "STOP": "*", "Stop": "*",
}

# Encoding maps for categorical predictors
AGVGD_MAP = {"C0": 0, "C15": 1, "C25": 2, "C35": 3, "C45": 4, "C55": 5, "C65": 6}
SIFT_MAP  = {"Tolerated": 0, "Damaging": 1, "tolerated": 0, "deleterious": 1}
PP2_MAP   = {
    "B": 0,
    "P": 1,
    "D": 2,
    "benign": 0,
    "possibly_damaging": 1,
    "probably_damaging": 2,
}   # Benign / Possibly damaging / Damaging


def _load_iarc_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Missing required IARC TP53 data file: {csv_path}"
        )
    return pd.read_csv(csv_path)


def parse_vcf(filepath: str) -> pd.DataFrame:

    records = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 10:
                continue

            chrom, pos, vid, ref, alt, qual, filt, info, fmt, sample = fields[:10]

            info_dict = {}
            for entry in info.split(";"):
                if "=" in entry:
                    k, v = entry.split("=", 1)
                    info_dict[k] = v

            def _safe_float(val, default=0.0):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default

            records.append({
                "Chromosome":   chrom,
                "Position":     int(pos),
                "ID":           vid,
                "Ref":          ref,
                "Alt":          alt,
                "QUAL":         _safe_float(qual, 0.0),
                "FILTER":       filt,
                "DP":           int(info_dict.get("DP", 0)),
                "AF":           _safe_float(info_dict.get("AF", "0")),
                "AA_REF":       info_dict.get("AA_REF", "."),
                "AA_ALT":       info_dict.get("AA_ALT", "."),
                "EFFECT":       info_dict.get("EFFECT", "unknown"),
                "REVEL":        _safe_float(info_dict.get("REVEL", "."), np.nan),
                "BAYESDEL":     _safe_float(info_dict.get("BAYESDEL", "."), np.nan),
                "HOTSPOT":      info_dict.get("HOTSPOT", "no"),
                "CPG":          info_dict.get("CPG", "no"),
                "TRANSCLASS":   info_dict.get("TRANSCLASS", "."),
                "TCGA_COUNT":   int(_safe_float(info_dict.get("TCGA_COUNT", "0"), 0)),
                "AGVGD":        info_dict.get("AGVGD", "."),
                "SIFT":         info_dict.get("SIFT", "."),
                "PP2":          info_dict.get("PP2", "."),
                "DB_SOURCE":    info_dict.get("DB_SOURCE", "unknown"),
                "GT":           sample,
            })

    df = pd.DataFrame(records)
    print(f" Parsed {len(df)} variants from {os.path.basename(filepath)}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Grantham Distance
    def _calc_grantham(row):
        if row["EFFECT"] == "missense" and row["AA_REF"] != "." and row["AA_ALT"] != ".":
            score = get_grantham_score(row["AA_REF"], row["AA_ALT"])
            return score if score is not None else 0
        return 0

    df["Grantham_Score"] = df.apply(_calc_grantham, axis=1)
    df["Grantham_Category"] = df["Grantham_Score"].apply(classify_grantham)

    # Grantham is only meaningful for missense (amino acid substitutions).
    # Non-missense variants get N/A to avoid misleading display.
    non_miss = df["EFFECT"] != "missense"
    df.loc[non_miss, "Grantham_Score"] = np.nan
    df.loc[non_miss, "Grantham_Category"] = "N/A"

    # Allele frequency
    df["VAF"] = df["AF"]

    # Variant type flags
    df["Is_Missense"] = (df["EFFECT"] == "missense").astype(int)
    df["Is_Hotspot"]  = (df["HOTSPOT"].str.lower() == "yes").astype(int)
    df["Is_CpG"]      = (df["CPG"].str.lower() == "yes").astype(int)

    # Database membership flags
    df["In_ClinVar"] = (df["DB_SOURCE"].str.contains("ClinVar", case=False, na=False)).astype(int)
    df["In_COSMIC"]  = (df["DB_SOURCE"].str.contains("COSMIC",  case=False, na=False)).astype(int)

    # Fill missing REVEL & BAYESDEL
    # REVEL and BayesDel are only meaningful for missense variants.
    # For missense: fill NaN with the missense-only median.
    # For non-missense: set to 0 (neutral) to avoid misleading the model.
    df["REVEL"]    = pd.to_numeric(df["REVEL"], errors="coerce")
    df["BAYESDEL"] = pd.to_numeric(df["BAYESDEL"], errors="coerce")

    is_miss = df["EFFECT"] == "missense"
    revel_median = df.loc[is_miss, "REVEL"].median() if df.loc[is_miss, "REVEL"].notna().any() else 0.5
    bd_median    = df.loc[is_miss, "BAYESDEL"].median() if df.loc[is_miss, "BAYESDEL"].notna().any() else 0.0

    df.loc[is_miss, "REVEL"]    = df.loc[is_miss, "REVEL"].fillna(revel_median)
    df.loc[is_miss, "BAYESDEL"] = df.loc[is_miss, "BAYESDEL"].fillna(bd_median)
    df.loc[~is_miss, "REVEL"]   = 0.0
    df.loc[~is_miss, "BAYESDEL"] = 0.0

    # Encode categorical predictors
    # AGVGD, SIFT, PolyPhen-2 are also missense-only predictors.
    # Non-missense variants get neutral defaults (C0 / Tolerated / Benign).
    if "AGVGD" in df.columns:
        df["AGVGDClass"] = df["AGVGD"].astype(str).str.strip().map(AGVGD_MAP)
        df.loc[is_miss, "AGVGDClass"]  = df.loc[is_miss, "AGVGDClass"].fillna(3)
        df.loc[~is_miss, "AGVGDClass"] = 0
        df["AGVGDClass"] = df["AGVGDClass"].astype(int)
    else:
        df["AGVGDClass"] = 0
    if "SIFT" in df.columns:
        df["SIFTClass"] = df["SIFT"].astype(str).str.strip().map(SIFT_MAP)
        df.loc[is_miss, "SIFTClass"]  = df.loc[is_miss, "SIFTClass"].fillna(0)
        df.loc[~is_miss, "SIFTClass"] = 0
        df["SIFTClass"] = df["SIFTClass"].astype(int)
    else:
        df["SIFTClass"] = 0
    if "PP2" in df.columns:
        df["Polyphen2"] = df["PP2"].astype(str).str.strip().map(PP2_MAP)
        df.loc[is_miss, "Polyphen2"]  = df.loc[is_miss, "Polyphen2"].fillna(1)
        df.loc[~is_miss, "Polyphen2"] = 0
        df["Polyphen2"] = df["Polyphen2"].astype(int)
    else:
        df["Polyphen2"] = 0

    n_miss = df["Is_Missense"].sum()
    n_hot  = df["Is_Hotspot"].sum()
    print(f"  ✓ Engineered features: {n_miss} missense, {n_hot} hotspot variants")
    return df



#  STAGE 1  PATHOGENICITY MODEL (Optimized)
#  Features: Grantham, REVEL, BayesDel, AGVGDClass, SIFTClass, Polyphen2, Hotspot, CpG
PATHOGENICITY_FEATURES = [
    "Grantham_Score", "REVEL", "BAYESDEL",
    "AGVGDClass", "SIFTClass", "Polyphen2",
    "Is_Hotspot", "Is_CpG"
]


def _load_pathogenicity_training_data() -> tuple[pd.DataFrame, pd.Series]:
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

    # Calculate Grantham scores
    combined["AA_REF_1"] = combined["WT_AA"].map(AA_3TO1)
    combined["AA_ALT_1"] = combined["Mutant_AA"].map(AA_3TO1)
    combined["Grantham_Score"] = combined.apply(
        lambda r: get_grantham_score(r["AA_REF_1"], r["AA_ALT_1"])
        if pd.notna(r["AA_REF_1"]) and pd.notna(r["AA_ALT_1"]) else 0,
        axis=1
    ).fillna(0)

    # Real annotation scores
    combined["REVEL"]    = pd.to_numeric(combined["REVEL"], errors="coerce").fillna(0.5)
    combined["BAYESDEL"] = pd.to_numeric(combined["BayesDel"], errors="coerce").fillna(0.0)

    #features: AGVGDClass, SIFTClass, Polyphen2
    combined["AGVGDClass"] = combined["AGVGDClass"].astype(str).str.strip().map(AGVGD_MAP).fillna(3).astype(int)
    combined["SIFTClass"]  = combined["SIFTClass"].astype(str).str.strip().map(SIFT_MAP).fillna(0).astype(int)
    combined["Polyphen2"]  = combined["Polyphen2"].astype(str).str.strip().map(PP2_MAP).fillna(1).astype(int)

    # Hotspot & CpG flags
    combined["Is_Hotspot"] = (combined["Hotspot"].str.lower() == "yes").astype(int)
    combined["Is_CpG"]     = (combined["CpG_site"].str.lower() == "yes").astype(int)

    # Labels: functional=0, non-functional=1
    label_map = {
        "functional": 0,
        "supertrans": 0,
        "non-functional": 1,
        "partially functional": 1,
    }
    combined["label"] = combined["TransactivationClass"].str.strip().str.lower().map(label_map)
    combined = combined.dropna(subset=["label"])
    combined["label"] = combined["label"].astype(int)

    X = combined[PATHOGENICITY_FEATURES].copy()
    y = combined["label"]

    # Metadata columns for case study identification
    meta_cols = []
    if "MUT_ID" in combined.columns:
        meta_cols.append("MUT_ID")
    if "Individual_ID" in combined.columns:
        meta_cols.append("Individual_ID")
    if "ProtDescription" in combined.columns:
        meta_cols.append("ProtDescription")
    meta = combined[meta_cols].copy() if meta_cols else pd.DataFrame(index=combined.index)

    print(f" Loaded {len(X)} real missense variants for pathogenicity training")
    print(f" Functional: {(y == 0).sum()}, Non-functional: {(y == 1).sum()}")
    print(f" Features: {PATHOGENICITY_FEATURES}")
    return X, y, meta


def train_pathogenicity_model(seed: int = 42) -> tuple:
    X, y, meta = _load_pathogenicity_training_data()
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

    # 1. GridSearchCV — Random Forest
    print("\n  ── GridSearchCV: Tuning Random Forest ──")
    rf_param_grid = {
        "n_estimators":     [200, 400, 600],
        "max_depth":        [8, 12, 16, None],
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
        print("    Parallel GridSearchCV is unavailable here; retrying with n_jobs=1...")
        grid_search = _make_grid_search(n_jobs=1)
        grid_search.fit(X_train, y_train)
    rf_best = grid_search.best_estimator_

    print(f"    Best params: {grid_search.best_params_}")
    print(f"    Best CV F1:  {grid_search.best_score_:.4f}")

    # 2. 5-Fold CV — Random Forest
    print("\n  ── 5-Fold Cross-Validation: Random Forest ──")
    rf_cv = cross_validate(
        rf_best, X_train, y_train, cv=cv,
        scoring=["accuracy", "f1_weighted", "roc_auc"],
        return_train_score=False
    )
    rf_acc_mean  = rf_cv["test_accuracy"].mean()
    rf_acc_std   = rf_cv["test_accuracy"].std()
    rf_f1_mean   = rf_cv["test_f1_weighted"].mean()
    rf_f1_std    = rf_cv["test_f1_weighted"].std()
    rf_auc_mean  = rf_cv["test_roc_auc"].mean()
    rf_auc_std   = rf_cv["test_roc_auc"].std()
    print(f"    Accuracy:  {rf_acc_mean:.4f} ± {rf_acc_std:.4f}")
    print(f"    F1 Score:  {rf_f1_mean:.4f} ± {rf_f1_std:.4f}")
    print(f"    AUC:       {rf_auc_mean:.4f} ± {rf_auc_std:.4f}")

    # 3. Evaluate on hold-out test set
    y_pred = rf_best.predict(X_test)
    print("\n  ── Stage 1: Random Forest (hold-out test evaluation) ──")
    print(classification_report(
        y_test, y_pred,
        target_names=["Functional", "Non-functional"],
        zero_division=0
    ))

    # Package CV results for visualisations
    cv_results = {
        "rf_accuracy":  (rf_acc_mean,  rf_acc_std),
        "rf_f1":        (rf_f1_mean,   rf_f1_std),
        "rf_auc":       (rf_auc_mean,  rf_auc_std),
        "best_model":   "Random Forest",
        "best_params":  grid_search.best_params_,
    }

    return rf_best, PATHOGENICITY_FEATURES, X_test, y_test, cv_results, meta_test



def extract_feature_importances(model,
                                 feature_names: list[str],
                                 stage_name: str = "") -> pd.DataFrame:
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    if stage_name:
        print(f"\n  ── Feature Importances: {stage_name} ──")
        for _, row in fi_df.iterrows():
            bar = "█" * int(row["Importance"] * 40)
            print(f"    {row['Feature']:<22s} {row['Importance']:.4f}  {bar}")

    return fi_df


# CLASSIFICATION — APPLY MODELS TO PATIENT VARIANTS

PATHOGENICITY_LABELS = {0: "Functional", 1: "Non-functional"}
ORIGIN_LABELS        = {0: "Germline",   1: "Somatic"}


def classify_variants(df: pd.DataFrame,
                       path_model) -> pd.DataFrame:
    df = df.copy()

    # Pathogenicity Classification
    X_path = df[PATHOGENICITY_FEATURES].values.astype(float)
    df["Pathogenicity_Pred_Code"] = path_model.predict(X_path)
    df["Pathogenicity_Prediction"] = df["Pathogenicity_Pred_Code"].map(PATHOGENICITY_LABELS)

    proba_path = path_model.predict_proba(X_path)
    df["Pathogenicity_Confidence"] = proba_path.max(axis=1).round(3)

    # Rule-based override for loss-of-function variants
    # Nonsense (stop-gain), frameshift, and splice variants always
    # destroy or truncate the protein → always Non-functional.
    # The model was trained on missense only and cannot predict these.
    lof_mask = df["EFFECT"].isin(["nonsense", "frameshift", "splice"])
    df.loc[lof_mask, "Pathogenicity_Pred_Code"] = 1
    df.loc[lof_mask, "Pathogenicity_Prediction"] = "Non-functional"
    df.loc[lof_mask, "Pathogenicity_Confidence"] = 1.0

    # Rule-based override for benign variants
    # Silent (synonymous) → no amino acid change → protein unchanged.
    # Intronic → outside coding region → usually no impact.
    benign_mask = df["EFFECT"].isin(["silent", "intronic"])
    df.loc[benign_mask, "Pathogenicity_Pred_Code"] = 0
    df.loc[benign_mask, "Pathogenicity_Prediction"] = "Functional"
    df.loc[benign_mask, "Pathogenicity_Confidence"] = 1.0

    # Rule-based override for large deletions
    # Deletes a significant portion of the gene → protein truncated or absent.
    del_mask = df["EFFECT"].isin(["large_deletion", "deletion"])
    df.loc[del_mask, "Pathogenicity_Pred_Code"] = 1
    df.loc[del_mask, "Pathogenicity_Prediction"] = "Non-functional"
    df.loc[del_mask, "Pathogenicity_Confidence"] = 1.0

    # Flag unknown/other effect types for manual review
    known_effects = [
        "missense", "nonsense", "frameshift", "splice",
        "silent", "intronic", "large_deletion", "deletion",
    ]
    other_mask = ~df["EFFECT"].isin(known_effects)
    df.loc[other_mask, "Pathogenicity_Prediction"] = "Unknown"
    df.loc[other_mask, "Pathogenicity_Confidence"] = 0.0

    n_func = (df["Pathogenicity_Pred_Code"] == 0).sum()
    n_nonfunc = (df["Pathogenicity_Pred_Code"] == 1).sum()
    print(f"\n  ✓ Classification complete:")
    print(f"    Functional: {n_func}, Non-functional: {n_nonfunc}")

    return df


OUTPUT_COLUMNS = [
    "Chromosome", "Position", "Ref", "Alt",
    "EFFECT", "AA_REF", "AA_ALT",
    "Grantham_Score", "Grantham_Category",
    "REVEL", "BAYESDEL",
    "Pathogenicity_Prediction", "Pathogenicity_Confidence",
]


def export_results(df: pd.DataFrame, output_path: str) -> str:

    out = df[OUTPUT_COLUMNS].copy()
    out.to_csv(output_path, index=False)
    print(f"\n  ✓ Results exported to: {output_path}")
    print(f"    → {len(out)} variants, {len(OUTPUT_COLUMNS)} columns")
    return output_path
