#!/usr/bin/env python3
import os
import sys
import json

# Ensure the project directory is on the Python path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
GENERATED_VCF_DIR = os.path.join(PROJECT_DIR, "VCF", "Generated vcf files")

from VCF.gen_vcf_files.generate_vcf import generate_germline_vcf, generate_somatic_vcf, generate_patient_vcf
from pipeline import (
    parse_vcf,
    engineer_features,
    train_pathogenicity_model,
    extract_feature_importances,
    classify_variants,
    export_results,
)
from PLOTS.visualizations import generate_all_plots


def main():

    print("=" * 65)
    print("  TP53 Variant Classification Pipeline (Optimized)")
    print("=" * 65)

    # Step 1: Generate synthetic VCF files
    print("\n▶ Step 1: Generating synthetic VCFs (sampled from real data)...")
    os.makedirs(GENERATED_VCF_DIR, exist_ok=True)
    germline_vcf = os.path.join(GENERATED_VCF_DIR, "mock_germline.vcf")
    somatic_vcf = os.path.join(GENERATED_VCF_DIR, "mock_somatic.vcf")
    patient_vcf = os.path.join(GENERATED_VCF_DIR, "mock_patient.vcf")

    generate_germline_vcf(germline_vcf, n_variants=50)
    generate_somatic_vcf(somatic_vcf,   n_variants=80)
    generate_patient_vcf(patient_vcf,   n_germline=20, n_somatic=30)

    # Step 2: Parse the patient VCF
    print("\n▶ Step 2: Parsing patient VCF...")
    variants_df = parse_vcf(patient_vcf)

    #Step 3: Feature engineering
    print("\n▶ Step 3: Engineering features...")
    features_df = engineer_features(variants_df)

    # Step 4: Train Stage 1 Pathogenicity (GridSearchCV + 5-fold CV)
    print("\n▶ Step 4: Training Pathogenicity classifier (optimized)...")
    path_model, path_features, path_X_test, path_y_test, cv_results = (
        train_pathogenicity_model()
    )

    # Save CV comparison results
    cv_json = os.path.join(PROJECT_DIR, "cv_model_comparison.json")
    cv_serializable = {}
    for k, v in cv_results.items():
        if isinstance(v, tuple):
            cv_serializable[k] = {"mean": round(v[0], 4), "std": round(v[1], 4)}
        else:
            cv_serializable[k] = v
    with open(cv_json, "w") as f:
        json.dump(cv_serializable, f, indent=2, default=str)
    print(f"\n  ✓ CV comparison saved to: {cv_json}")



    # Step 6: Classify patient variants
    print("\n▶ Step 6: Classifying patient variants...")
    results_df = classify_variants(features_df, path_model)

    # Step 7: Feature importances
    print("\n▶ Step 7: Extracting feature importances...")
    fi_pathogenicity = extract_feature_importances(
        path_model, path_features, stage_name="Stage 1 — Pathogenicity"
    )


    # Save feature importances to CSV
    fi_path_csv    = os.path.join(PROJECT_DIR, "feature_importances_pathogenicity.csv")
    fi_pathogenicity.to_csv(fi_path_csv, index=False)

    print(f"\n  ✓ Feature importances saved to:")
    print(f"    → {fi_path_csv}")

    # Step 8: Export final results
    print("\n▶ Step 8: Exporting results...")
    output_csv = os.path.join(PROJECT_DIR, "pipeline_results.csv")
    export_results(results_df, output_csv)

    # Step 9: Generate thesis visualizations
    print("\n▶ Step 9: Generating thesis visualizations...")
    plots = generate_all_plots(
        results_df=results_df,
        path_model=path_model,
        path_features=path_features,
        path_X_test=path_X_test,
        path_y_test=path_y_test,
        output_dir=PROJECT_DIR,
    )

    # Summary
    print("\n" + "=" * 65)
    print("  Pipeline complete!")
    print("=" * 65)

    # Model summary
    rf_f1  = cv_results["rf_f1"]
    print(f"\n  ── Stage 1 Model (5-fold CV) ──")
    print(f"    Random Forest:  F1 = {rf_f1[0]:.4f} ± {rf_f1[1]:.4f}")

    print(f"\n  Output files:")
    print(f"    • pipeline_results.csv     — Final classified variants")
    print(f"    • feature_importances_pathogenicity.csv")

    print(f"\n  Thesis plots ({len(plots)}):")
    for p in plots:
        print(f"    • {os.path.basename(p)}")
    print()


    print("  ── Results Preview (first 5 rows) ──")
    preview_cols = [
        "Chromosome", "Position", "Ref", "Alt",
        "Grantham_Score", "VAF", "REVEL",
        "Pathogenicity_Prediction"
    ]
    print(results_df[preview_cols].head().to_string(index=False))
    print()


if __name__ == "__main__":
    main()
