#!/usr/bin/env python3
import os
import sys
import json

# Ensure the project directory is on the Python path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
GENERATED_VCF_DIR = os.path.join(PROJECT_DIR, "VCF", "Generated vcf files")

from VCF.gen_vcf_files.generate_vcf import generate_germline_vcf, generate_somatic_vcf, generate_patient_vcf
from pipeline_revel import (
    parse_vcf,
    engineer_features,
    train_pathogenicity_model,
    extract_feature_importances,
    classify_variants,
    export_results,
)



def main():

    print("=" * 65)
    print("  TP53 Variant Classification Pipeline (Optimized)")
    print("=" * 65)


    # Step 4: Train Stage 1 Pathogenicity (GridSearchCV + 5-fold CV)
    print("\n▶ Step 4: Training Pathogenicity classifier (optimized)...")
    path_model, path_features, path_X_test, path_y_test, cv_results, meta_test = (
        train_pathogenicity_model()
    )

   # Step 7: Feature importances
    print("\n▶ Step 7: Extracting feature importances...")
    fii_pathogenicity = extract_feature_importances(
        path_model, path_features, stage_name="Stage 1 — Pathogenicity"
    )

 # Save feature importances to CSV
    fi_path_csv    = os.path.join(PROJECT_DIR, "feature_importances_pathogenicity_RF_without_BAYESDEL.csv")
    fii_pathogenicity.to_csv(fi_path_csv, index=False)

    print(f"\n  ✓ Feature importances saved to:")
    print(f"    → {fi_path_csv}")



if __name__ == "__main__":
    main()
