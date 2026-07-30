#!/usr/bin/env python3
import os, sys
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from pipeline import train_pathogenicity_model
from pipeline_xgboost import train_pathogenicity_model_xgb

def main():
    print("Training RF Baseline...")
    rf_model, rf_feats, _, _, _, _ = train_pathogenicity_model(seed=42)
    rf_importances = rf_model.feature_importances_
    
    print("Training XGBoost Baseline...")
    xgb_model, xgb_feats, _, _, _, _ = train_pathogenicity_model_xgb(seed=42)
    # XGBoost feature_importances_ in sklearn API is Gain by default
    xgb_importances = xgb_model.feature_importances_
    
    # Ensure features match
    features = list(rf_feats)
    if list(xgb_feats) != features:
        print("Feature mismatch!")
    
    # Create DataFrame
    df = pd.DataFrame({
        "Feature": features,
        "RF_Gini": rf_importances,
        "XGBoost_Gain": xgb_importances
    })
    
    # Sort by RF importance descending for a logical order
    df = df.sort_values(by="RF_Gini", ascending=False).reset_index(drop=True)
    
    print("\n--- TABLE 13: FEATURE IMPORTANCES ---")
    print("| Feature | RF Baseline (Gini) | XGBoost Baseline (Gain) |")
    print("|---|---|---|")
    for _, row in df.iterrows():
        print(f"| {row['Feature']} | {row['RF_Gini']:.4f} | {row['XGBoost_Gain']:.4f} |")

if __name__ == "__main__":
    main()
