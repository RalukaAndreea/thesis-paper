#!/usr/bin/env python3
"""
Extract SHAP statistics conditioned on feature value zones for dependence plots.
Covers REVEL, BAYESDEL, and Is_Hotspot on the RF Baseline test set.
Does NOT modify any existing code.
"""
import os, sys, json
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import shap
from pipeline import train_pathogenicity_model, PATHOGENICITY_FEATURES

SEED = 42
OUTPUT_DIR = os.path.join(PROJECT_DIR, "shap_extracted_values")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def zone_stats(shap_col, label=""):
    """Compute stats for a subset of SHAP values."""
    if len(shap_col) == 0:
        return {"n": 0}
    return {
        "n": int(len(shap_col)),
        "min": round(float(np.min(shap_col)), 4),
        "max": round(float(np.max(shap_col)), 4),
        "mean": round(float(np.mean(shap_col)), 4),
        "median": round(float(np.median(shap_col)), 4),
        "std": round(float(np.std(shap_col)), 4),
        "P5": round(float(np.percentile(shap_col, 5)), 4),
        "P25": round(float(np.percentile(shap_col, 25)), 4),
        "P75": round(float(np.percentile(shap_col, 75)), 4),
        "P95": round(float(np.percentile(shap_col, 95)), 4),
    }


def print_zone(name, stats):
    """Pretty-print a zone's statistics."""
    if stats["n"] == 0:
        print(f"    {name}: (empty)")
        return
    print(f"    {name} (n={stats['n']}):")
    print(f"      Range: [{stats['min']}, {stats['max']}]")
    print(f"      Mean={stats['mean']}, Median={stats['median']}, Std={stats['std']}")
    print(f"      P5={stats['P5']}, P25={stats['P25']}, P75={stats['P75']}, P95={stats['P95']}")


def main():
    print("=" * 70)
    print("  SHAP Dependence Zone Analysis — RF Baseline, Test Set")
    print("=" * 70)

    # ── Train model ──
    print("\n▶ Training RF Baseline...")
    rf_model, rf_feats, X_test, y_test, cv_results, meta_test = train_pathogenicity_model(seed=SEED)
    feature_names = list(rf_feats)

    # ── Compute SHAP ──
    print(f"\n▶ Computing SHAP on test set (n={len(X_test)})...")
    explainer = shap.TreeExplainer(rf_model)
    sv_raw = explainer.shap_values(X_test)
    if isinstance(sv_raw, list):
        sv = np.asarray(sv_raw[1])  # class 1 = Non-functional
    elif sv_raw.ndim == 3:
        sv = sv_raw[:, :, 1]
    else:
        sv = np.asarray(sv_raw)

    # Feature indices
    fi = {name: i for i, name in enumerate(feature_names)}

    results = {}

    # ═══════════════════════════════════════════════════════════
    #  REVEL zones: <0.5, 0.5–0.7, >0.7
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("  REVEL — Dependence Zones")
    print("─" * 50)

    revel_raw = X_test[:, fi["REVEL"]]
    revel_shap = sv[:, fi["REVEL"]]
    hotspot_raw = X_test[:, fi["Is_Hotspot"]]

    zones_revel = {
        "below_0.5": revel_raw < 0.5,
        "0.5_to_0.7": (revel_raw >= 0.5) & (revel_raw <= 0.7),
        "above_0.7": revel_raw > 0.7,
    }

    results["REVEL"] = {}
    for zname, mask in zones_revel.items():
        s = zone_stats(revel_shap[mask])
        results["REVEL"][zname] = s
        print_zone(zname, s)

        # Sub-analysis: hotspot vs non-hotspot within each zone
        mask_hot = mask & (hotspot_raw == 1)
        mask_nothot = mask & (hotspot_raw == 0)
        s_hot = zone_stats(revel_shap[mask_hot])
        s_nothot = zone_stats(revel_shap[mask_nothot])
        results["REVEL"][f"{zname}_hotspot"] = s_hot
        results["REVEL"][f"{zname}_non_hotspot"] = s_nothot
        if s_hot["n"] > 0 and s_nothot["n"] > 0:
            print(f"      → Hotspot=1 (n={s_hot['n']}): mean={s_hot['mean']}, range=[{s_hot['min']}, {s_hot['max']}]")
            print(f"      → Hotspot=0 (n={s_nothot['n']}): mean={s_nothot['mean']}, range=[{s_nothot['min']}, {s_nothot['max']}]")
            gap = round(s_hot["mean"] - s_nothot["mean"], 4)
            print(f"      → Hotspot gap (mean diff): {gap}")

    # ═══════════════════════════════════════════════════════════
    #  BAYESDEL zones: <0, 0–0.1, 0.1–0.3, >0.3
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("  BAYESDEL — Dependence Zones")
    print("─" * 50)

    bayesdel_raw = X_test[:, fi["BAYESDEL"]]
    bayesdel_shap = sv[:, fi["BAYESDEL"]]
    revel_raw_for_interaction = X_test[:, fi["REVEL"]]

    zones_bayesdel = {
        "below_0": bayesdel_raw < 0,
        "0_to_0.1": (bayesdel_raw >= 0) & (bayesdel_raw <= 0.1),
        "0.1_to_0.3": (bayesdel_raw > 0.1) & (bayesdel_raw <= 0.3),
        "above_0.3": bayesdel_raw > 0.3,
    }

    results["BAYESDEL"] = {}
    for zname, mask in zones_bayesdel.items():
        s = zone_stats(bayesdel_shap[mask])
        results["BAYESDEL"][zname] = s
        print_zone(zname, s)

        # Sub-analysis: high REVEL vs low REVEL within each zone
        mask_high_revel = mask & (revel_raw_for_interaction > 0.7)
        mask_low_revel = mask & (revel_raw_for_interaction <= 0.7)
        s_hr = zone_stats(bayesdel_shap[mask_high_revel])
        s_lr = zone_stats(bayesdel_shap[mask_low_revel])
        results["BAYESDEL"][f"{zname}_high_revel"] = s_hr
        results["BAYESDEL"][f"{zname}_low_revel"] = s_lr
        if s_hr["n"] > 0 and s_lr["n"] > 0:
            print(f"      → REVEL>0.7 (n={s_hr['n']}): mean={s_hr['mean']}, range=[{s_hr['min']}, {s_hr['max']}]")
            print(f"      → REVEL≤0.7 (n={s_lr['n']}): mean={s_lr['mean']}, range=[{s_lr['min']}, {s_lr['max']}]")
            gap = round(s_hr["mean"] - s_lr["mean"], 4)
            print(f"      → REVEL interaction gap (mean diff): {gap}")

    # ═══════════════════════════════════════════════════════════
    #  Is_Hotspot zones: 0, 1
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("  Is_Hotspot — Dependence Zones")
    print("─" * 50)

    hotspot_shap = sv[:, fi["Is_Hotspot"]]
    bayesdel_for_interaction = X_test[:, fi["BAYESDEL"]]

    zones_hotspot = {
        "non_hotspot_0": hotspot_raw == 0,
        "hotspot_1": hotspot_raw == 1,
    }

    results["Is_Hotspot"] = {}
    for zname, mask in zones_hotspot.items():
        s = zone_stats(hotspot_shap[mask])
        results["Is_Hotspot"][zname] = s
        print_zone(zname, s)

        # Sub-analysis: high vs low BAYESDEL within each hotspot group
        mask_high_bd = mask & (bayesdel_for_interaction > 0.3)
        mask_low_bd = mask & (bayesdel_for_interaction <= 0.3)
        s_hbd = zone_stats(hotspot_shap[mask_high_bd])
        s_lbd = zone_stats(hotspot_shap[mask_low_bd])
        results["Is_Hotspot"][f"{zname}_high_bayesdel"] = s_hbd
        results["Is_Hotspot"][f"{zname}_low_bayesdel"] = s_lbd
        if s_hbd["n"] > 0 and s_lbd["n"] > 0:
            print(f"      → BAYESDEL>0.3 (n={s_hbd['n']}): mean={s_hbd['mean']}, range=[{s_hbd['min']}, {s_hbd['max']}]")
            print(f"      → BAYESDEL≤0.3 (n={s_lbd['n']}): mean={s_lbd['mean']}, range=[{s_lbd['min']}, {s_lbd['max']}]")

    # Cluster center distance
    s_hot = results["Is_Hotspot"]["hotspot_1"]
    s_not = results["Is_Hotspot"]["non_hotspot_0"]
    if s_hot["n"] > 0 and s_not["n"] > 0:
        center_dist = round(abs(s_hot["mean"] - s_not["mean"]), 4)
        print(f"\n    Cluster center distance (|mean_hot - mean_nonhot|): {center_dist}")
        results["Is_Hotspot"]["cluster_center_distance"] = center_dist

    # ── Save ──
    json_path = os.path.join(OUTPUT_DIR, "shap_dependence_zones_rf_baseline_test.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved: {json_path}")

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
