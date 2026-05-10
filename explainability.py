#!/usr/bin/env python3
"""
SHAP Explainability & Case Study Extraction for TP53 Pipeline.

Extracts specific case studies from the 20% hold-out test set and generates
SHAP explanations (global + individual) for thesis defense demonstration.

Case studies per model stage:
  1. True Positive   — clear-cut pathogenic/somatic, highest confidence
  2. True Negative   — clear-cut benign/germline, highest confidence
  3. Edge Case       — borderline prediction, confidence closest to 50%
  4. Error           — misclassification (FP or FN), most confident mistake
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from pipeline import (
    train_pathogenicity_model,
    PATHOGENICITY_FEATURES,
)

CASE_STUDIES_DIR = os.path.join(PROJECT_DIR, "case_studies")


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ───────────────────────────────────────────────────────────────
#  CASE STUDY EXTRACTION
# ───────────────────────────────────────────────────────────────

def extract_cases(model, X_test, y_test, feature_names, class_names):
    """
    Extract 4 representative case studies from the hold-out test set.

    Returns a dict with keys: true_positive, true_negative, edge_case, error.
    Each value contains the sample index, feature values, predictions, and
    narrative description.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    confidence = y_proba.max(axis=1)
    prob_positive = y_proba[:, 1]

    cases = {}

    # 1. True Positive: correct class-1, highest confidence
    tp_mask = (y_pred == 1) & (y_test == 1)
    if tp_mask.any():
        tp_indices = np.where(tp_mask)[0]
        best = tp_indices[np.argmax(confidence[tp_indices])]
        cases["true_positive"] = {
            "index": int(best),
            "label": f"Clear-Cut {class_names[1]}",
            "description": (
                f"The model correctly identified this variant as "
                f"{class_names[1]} with very high confidence. "
                f"All major features align with the positive class."
            ),
        }

    # 2. True Negative: correct class-0, highest confidence
    tn_mask = (y_pred == 0) & (y_test == 0)
    if tn_mask.any():
        tn_indices = np.where(tn_mask)[0]
        best = tn_indices[np.argmax(confidence[tn_indices])]
        cases["true_negative"] = {
            "index": int(best),
            "label": f"Clear-Cut {class_names[0]}",
            "description": (
                f"The model correctly identified this variant as "
                f"{class_names[0]} with very high confidence. "
                f"Feature values are consistent with the negative class."
            ),
        }

    # 3. Edge Case: probability closest to 0.50
    edge_dist = np.abs(prob_positive - 0.5)
    edge_idx = int(np.argmin(edge_dist))
    correct = y_pred[edge_idx] == y_test[edge_idx]
    cases["edge_case"] = {
        "index": edge_idx,
        "label": "Edge Case (Borderline)",
        "description": (
            f"This variant sits on the decision boundary — the model's "
            f"probability is near 50%, making it the least confident "
            f"prediction. It was {'correctly' if correct else 'incorrectly'} "
            f"classified."
        ),
    }

    # 4. Error: misclassification with highest confidence
    error_mask = y_pred != y_test
    if error_mask.any():
        err_indices = np.where(error_mask)[0]
        worst = err_indices[np.argmax(confidence[err_indices])]
        is_fp = y_pred[worst] == 1 and y_test[worst] == 0
        err_type = "False Positive" if is_fp else "False Negative"
        cases["error"] = {
            "index": int(worst),
            "label": f"Misclassification ({err_type})",
            "description": (
                f"The model predicted {class_names[int(y_pred[worst])]} "
                f"but the true label is {class_names[int(y_test[worst])]}. "
                f"This {err_type.lower()} reveals which features misled "
                f"the model."
            ),
        }

    # Enrich every case with feature values and prediction details
    for case_info in cases.values():
        idx = case_info["index"]
        case_info["features"] = {
            fname: round(float(X_test[idx, fi]), 4)
            for fi, fname in enumerate(feature_names)
        }
        case_info["true_label"] = class_names[int(y_test[idx])]
        case_info["predicted_label"] = class_names[int(y_pred[idx])]
        case_info["confidence"] = round(float(confidence[idx]), 4)
        case_info["prob_class_0"] = round(float(y_proba[idx, 0]), 4)
        case_info["prob_class_1"] = round(float(y_proba[idx, 1]), 4)

    return cases


# ───────────────────────────────────────────────────────────────
#  SHAP COMPUTATION
# ───────────────────────────────────────────────────────────────

def _compute_shap(model, X_test, feature_names):
    """
    Compute SHAP values for the positive class (class 1) using TreeExplainer.
    Returns (shap_values_2d_array, base_value_float) for class 1.
    """
    explainer = shap.TreeExplainer(model)
    sv_raw = explainer.shap_values(X_test)

    # For binary classification RF, shap_values() returns a list [class0, class1]
    if isinstance(sv_raw, list):
        sv = np.asarray(sv_raw[1])
        base = float(explainer.expected_value[1])
    elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
        # (n_samples, n_features, n_classes) → take class 1
        sv = sv_raw[:, :, 1]
        base = float(explainer.expected_value[1])
    else:
        sv = np.asarray(sv_raw)
        ev = explainer.expected_value
        base = float(ev[1]) if hasattr(ev, "__len__") else float(ev)

    # Ensure 2D: (n_samples, n_features)
    assert sv.ndim == 2, f"Expected 2D SHAP values, got shape {sv.shape}"
    return sv, base


# ───────────────────────────────────────────────────────────────
#  GLOBAL SHAP PLOTS
# ───────────────────────────────────────────────────────────────

def generate_shap_global(model, X_test, feature_names, output_dir, stage_name):
    """Generate beeswarm + bar plots for the entire test set."""
    _ensure_dir(output_dir)

    sv, base = _compute_shap(model, X_test, feature_names)
    X_df = pd.DataFrame(X_test, columns=feature_names)

    explanation = shap.Explanation(
        values=sv,
        base_values=np.full(sv.shape[0], base),
        data=X_df.values,
        feature_names=list(feature_names),
    )

    # 1. Beeswarm summary
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(explanation, show=False, max_display=len(feature_names))
    plt.title(f"SHAP Summary — {stage_name}", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "global_shap_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  ✓ Saved: global_shap_summary.png")

    # 2. Bar (mean |SHAP|)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(explanation, show=False, max_display=len(feature_names))
    plt.title(
        f"Mean |SHAP| Feature Importance — {stage_name}",
        fontsize=14, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "global_shap_bar.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  ✓ Saved: global_shap_bar.png")

    return sv, base, summary_path, bar_path


# ───────────────────────────────────────────────────────────────
#  PER-CASE SHAP WATERFALL
# ───────────────────────────────────────────────────────────────

def generate_shap_case(sv, base, X_test, feature_names, case_info, output_dir):
    """Generate a SHAP waterfall plot for a single case study."""
    _ensure_dir(output_dir)
    idx = case_info["index"]

    single_exp = shap.Explanation(
        values=sv[idx],
        base_values=base,
        data=X_test[idx],
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(single_exp, show=False, max_display=len(feature_names))

    pred = case_info["predicted_label"]
    true = case_info["true_label"]
    conf = case_info["confidence"]
    plt.title(
        f"{case_info['label']}\n"
        f"Predicted: {pred} ({conf:.1%})  |  True: {true}",
        fontsize=12, fontweight="bold", pad=15,
    )
    plt.tight_layout()

    path = os.path.join(output_dir, "shap_waterfall.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"    ✓ Saved: {os.path.basename(output_dir)}/shap_waterfall.png")
    return path


# ───────────────────────────────────────────────────────────────
#  STAGE RUNNER
# ───────────────────────────────────────────────────────────────

def run_stage(stage_key, stage_name, class_names,
              model, X_test, y_test, feature_names):
    """Full explainability pipeline for one classification stage."""
    stage_dir = os.path.join(CASE_STUDIES_DIR, stage_key)
    _ensure_dir(stage_dir)

    print(f"\n{'=' * 60}")
    print(f"  {stage_name}")
    print(f"{'=' * 60}")

    # 1. Extract cases
    print(f"\n  ── Extracting Case Studies ──")
    cases = extract_cases(model, X_test, y_test, feature_names, class_names)
    for key, info in cases.items():
        print(f"    {key:16s}  →  {info['predicted_label']:16s} "
              f"(conf {info['confidence']:.1%})  true={info['true_label']}")

    # 2. Global SHAP
    print(f"\n  ── Global SHAP (entire test set, n={len(y_test)}) ──")
    sv, base, summary_path, bar_path = generate_shap_global(
        model, X_test, feature_names, stage_dir, stage_name,
    )

    # 3. Per-case SHAP waterfall
    print(f"\n  ── Per-Case SHAP Waterfalls ──")
    for case_key, case_info in cases.items():
        case_dir = os.path.join(stage_dir, f"case_{case_key}")
        wf_path = generate_shap_case(
            sv, base, X_test, feature_names, case_info, case_dir,
        )
        case_info["shap_waterfall"] = wf_path

        # Save case JSON (drop internal index)
        save = {k: v for k, v in case_info.items() if k != "index"}
        with open(os.path.join(case_dir, "info.json"), "w") as f:
            json.dump(save, f, indent=2)

    # 4. Stage summary JSON
    summary = {
        "stage_name": stage_name,
        "class_names": class_names,
        "test_set_size": int(len(y_test)),
        "class_distribution": {
            class_names[0]: int((y_test == 0).sum()),
            class_names[1]: int((y_test == 1).sum()),
        },
        "global_shap_summary": summary_path,
        "global_shap_bar": bar_path,
        "cases": {
            k: {kk: vv for kk, vv in v.items() if kk != "index"}
            for k, v in cases.items()
        },
    }
    with open(os.path.join(stage_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  ✓ Stage complete → {stage_dir}")
    return cases


# ───────────────────────────────────────────────────────────────
#  MAIN
# ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TP53 Explainability & Case Study Extraction")
    print("  SHAP Analysis on 20% Hold-Out Test Sets")
    print("=" * 60)

    _ensure_dir(CASE_STUDIES_DIR)

    # Pathogenicity
    print("\n▶ Training Pathogenicity Model...")
    path_model, path_feats, path_Xt, path_yt, _ = train_pathogenicity_model()
    run_stage(
        "stage1_pathogenicity", "Pathogenicity",
        ["Functional", "Non-functional"],
        path_model, path_Xt, path_yt, path_feats,
    )

    print("\n" + "=" * 60)
    print("  ✓ Case studies & SHAP explanations generated!")
    print(f"    → {CASE_STUDIES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
