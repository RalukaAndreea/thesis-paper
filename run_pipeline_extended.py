#!/usr/bin/env python3
"""
run_pipeline_extended.py
────────────────────────
Runner script for the extended 12-feature TP53 pathogenicity model.
Mirrors ``run_pipeline.py`` but trains the extended model and generates
plots with the 'extended_' filename prefix so that baseline outputs are
never overwritten.
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Ensure the project directory is on the Python path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from pipeline_extended import (
    train_extended_pathogenicity_model,
    extract_feature_importances,
    EXTENDED_PATHOGENICITY_FEATURES,
)

# ──────────────────────────────────────────────────────────────────────
#  Dark-theme style (matches PLOTS/visualizations.py exactly)
# ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#e0e0e0",
    "axes.labelcolor":  "#e0e0e0",
    "text.color":       "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "grid.color":       "#2a2a4a",
    "grid.alpha":       0.4,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.labelsize":   12,
})

COLORS = {
    "primary":    "#7f5af0",
    "secondary":  "#2cb67d",
    "accent":     "#e53170",
    "highlight":  "#f9c74f",
    "info":       "#72b4eb",
}
STAGE1_COLORS = ["#2cb67d", "#e53170"]  # Functional=green, Non-functional=pink


# ──────────────────────────────────────────────────────────────────────
#  Plot helpers
# ──────────────────────────────────────────────────────────────────────

def _save_fig(fig, filepath: str) -> str:
    """Save a figure and close it. Returns the absolute filepath."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {os.path.basename(filepath)}")
    return filepath


def _plot_roc_curve(model, X_test, y_test, optimal_threshold, output_dir) -> str:
    """ROC curve with AUC and CV-optimal threshold marker."""
    filepath = os.path.join(output_dir, "extended_roc_curve.png")

    fig, ax = plt.subplots(figsize=(8, 7))

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    ax.fill_between(fpr, tpr, alpha=0.15, color=STAGE1_COLORS[1])
    ax.plot(fpr, tpr, color=STAGE1_COLORS[1], linewidth=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#666666", linewidth=1.2,
            linestyle="--", label="Random Classifier (AUC = 0.500)")

    # Optimal threshold point (from CV, not computed on test set)
    if optimal_threshold is not None:
        thresh_idx = np.argmin(np.abs(thresholds - optimal_threshold))
        ax.scatter(fpr[thresh_idx], tpr[thresh_idx],
                   color=COLORS["highlight"], s=120, zorder=5,
                   edgecolors="#1a1a2e", linewidth=2,
                   label=f"CV Optimal Threshold = {optimal_threshold:.3f}")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve — Extended Pathogenicity Model",
                 fontsize=15, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="#1e2a4a",
              edgecolor="#444", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    return _save_fig(fig, filepath)


def _plot_confusion_matrix(model, X_test, y_test, optimal_threshold,
                           output_dir) -> str:
    """Confusion matrix with counts and percentages."""
    filepath = os.path.join(output_dir, "extended_confusion_matrix.png")
    class_names = ["Functional", "Non-functional"]

    fig, ax = plt.subplots(figsize=(7, 6))

    if optimal_threshold is not None:
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= optimal_threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    cmap = LinearSegmentedColormap.from_list(
        "custom", ["#16213e", STAGE1_COLORS[1]], N=256
    )
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, aspect="auto")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color="#e0e0e0")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=10)
    ax.set_ylabel("True Label", fontsize=13, labelpad=10)

    title = "Confusion Matrix — Extended Pathogenicity Model"
    if optimal_threshold is not None:
        title += f"\n(threshold = {optimal_threshold:.3f})"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("Proportion", fontsize=11)

    fig.tight_layout()
    return _save_fig(fig, filepath)


def _plot_feature_importance(model, feature_names, output_dir) -> str:
    """Horizontal bar chart of Gini importances."""
    filepath = os.path.join(output_dir, "extended_feature_importance.png")

    fig, ax = plt.subplots(figsize=(9, 6))

    importances = model.feature_importances_
    indices = np.argsort(importances)

    sorted_names = [feature_names[i] for i in indices]
    sorted_vals = importances[indices]

    n = len(sorted_vals)
    bar_colors = [plt.cm.get_cmap("cool")(i / max(n - 1, 1)) for i in range(n)]

    bars = ax.barh(range(n), sorted_vals, color=bar_colors,
                   edgecolor="#1a1a2e", linewidth=0.5, height=0.65)

    for bar, val in zip(bars, sorted_vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel("Importance (Gini)", fontsize=13)
    ax.set_title("Feature Importances — Extended Pathogenicity Model",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_xlim(0, max(sorted_vals) * 1.25)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    return _save_fig(fig, filepath)


def _plot_feature_correlation(X_test, feature_names, output_dir) -> str:
    """Pearson correlation heatmap for the 12 model features."""
    filepath = os.path.join(output_dir, "extended_feature_correlation_heatmap.png")

    df = pd.DataFrame(X_test, columns=feature_names)
    corr = df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 9))

    cmap = LinearSegmentedColormap.from_list(
        "custom_diverging", ["#2cb67d", "#16213e", "#e53170"], N=256
    )
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    n = len(feature_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(feature_names, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(feature_names, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            color = "#e0e0e0" if abs(val) > 0.4 else "#a0a0a0"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)

    ax.set_title("Feature Correlation Heatmap (Pearson) — Extended Model",
                 fontsize=15, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("Correlation", fontsize=11)

    fig.tight_layout()
    return _save_fig(fig, filepath)


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  TP53 Extended Pathogenicity Pipeline (12 features)")
    print("=" * 65)

    # ── Step 1: Train extended model ──────────────────────────────
    print("\n▶ Step 1: Training extended pathogenicity classifier...")
    (path_model, path_features,
     path_X_test, path_y_test,
     cv_results, meta_test) = train_extended_pathogenicity_model()

    # ── Step 2: Save CV results ───────────────────────────────────
    cv_json = os.path.join(PROJECT_DIR, "cv_model_comparison_extended.json")
    cv_serializable = {}
    for k, v in cv_results.items():
        if isinstance(v, tuple):
            cv_serializable[k] = {"mean": round(v[0], 4), "std": round(v[1], 4)}
        else:
            cv_serializable[k] = v
    with open(cv_json, "w") as f:
        json.dump(cv_serializable, f, indent=2, default=str)
    print(f"\n  ✓ CV comparison saved to: {cv_json}")

    # ── Step 3: Feature importances ───────────────────────────────
    print("\n▶ Step 2: Extracting feature importances...")
    fi_df = extract_feature_importances(
        path_model, path_features,
        stage_name="Extended Pathogenicity (12 features)",
    )
    fi_csv = os.path.join(PROJECT_DIR, "feature_importances_extended.csv")
    fi_df.to_csv(fi_csv, index=False)
    print(f"  ✓ Feature importances saved to: {fi_csv}")

    # ── Step 4: Generate plots ────────────────────────────────────
    optimal_threshold = cv_results.get("optimal_threshold", 0.5)
    print(f"\n▶ Step 3: Generating thesis visualizations (threshold={optimal_threshold:.4f})...")

    plots = []
    plots.append(_plot_roc_curve(
        path_model, path_X_test, path_y_test,
        optimal_threshold, PROJECT_DIR,
    ))
    plots.append(_plot_confusion_matrix(
        path_model, path_X_test, path_y_test,
        optimal_threshold, PROJECT_DIR,
    ))
    plots.append(_plot_feature_importance(
        path_model, path_features, PROJECT_DIR,
    ))
    plots.append(_plot_feature_correlation(
        path_X_test, path_features, PROJECT_DIR,
    ))

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Extended Pipeline complete!")
    print("=" * 65)

    rf_f1  = cv_results["rf_f1"]
    rf_auc = cv_results["rf_auc"]
    rf_acc = cv_results["rf_accuracy"]
    rf_sens = cv_results["rf_sensitivity"]
    rf_spec = cv_results["rf_specificity"]

    print(f"\n  ── Extended Model — 5-Fold CV ──")
    print(f"    Accuracy:     {rf_acc[0]:.4f} ± {rf_acc[1]:.4f}")
    print(f"    F1 Score:     {rf_f1[0]:.4f} ± {rf_f1[1]:.4f}")
    print(f"    AUC:          {rf_auc[0]:.4f} ± {rf_auc[1]:.4f}")
    print(f"    Sensitivity:  {rf_sens[0]:.4f} ± {rf_sens[1]:.4f}")
    print(f"    Specificity:  {rf_spec[0]:.4f} ± {rf_spec[1]:.4f}")
    print(f"    Threshold:    {optimal_threshold:.4f}")

    print(f"\n  ── Hold-out Test ──")
    print(f"    Sensitivity:  {cv_results['test_sensitivity']:.4f}")
    print(f"    Specificity:  {cv_results['test_specificity']:.4f}")

    print(f"\n  Output files:")
    print(f"    • cv_model_comparison_extended.json")
    print(f"    • feature_importances_extended.csv")

    print(f"\n  Thesis plots ({len(plots)}):")
    for p in plots:
        print(f"    • {os.path.basename(p)}")
    print()


if __name__ == "__main__":
    main()
