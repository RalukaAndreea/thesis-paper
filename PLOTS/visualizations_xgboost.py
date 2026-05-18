# PLOTS/visualizations_xgboost.py — XGBoost thesis figures
# ──────────────────────────────────────────────────────────
# Self-contained plot module for the XGBoost (Optuna) model.
# Uses the exact same dark theme, palette, fonts, DPI, and
# figure dimensions as PLOTS/visualizations.py so the RF
# and XGBoost figures can sit side-by-side in the thesis.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ── Dark theme (verbatim copy from visualizations.py) ─────────
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

# ── Color palette (verbatim copy from visualizations.py) ──────
COLORS = {
    "primary":    "#7f5af0",
    "secondary":  "#2cb67d",
    "accent":     "#e53170",
    "highlight":  "#f9c74f",
    "info":       "#72b4eb",
    "bg_card":    "#1e2a4a",
    "gradient_1": "#7f5af0",
    "gradient_2": "#2cb67d",
}

STAGE1_COLORS = ["#2cb67d", "#e53170"]   # Functional=green, Non-functional=pink


def _save_fig(fig, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {os.path.basename(filepath)}")


# ──────────────────────────────────────────────────────────
#  0. Class Distribution (per-upload pie chart)
# ──────────────────────────────────────────────────────────

def plot_class_distribution_xgb(results_df: pd.DataFrame, output_dir: str) -> str:
    """Donut chart of predicted pathogenicity classes (XGBoost upload)."""
    filepath = os.path.join(output_dir, "xgb_class_distribution.png")
    counts = results_df["Pathogenicity_Prediction"].dropna().astype(str).str.strip()
    counts = counts[(counts != "") & (counts != "N/A")].value_counts()

    labels = counts.index.tolist()
    sizes = counts.values
    colors = list(STAGE1_COLORS[:len(labels)])
    if len(colors) < len(labels):
        colors.extend(plt.cm.Set2(np.linspace(0, 1, len(labels) - len(colors))))

    fig, ax = plt.subplots(figsize=(7, 6))
    _, _, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="#1a1a2e", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax.legend(
        [f"{label} (n={size})" for label, size in zip(labels, sizes)],
        loc="lower center",
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.08),
    )
    ax.set_title("Stage 1 — Pathogenicity (XGBoost)", fontweight="bold", pad=15)

    _save_fig(fig, filepath)
    return filepath


# ──────────────────────────────────────────────────────────
#  1. ROC Curve
# ──────────────────────────────────────────────────────────

def plot_roc_curve_xgb(model, X_test: np.ndarray,
                       y_test: np.ndarray, output_dir: str,
                       optimal_threshold: float) -> str:
    """ROC curve for the XGBoost pathogenicity model."""
    filepath = os.path.join(output_dir, "xgb_roc_curve.png")

    fig, ax = plt.subplots(figsize=(8, 7))

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    ax.fill_between(fpr, tpr, alpha=0.15, color=STAGE1_COLORS[1])
    ax.plot(fpr, tpr, color=STAGE1_COLORS[1], linewidth=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#666666", linewidth=1.2,
            linestyle="--", label="Random Classifier (AUC = 0.500)")

    # Mark point closest to the CV-derived optimal threshold
    thresh_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    ax.scatter(fpr[thresh_idx], tpr[thresh_idx],
               color=COLORS["highlight"], s=120, zorder=5,
               edgecolors="#1a1a2e", linewidth=2,
               label=f"CV Optimal Threshold = {optimal_threshold:.3f}")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve — XGBoost (Optuna) Pathogenicity",
                 fontsize=15, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="#1e2a4a",
              edgecolor="#444", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    _save_fig(fig, filepath)
    return filepath


# ──────────────────────────────────────────────────────────
#  2. Confusion Matrix
# ──────────────────────────────────────────────────────────

def plot_confusion_matrix_xgb(model, X_test: np.ndarray,
                              y_test: np.ndarray, output_dir: str,
                              optimal_threshold: float) -> str:
    """Confusion matrix for the XGBoost pathogenicity model."""
    filepath = os.path.join(output_dir, "xgb_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(7, 6))

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    cmap = LinearSegmentedColormap.from_list(
        "custom", ["#16213e", STAGE1_COLORS[1]], N=256
    )

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, aspect="auto")

    class_names = ["Functional", "Non-functional"]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_color = "#e0e0e0" if cm_norm[i, j] > 0.5 else "#e0e0e0"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color=text_color)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=10)
    ax.set_ylabel("True Label", fontsize=13, labelpad=10)

    ax.set_title(f"Confusion Matrix — XGBoost (Optuna)\n"
                 f"(threshold = {optimal_threshold:.3f})",
                 fontsize=15, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("Proportion", fontsize=11)

    fig.tight_layout()
    _save_fig(fig, filepath)
    return filepath


# ──────────────────────────────────────────────────────────
#  3. Feature Importance
# ──────────────────────────────────────────────────────────

def plot_feature_importance_xgb(model,
                                feature_names: list[str],
                                output_dir: str) -> str:
    """Horizontal bar chart of XGBoost feature importances (gain)."""
    filepath = os.path.join(output_dir, "xgb_feature_importance.png")

    fig, ax = plt.subplots(figsize=(9, 5))

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
    ax.set_xlabel("Importance (gain)", fontsize=13)
    ax.set_title("Feature Importances — XGBoost (Optuna) Pathogenicity",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_xlim(0, max(sorted_vals) * 1.25)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, filepath)
    return filepath


# ──────────────────────────────────────────────────────────
#  4. Feature Correlation Heatmap
# ──────────────────────────────────────────────────────────

def plot_feature_correlation_xgb(X_test: np.ndarray,
                                 feature_names: list[str],
                                 output_dir: str) -> str:
    """Generate a Pearson correlation heatmap for the model features."""
    filepath = os.path.join(output_dir, "xgb_feature_correlation_heatmap.png")

    df = pd.DataFrame(X_test, columns=feature_names)
    corr = df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(9, 8))

    cmap = LinearSegmentedColormap.from_list(
        "custom_diverging", ["#2cb67d", "#16213e", "#e53170"], N=256
    )

    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    n = len(feature_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(feature_names, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(feature_names, fontsize=9)

    # Annotate each cell with the correlation value
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            color = "#e0e0e0" if abs(val) > 0.4 else "#a0a0a0"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    ax.set_title("Feature Correlation Heatmap — XGBoost (Pearson)",
                 fontsize=15, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("Correlation", fontsize=11)

    fig.tight_layout()
    _save_fig(fig, filepath)
    return filepath


# ──────────────────────────────────────────────────────────
#  Entry point — generate all XGBoost plots
# ──────────────────────────────────────────────────────────

def generate_all_xgb_plots(path_model,
                           path_features: list[str],
                           path_X_test: np.ndarray,
                           path_y_test: np.ndarray,
                           output_dir: str,
                           optimal_threshold: float) -> list[str]:
    """Generate all XGBoost thesis figures.

    Parameters mirror ``generate_all_plots`` in visualizations.py
    so the runner script reads identically.
    """
    print("\n  Generating XGBoost thesis plots...")
    plots = []

    plots.append(plot_roc_curve_xgb(
        path_model, path_X_test, path_y_test,
        output_dir, optimal_threshold,
    ))
    plots.append(plot_confusion_matrix_xgb(
        path_model, path_X_test, path_y_test,
        output_dir, optimal_threshold,
    ))
    plots.append(plot_feature_importance_xgb(
        path_model, path_features, output_dir,
    ))
    plots.append(plot_feature_correlation_xgb(
        path_X_test, path_features, output_dir,
    ))

    print(f"\n  ✓ All {len(plots)} XGBoost plots saved to: {output_dir}")
    return plots

