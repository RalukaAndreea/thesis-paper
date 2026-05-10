
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix
)

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

# Color palette
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

STAGE1_COLORS  = ["#2cb67d", "#e53170"]  # Functional=green, Non-functional=pink
def _save_fig(fig, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {os.path.basename(filepath)}")


def plot_class_distribution(results_df: pd.DataFrame, output_dir: str) -> str:
    filepath = os.path.join(output_dir, "class_distribution.png")
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
    ax.set_title("Stage 1 — Pathogenicity", fontweight="bold", pad=15)

    _save_fig(fig, filepath)
    return filepath




#  2. ROC CURVE

def plot_roc_curve(model, X_test: np.ndarray,
                   y_test: np.ndarray, stage_name: str,
                   class_names: list[str], colors: list[str],
                   output_dir: str) -> str:

    stage_tag = stage_name.lower().replace(" ", "").replace("—", "")
    filename = "roc_curve.png"
    if "pathogenicity" in stage_tag or "stage1" in stage_tag:
        filename = "stage1_roc_curve.png"
    filepath = os.path.join(output_dir, filename)

    fig, ax = plt.subplots(figsize=(8, 7))

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    ax.fill_between(fpr, tpr, alpha=0.15, color=colors[1])
    ax.plot(fpr, tpr, color=colors[1], linewidth=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#666666", linewidth=1.2,
            linestyle="--", label="Random Classifier (AUC = 0.500)")

    # Optimal threshold point
    optimal_idx = np.argmax(tpr - fpr)
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx],
               color=COLORS["highlight"], s=120, zorder=5, edgecolors="#1a1a2e",
               linewidth=2, label=f"Optimal Threshold")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(f"ROC Curve — {stage_name}", fontsize=15, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="#1e2a4a",
              edgecolor="#444", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    _save_fig(fig, filepath)
    return filepath

#  3. CONFUSION MATRIX


def plot_confusion_matrix(model, X_test: np.ndarray,
                          y_test: np.ndarray, stage_name: str,
                          class_names: list[str], color: str,
                          output_dir: str) -> str:
    stage_tag = stage_name.lower().replace(" ", "").replace("—", "")
    filename = "confusion_matrix.png"
    if "pathogenicity" in stage_tag or "stage1" in stage_tag:
        filename = "stage1_confusion_matrix.png"

    filepath = os.path.join(output_dir, filename)

    fig, ax = plt.subplots(figsize=(7, 6))

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "custom", ["#16213e", color], N=256
    )

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, aspect="auto")

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
    ax.set_title(f"Confusion Matrix — {stage_name}",
                 fontsize=15, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("Proportion", fontsize=11)

    fig.tight_layout()
    _save_fig(fig, filepath)
    return filepath

#  4. FEATURE IMPORTANCE BAR CHART


def plot_feature_importance(model,
                            feature_names: list[str], stage_name: str,
                            color: str, output_dir: str) -> str:

    stage_tag = stage_name.lower().replace(" ", "").replace("—", "")
    filename = "feature_importance.png"
    if "pathogenicity" in stage_tag or "stage1" in stage_tag:
        filename = "stage1_feature_importance.png"
    filepath = os.path.join(output_dir, filename)

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
    ax.set_xlabel("Importance (Gini)", fontsize=13)
    ax.set_title(f"Feature Importances — {stage_name}",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_xlim(0, max(sorted_vals) * 1.25)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, filepath)
    return filepath

#  Generate all plots


def generate_all_plots(results_df: pd.DataFrame,
                       path_model,
                       path_features: list[str],
                       path_X_test: np.ndarray,
                       path_y_test: np.ndarray,
                       output_dir: str) -> list[str]:

    print("\n  Generating thesis plots...")
    plots = []

    # 1. Class distribution (handles 2 or 3 panels automatically)
    plots.append(plot_class_distribution(results_df, output_dir))

    # 2–4. Stage 1 plots
    plots.append(plot_roc_curve(
        path_model, path_X_test, path_y_test,
        "Stage 1 — Pathogenicity",
        ["Functional", "Non-functional"], STAGE1_COLORS, output_dir
    ))
    plots.append(plot_confusion_matrix(
        path_model, path_X_test, path_y_test,
        "Stage 1 — Pathogenicity",
        ["Functional", "Non-functional"], STAGE1_COLORS[1], output_dir
    ))
    plots.append(plot_feature_importance(
        path_model, path_features,
        "Stage 1 — Pathogenicity", COLORS["primary"], output_dir
    ))

    print(f"\n  ✓ All {len(plots)} plots saved to: {output_dir}")
    return plots
