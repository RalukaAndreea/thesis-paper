
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



def plot_feature_importance(model,
                            feature_names: list[str], stage_name: str,
                            color: str, output_dir: str) -> str:

    stage_tag = stage_name.lower().replace(" ", "").replace("—", "")
    filename = "feature_importance_revel.png"
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





def generate_all_plots(results_df: pd.DataFrame,
                       path_model,
                       path_features: list[str],
                       path_X_test: np.ndarray,
                       path_y_test: np.ndarray,
                       output_dir: str,
                       optimal_threshold: float = None) -> list[str]:

    print("\n  Generating thesis plots...")
    plots = []




    plots.append(plot_feature_importance(
        path_model, path_features,
        "Stage 1 — Pathogenicity", COLORS["primary"], output_dir
    ))



    print(f"\n  ✓ All {len(plots)} plots saved to: {output_dir}")
    return plots
