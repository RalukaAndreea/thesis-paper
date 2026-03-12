
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier

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

STAGE1_COLORS = ["#2cb67d", "#e53170"]  # Functional=green, Non-functional=pink
STAGE2_COLORS = ["#72b4eb", "#f9c74f"]  # Germline=blue, Somatic=gold


def _save_fig(fig, filepath: str):
    fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {os.path.basename(filepath)}")


def plot_class_distribution(results_df: pd.DataFrame, output_dir: str) -> str:

    filepath = os.path.join(output_dir, "class_distribution.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    path_counts = results_df["Pathogenicity_Prediction"].value_counts()
    labels1 = path_counts.index.tolist()
    sizes1 = path_counts.values
    pct1 = sizes1 / sizes1.sum() * 100

    wedges1, texts1, autotexts1 = ax1.pie(
        sizes1, labels=None, colors=STAGE1_COLORS,
        autopct=lambda p: f"{p:.1f}%", startangle=90,
        pctdistance=0.75, wedgeprops=dict(width=0.45, edgecolor="#1a1a2e", linewidth=2)
    )
    for at in autotexts1:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax1.legend(
        [f"{l} (n={s})" for l, s in zip(labels1, sizes1)],
        loc="lower center", frameon=False, fontsize=10,
        bbox_to_anchor=(0.5, -0.08)
    )
    ax1.set_title("Stage 1 — Pathogenicity", fontweight="bold", pad=15)

    orig_counts = results_df["Origin_Prediction"].value_counts()
    labels2 = orig_counts.index.tolist()
    sizes2 = orig_counts.values

    wedges2, texts2, autotexts2 = ax2.pie(
        sizes2, labels=None, colors=STAGE2_COLORS,
        autopct=lambda p: f"{p:.1f}%", startangle=90,
        pctdistance=0.75, wedgeprops=dict(width=0.45, edgecolor="#1a1a2e", linewidth=2)
    )
    for at in autotexts2:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax2.legend(
        [f"{l} (n={s})" for l, s in zip(labels2, sizes2)],
        loc="lower center", frameon=False, fontsize=10,
        bbox_to_anchor=(0.5, -0.08)
    )
    ax2.set_title("Stage 2 — Origin", fontweight="bold", pad=15)

    fig.suptitle("Class Distribution of Patient Variant Predictions",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, filepath)
    return filepath


#  2. ROC CURVE

def plot_roc_curve(model, X_test: np.ndarray,
                   y_test: np.ndarray, stage_name: str,
                   class_names: list[str], colors: list[str],
                   output_dir: str) -> str:

    stage_tag = stage_name.lower().replace(" ", "").replace("—", "")
    if "pathogenicity" in stage_tag or "1" in stage_tag:
        filename = "stage1_roc_curve.png"
    else:
        filename = "stage2_roc_curve.png"
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
    if "1" in stage_name:
        filename = "stage1_confusion_matrix.png"
    else:
        filename = "stage2_confusion_matrix.png"
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

    if "1" in stage_name:
        filename = "stage1_feature_importance.png"
    else:
        filename = "stage2_feature_importance.png"
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


#  5. MODEL COMPARISON

def plot_model_comparison(path_model,
                          path_X_test: np.ndarray, path_y_test: np.ndarray,
                          orig_model,
                          orig_X_test: np.ndarray, orig_y_test: np.ndarray,
                          output_dir: str) -> str:

    filepath = os.path.join(output_dir, "model_comparison.png")

    metrics = {}
    for name, model, X, y in [
        ("Stage 1\nPathogenicity", path_model, path_X_test, path_y_test),
        ("Stage 2\nOrigin",        orig_model, orig_X_test, orig_y_test),
    ]:
        y_pred = model.predict(X)
        metrics[name] = {
            "Accuracy":  accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall":    recall_score(y, y_pred, zero_division=0),
            "F1 Score":  f1_score(y, y_pred, zero_division=0),
        }

    fig, ax = plt.subplots(figsize=(11, 6))

    metric_names = list(list(metrics.values())[0].keys())
    model_names = list(metrics.keys())
    x = np.arange(len(metric_names))
    width = 0.32

    colors_bar = [COLORS["primary"], COLORS["secondary"]]

    for i, (model_name, color) in enumerate(zip(model_names, colors_bar)):
        vals = [metrics[model_name][m] for m in metric_names]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model_name,
                      color=color, edgecolor="#1a1a2e", linewidth=0.5,
                      alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Performance Comparison",
                 fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="upper right", frameon=True, facecolor="#1e2a4a",
              edgecolor="#444", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    ax.axhline(y=0.5, color="#666", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(len(metric_names) - 0.5, 0.51, "Random baseline",
            fontsize=9, color="#888", ha="right")

    fig.tight_layout()
    _save_fig(fig, filepath)
    return filepath

#  Generate all plots


def generate_all_plots(results_df: pd.DataFrame,
                       path_model,
                       path_features: list[str],
                       path_X_test: np.ndarray,
                       path_y_test: np.ndarray,
                       orig_model,
                       orig_features: list[str],
                       orig_X_test: np.ndarray,
                       orig_y_test: np.ndarray,
                       output_dir: str) -> list[str]:

    print("\n  Generating thesis plots...")
    plots = []

    # 1. Class distribution
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

    # 5–7. Stage 2 plots
    plots.append(plot_roc_curve(
        orig_model, orig_X_test, orig_y_test,
        "Stage 2 — Origin",
        ["Germline", "Somatic"], STAGE2_COLORS, output_dir
    ))
    plots.append(plot_confusion_matrix(
        orig_model, orig_X_test, orig_y_test,
        "Stage 2 — Origin",
        ["Germline", "Somatic"], STAGE2_COLORS[1], output_dir
    ))
    plots.append(plot_feature_importance(
        orig_model, orig_features,
        "Stage 2 — Origin", COLORS["secondary"], output_dir
    ))

    # 8. Model comparison
    plots.append(plot_model_comparison(
        path_model, path_X_test, path_y_test,
        orig_model, orig_X_test, orig_y_test,
        output_dir
    ))

    print(f"\n  ✓ All {len(plots)} plots saved to: {output_dir}")
    return plots
