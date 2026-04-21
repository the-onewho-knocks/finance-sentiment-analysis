import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

LABEL_NAMES  = ["negative", "neutral", "positive"]
LABEL_INT    = [0, 1, 2]
PLOTS_DIR    = MODELS_DIR.parent.parent / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Core Metrics ──────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> dict:
    """Compute all core classification metrics for one model."""
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(
            y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(
            y_true, y_pred, average="weighted", zero_division=0), 4),
        "f1_score":  round(f1_score(
            y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def print_model_report(model_name: str, y_true, y_pred) -> None:
    """Pretty-print full classification report for one model."""
    print(f"\n── {model_name.upper().replace('_',' ')} ──")
    print(classification_report(
        y_true, y_pred,
        target_names  = LABEL_NAMES,
        zero_division = 0,
    ))


# ── Comparison Table ──────────────────────────────────────────────────────────

def build_comparison_table(results: dict) -> pd.DataFrame:
    """Build a clean DataFrame comparing all models side by side."""
    rows = []
    for name, result in results.items():
        report = result["report"]
        rows.append({
            "Model":         name.replace("_", " ").title(),
            "Accuracy":      round(result["accuracy"], 4),
            "Precision":     round(report["weighted avg"]["precision"], 4),
            "Recall":        round(report["weighted avg"]["recall"], 4),
            "F1-Score":      round(report["weighted avg"]["f1-score"], 4),
            "Training Time": f"{result['training_time']}s",
        })
    return pd.DataFrame(rows).set_index("Model")


def print_comparison_table(results: dict) -> pd.DataFrame:
    """Print and return the full model comparison table."""
    df = build_comparison_table(results)
    print("\n" + "="*65)
    print("📊  MODEL COMPARISON TABLE")
    print("="*65)
    print(df.to_string())
    print("="*65)
    best = df["F1-Score"].idxmax()
    print(f"\n🏆  Best model by F1-Score: {best}")
    return df


def build_tuned_comparison_table(
    before_results: dict,
    after_results:  dict,
) -> pd.DataFrame:
    """Build DataFrame comparing before/after tuning for all models."""
    rows = []
    for name in before_results:
        b = before_results[name]["report"]["weighted avg"]
        a = after_results[name]["report"]["weighted avg"]
        rows.append({
            "Model":           name.replace("_", " ").title(),
            "Before Accuracy": round(before_results[name]["accuracy"], 4),
            "After Accuracy":  round(after_results[name]["accuracy"],  4),
            "Before F1":       round(b["f1-score"], 4),
            "After F1":        round(a["f1-score"], 4),
            "Improvement":     round(a["f1-score"] - b["f1-score"], 4),
        })
    return pd.DataFrame(rows).set_index("Model")


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    save: bool = True,
) -> None:
    """
    Plot a heatmap confusion matrix for one model.

    Rows   = actual labels
    Columns = predicted labels
    Diagonal = correct predictions (want these HIGH)
    Off-diagonal = errors (want these LOW)
    """
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_INT)

    # Normalize to percentages for easier reading
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Confusion Matrix — {model_name.replace('_', ' ').title()}",
        fontsize=14, fontweight="bold"
    )

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        ax=axes[0], linewidths=0.5,
    )
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Percentages
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Greens",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        ax=axes[1], linewidths=0.5,
    )
    axes[1].set_title("Row-Normalized (%)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()

    if save:
        path = PLOTS_DIR / f"confusion_matrix_{model_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved confusion matrix → {path}")

    plt.show()
    plt.close()


def plot_all_confusion_matrices(results: dict, y_test) -> None:
    """Plot confusion matrices for all models in one grid."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Confusion Matrices — All Models", fontsize=15, fontweight="bold")

    for ax, (name, result) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, result["y_pred"], labels=LABEL_INT)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            ax=ax, linewidths=0.5, cbar=False,
        )
        ax.set_title(name.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    path = PLOTS_DIR / "confusion_matrices_all.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved all confusion matrices → {path}")
    plt.show()
    plt.close()


# ── ROC-AUC Curves ────────────────────────────────────────────────────────────

def plot_roc_curves(
    results:  dict,
    X_test,
    y_test,
) -> None:
    """
    Plot One-vs-Rest ROC curves for all 3 models on one chart.

    Why One-vs-Rest?
      ROC is binary by nature. For 3 classes we treat each class
      as positive vs all others, plot 3 curves per model.
    """
    # Binarize y_test for one-vs-rest ROC
    y_bin = label_binarize(y_test, classes=LABEL_INT)

    colors = {
        "logistic_regression": "#2196F3",
        "naive_bayes":         "#FF9800",
        "random_forest":       "#4CAF50",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "ROC Curves — One-vs-Rest (per sentiment class)",
        fontsize=14, fontweight="bold"
    )

    import scipy.sparse as sp

    for ax, label_idx in zip(axes, range(3)):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")

        for name, result in results.items():
            model = result["model"]

            # Get probability scores
            try:
                if name == "naive_bayes":
                    X = X_test.copy()
                    if sp.issparse(X):
                        X.data = np.clip(X.data, 0, None)
                    proba = model.predict_proba(X)
                else:
                    proba = model.predict_proba(X_test)

                fpr, tpr, _ = roc_curve(y_bin[:, label_idx], proba[:, label_idx])
                auc = roc_auc_score(y_bin[:, label_idx], proba[:, label_idx])

                ax.plot(
                    fpr, tpr,
                    color = colors[name],
                    lw    = 2,
                    label = f"{name.replace('_',' ').title()} (AUC={auc:.3f})",
                )
            except Exception as e:
                logger.warning(f"ROC failed for {name}: {e}")

        ax.set_title(f"Class: {LABEL_NAMES[label_idx].upper()}", fontsize=11)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved ROC curves → {path}")
    plt.show()
    plt.close()


# ── Metrics Bar Chart ─────────────────────────────────────────────────────────

def plot_metrics_comparison(results: dict) -> None:
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1
    across all 3 models side by side.
    """
    df = build_comparison_table(results).drop(columns=["Training Time"])
    metrics     = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = df.index.tolist()
    x           = np.arange(len(metrics))
    width       = 0.22
    colors      = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        values = [df.loc[model_name, m] for m in metrics]
        bars   = ax.bar(x + i * width, values, width, label=model_name, color=color, alpha=0.85)

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

    ax.set_title("Model Performance Comparison — All Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.80, color="red", linestyle="--", alpha=0.4, label="0.80 threshold")

    plt.tight_layout()
    path = PLOTS_DIR / "metrics_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved metrics comparison → {path}")
    plt.show()
    plt.close()


# ── Before vs After Tuning Chart ──────────────────────────────────────────────

def plot_tuning_improvement(
    before_results: dict,
    after_results:  dict,
) -> None:
    """Bar chart showing F1-score before and after tuning per model."""
    model_names = [n.replace("_", " ").title() for n in before_results]
    before_f1   = [
        before_results[n]["report"]["weighted avg"]["f1-score"]
        for n in before_results
    ]
    after_f1    = [
        after_results[n]["report"]["weighted avg"]["f1-score"]
        for n in after_results
    ]

    x     = np.arange(len(model_names))
    width = 0.30

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width/2, before_f1, width, label="Before Tuning", color="#90CAF9", alpha=0.9)
    b2 = ax.bar(x + width/2, after_f1,  width, label="After Tuning",  color="#1565C0", alpha=0.9)

    for bar in list(b1) + list(b2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_title("F1-Score: Before vs After Hyperparameter Tuning",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Weighted F1-Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "tuning_improvement.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved tuning improvement chart → {path}")
    plt.show()
    plt.close()


# ── Final Model Selector ──────────────────────────────────────────────────────

def select_best_model(results: dict) -> tuple[str, object]:
    """
    Pick the best model by F1-Score.
    Ties broken by Accuracy, then Precision.
    Returns (model_name, model_object).
    """
    df = build_comparison_table(results)
    df_sorted = df.sort_values(
        by=["F1-Score", "Accuracy", "Precision"],
        ascending=False
    )
    best_name = df_sorted.index[0]
    best_key  = best_name.lower().replace(" ", "_")

    logger.info(f"🏆 Selected best model: {best_name} "
                f"(F1={df_sorted.loc[best_name, 'F1-Score']})")

    return best_key, results[best_key]["model"]