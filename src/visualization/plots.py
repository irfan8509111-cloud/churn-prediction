"""Plotting utilities for exploratory analysis and model evaluation."""

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def plot_correlation_matrix(
    df: pd.DataFrame,
    numerical_cols: List[str],
    output_path: str = "reports/figures/correlation_matrix.png"
) -> None:
    """Plot correlation heatmap for continuous features."""
    corr = df[numerical_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", square=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str = "reports/figures/confusion_matrix.png"
) -> None:
    """Plot confusion matrix with counts and percentages."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"]
    )
    plt.title("Confusion Matrix (Test Set)")
    plt.ylabel("True Status")
    plt.xlabel("Predicted Status")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def plot_model_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str = "reports/figures/model_curves.png"
) -> None:
    """Plot ROC and Precision-Recall curves side by side."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ROC
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    # PR
    axes[1].plot(recall, precision, color="#ff7f0e", lw=2, label=f"PR (AUC = {pr_auc:.3f})")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def plot_feature_importances(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 10,
    output_path: str = "reports/figures/feature_importances.png"
) -> None:
    """Plot top N feature importances from tree-based model."""
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=fi_df, x="importance", y="feature", hue="feature", palette="Blues_r", legend=False)
    plt.title(f"Top {top_n} Most Predictive Features")
    plt.xlabel("Feature Importance (Gini)")
    plt.ylabel("")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()
