"""Visualization helpers for evaluation reports.

Provides reusable plotting functions for confusion matrices, training curves,
ROC curves, and model comparison charts. All functions accept a ``save_path``
argument — when provided the figure is saved to disk and closed automatically.
"""

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving only

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc


# ── Confusion Matrix ─────────────────────────────────────────

def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str],
    save_path: str | Path | None = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and optionally save a confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Training Curves ──────────────────────────────────────────

def plot_training_curves(
    csv_path: str | Path,
    save_path: str | Path | None = None,
    title: str = "Training Curves",
) -> None:
    """Plot loss and accuracy curves from a Kaggle training_log.csv.

    Expected CSV columns: epoch, train_loss, train_acc, val_loss, val_acc
    """
    import csv as csv_mod

    rows: list[dict] = []
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    train_acc = [float(r["train_acc"]) for r in rows]
    val_acc = [float(r["val_acc"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_loss, label="Train Loss")
    ax1.plot(epochs, val_loss, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, label="Train Acc")
    ax2.plot(epochs, val_acc, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ── ROC Curve ────────────────────────────────────────────────

def plot_roc_curve(
    y_true: Sequence[int],
    y_probs: Sequence[float],
    save_path: str | Path | None = None,
    title: str = "ROC Curve",
) -> float:
    """Plot ROC curve for binary classification (positive class = 1).

    Returns the AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return roc_auc


# ── Model Comparison Bar Chart ───────────────────────────────

def plot_model_comparison(
    model_names: list[str],
    metric_values: list[float],
    metric_name: str = "F1-Score",
    save_path: str | Path | None = None,
    title: str = "Model Comparison",
) -> None:
    """Horizontal bar chart comparing a single metric across models."""
    fig, ax = plt.subplots(figsize=(8, max(4, len(model_names) * 0.8)))
    colors = sns.color_palette("viridis", len(model_names))
    y_pos = range(len(model_names))

    bars = ax.barh(y_pos, metric_values, color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.set_xlabel(metric_name)
    ax.set_title(title)

    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlim(0, max(metric_values) * 1.15 if metric_values else 1)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
