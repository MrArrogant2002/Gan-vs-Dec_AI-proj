from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from training.utils import ensure_dir


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def plot_metric_history(
    history: Sequence[dict],
    output_path: str | Path,
    x_key: str,
    y_keys: Sequence[str],
    title: str,
    ylabel: str,
) -> Path:
    _style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    x_values = [row[x_key] for row in history if x_key in row]
    for key in y_keys:
        y_values = [row.get(key, math.nan) for row in history if x_key in row]
        plt.plot(x_values, y_values, marker="o", label=key)
    plt.title(title)
    plt.xlabel(x_key.replace("_", " ").title())
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_round_metrics(history: Sequence[dict], output_dir: str | Path) -> Dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    image_paths: Dict[str, Path] = {}

    metric_groups = {
        "round_performance": ["auc", "f1", "accuracy", "balanced_accuracy"],
        "round_robustness": ["evasion_rate", "rewrite_quality", "robustness_delta"],
    }
    for name, keys in metric_groups.items():
        available = [key for key in keys if any(key in row for row in history)]
        if not available:
            continue
        image_paths[name] = plot_metric_history(
            history=history,
            output_path=output_dir / f"{name}.png",
            x_key="round",
            y_keys=available,
            title=name.replace("_", " ").title(),
            ylabel="Metric Value",
        )
    return image_paths


def plot_training_history(history: Sequence[dict], output_path: str | Path, title: str) -> Path:
    if not history:
        return Path(output_path)
    numeric_keys = [
        key
        for key in history[0].keys()
        if key != "epoch" and any(isinstance(row.get(key), (int, float)) for row in history)
    ]
    return plot_metric_history(
        history=history,
        output_path=output_path,
        x_key="epoch",
        y_keys=numeric_keys,
        title=title,
        ylabel="Value",
    )


def plot_confusion_heatmap(
    labels: Sequence[int],
    predictions: Sequence[int],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> Path:
    _style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_roc(
    labels: Sequence[int],
    probabilities: Sequence[float],
    output_path: str | Path,
    title: str = "ROC Curve",
) -> Path:
    _style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(labels, probabilities)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_precision_recall(
    labels: Sequence[int],
    probabilities: Sequence[float],
    output_path: str | Path,
    title: str = "Precision-Recall Curve",
) -> Path:
    _style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(labels, probabilities)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label="PR")
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_probability_histogram(
    labels: Sequence[int],
    probabilities: Sequence[float],
    output_path: str | Path,
    title: str = "Prediction Confidence Distribution",
) -> Path:
    _style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"label": labels, "probability": probabilities})
    frame["label_name"] = frame["label"].map({0: "Real", 1: "Fake"})

    plt.figure(figsize=(8, 6))
    sns.histplot(data=frame, x="probability", hue="label_name", bins=20, kde=True, stat="density", common_norm=False)
    plt.title(title)
    plt.xlabel("Predicted Fake Probability")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def generate_classification_plots(
    labels: Sequence[int],
    probabilities: Sequence[float],
    predictions: Sequence[int],
    output_dir: str | Path,
    prefix: str,
) -> Dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    paths = {
        f"{prefix}_confusion_matrix": plot_confusion_heatmap(labels, predictions, output_dir / f"{prefix}_confusion_matrix.png"),
        f"{prefix}_confidence_histogram": plot_probability_histogram(
            labels,
            probabilities,
            output_dir / f"{prefix}_confidence_histogram.png",
        ),
    }
    if len(set(labels)) > 1:
        paths[f"{prefix}_roc_curve"] = plot_roc(labels, probabilities, output_dir / f"{prefix}_roc_curve.png")
        paths[f"{prefix}_pr_curve"] = plot_precision_recall(labels, probabilities, output_dir / f"{prefix}_pr_curve.png")
    return paths
