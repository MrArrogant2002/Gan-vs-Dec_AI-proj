from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def compute_classification_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> dict:
    label_array = np.asarray(labels)
    prob_array = np.asarray(probabilities, dtype=float)
    predictions = (prob_array >= threshold).astype(int)

    metrics = {
        "f1": float(f1_score(label_array, predictions, zero_division=0)),
    }
    if len(np.unique(label_array)) > 1:
        metrics["auc"] = float(roc_auc_score(label_array, prob_array))
    else:
        metrics["auc"] = math.nan
    return metrics


def compute_evasion_rate(
    detector_confidences: Sequence[float],
    evasion_threshold: float,
) -> float:
    if not detector_confidences:
        return 0.0
    evaded = sum(score < evasion_threshold for score in detector_confidences)
    return evaded / len(detector_confidences)


def build_prediction_rows(
    texts: Sequence[str],
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> List[dict]:
    predictions = [int(score >= threshold) for score in probabilities]
    rows = []
    for text, label, probability, prediction in zip(texts, labels, probabilities, predictions):
        rows.append(
            {
                "text": text,
                "label": int(label),
                "predicted_label": int(prediction),
                "fake_probability": float(probability),
            }
        )
    return rows


def average_metrics(history: Iterable[dict]) -> dict:
    history_list = list(history)
    if not history_list:
        return {}

    keys = sorted({key for row in history_list for key in row})
    summary = {}
    for key in keys:
        values = [row[key] for row in history_list if key in row and isinstance(row[key], (int, float))]
        summary[key] = float(np.mean(values)) if values else math.nan
    return summary
