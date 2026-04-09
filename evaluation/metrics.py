from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> dict:
    label_array = np.asarray(labels)
    prob_array = np.asarray(probabilities)
    predictions = (prob_array >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(label_array, predictions)),
        "precision": float(precision_score(label_array, predictions, zero_division=0)),
        "recall": float(recall_score(label_array, predictions, zero_division=0)),
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


def compute_robustness_delta(current_auc: float, baseline_auc: float) -> float:
    if math.isnan(current_auc) or math.isnan(baseline_auc):
        return math.nan
    return current_auc - baseline_auc


def _fallback_text_similarity(references: Sequence[str], candidates: Sequence[str]) -> float:
    scores: List[float] = []
    for reference, candidate in zip(references, candidates):
        reference_tokens = set(reference.lower().split())
        candidate_tokens = set(candidate.lower().split())
        if not reference_tokens and not candidate_tokens:
            scores.append(1.0)
            continue
        overlap = len(reference_tokens & candidate_tokens)
        denom = max(len(reference_tokens | candidate_tokens), 1)
        scores.append(overlap / denom)
    return float(np.mean(scores)) if scores else 0.0


def compute_rewrite_quality(
    references: Sequence[str],
    candidates: Sequence[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> float:
    if not references or not candidates:
        return 0.0

    try:
        from bert_score import score as bert_score

        _, _, f1 = bert_score(
            list(candidates),
            list(references),
            model_type=model_type,
            lang="en",
            verbose=False,
        )
        return float(f1.mean().item())
    except Exception:
        return _fallback_text_similarity(references, candidates)


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
