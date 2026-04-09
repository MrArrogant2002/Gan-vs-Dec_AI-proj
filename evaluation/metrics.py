from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
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
    prob_array = np.asarray(probabilities, dtype=float)
    prob_array = np.clip(prob_array, 1e-7, 1.0 - 1e-7)
    predictions = (prob_array >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(label_array, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    metrics = {
        "accuracy": float(accuracy_score(label_array, predictions)),
        "precision": float(precision_score(label_array, predictions, zero_division=0)),
        "recall": float(recall_score(label_array, predictions, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(label_array, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(label_array, predictions)),
        "mcc": float(matthews_corrcoef(label_array, predictions)) if len(np.unique(predictions)) > 1 else 0.0,
        "brier_score": float(brier_score_loss(label_array, prob_array)),
        "log_loss": float(log_loss(label_array, np.column_stack([1.0 - prob_array, prob_array]), labels=[0, 1])),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confidence_mean": float(np.mean(prob_array)),
        "confidence_std": float(np.std(prob_array)),
        "positive_prediction_rate": float(np.mean(predictions)),
    }
    if len(np.unique(label_array)) > 1:
        metrics["auc"] = float(roc_auc_score(label_array, prob_array))
        metrics["pr_auc"] = float(average_precision_score(label_array, prob_array))
    else:
        metrics["auc"] = math.nan
        metrics["pr_auc"] = math.nan
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


def compute_confidence_shift(before: Sequence[float], after: Sequence[float]) -> dict:
    if not before or not after:
        return {
            "mean_original_confidence": 0.0,
            "mean_rewritten_confidence": 0.0,
            "mean_confidence_drop": 0.0,
            "median_confidence_drop": 0.0,
        }

    before_array = np.asarray(before, dtype=float)
    after_array = np.asarray(after, dtype=float)
    drops = before_array - after_array
    return {
        "mean_original_confidence": float(before_array.mean()),
        "mean_rewritten_confidence": float(after_array.mean()),
        "mean_confidence_drop": float(drops.mean()),
        "median_confidence_drop": float(np.median(drops)),
    }


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
