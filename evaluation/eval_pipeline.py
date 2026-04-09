from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from evaluation.metrics import (
    compute_classification_metrics,
    compute_evasion_rate,
    compute_rewrite_quality,
)
from models.detector.train_detector import load_predictor
from training.utils import configure_logging, ensure_dir, load_config, resolve_path, save_json


LOGGER = logging.getLogger(__name__)


def evaluate_detector_checkpoint(
    config: dict,
    model_dir: str | Path,
    split_path: str | Path,
    rewrites_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> dict:
    frame = pd.read_csv(resolve_path(split_path))
    predictor = load_predictor(
        model_dir=model_dir,
        max_length=config["detector"]["max_length"],
        batch_size=config["detector"]["batch_size"],
        device_name=config["project"].get("device", "auto"),
    )

    scores = predictor.score_texts(frame["text"].astype(str).tolist())
    metrics = compute_classification_metrics(frame["label"].astype(int).tolist(), scores)
    metrics["num_samples"] = len(frame)

    if rewrites_path:
        rewrite_frame = pd.read_csv(resolve_path(rewrites_path))
        if {"original_text", "rewritten_text", "detector_confidence"}.issubset(rewrite_frame.columns):
            metrics["evasion_rate"] = compute_evasion_rate(
                rewrite_frame["detector_confidence"].astype(float).tolist(),
                evasion_threshold=config["agent"]["evasion_threshold"],
            )
            metrics["rewrite_quality"] = compute_rewrite_quality(
                rewrite_frame["original_text"].astype(str).tolist(),
                rewrite_frame["rewritten_text"].astype(str).tolist(),
            )

    if output_path:
        save_json(metrics, output_path)

    LOGGER.info("Evaluation metrics: %s", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a detector checkpoint.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--model-dir", required=True, help="Checkpoint directory to evaluate.")
    parser.add_argument("--split-path", required=True, help="CSV split with text and label columns.")
    parser.add_argument("--rewrites-path", default=None, help="Optional CSV with rewrite analysis columns.")
    parser.add_argument("--output-path", default=None, help="Optional JSON path for saved metrics.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    evaluate_detector_checkpoint(
        config=config,
        model_dir=args.model_dir,
        split_path=args.split_path,
        rewrites_path=args.rewrites_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
