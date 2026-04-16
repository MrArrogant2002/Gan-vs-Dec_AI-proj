from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from evaluation.metrics import (
    build_prediction_rows,
    compute_classification_metrics,
    compute_evasion_rate,
)
from evaluation.visualization import generate_classification_plots
from models.detector.train_detector import load_predictor
from training.experiment_logger import ExperimentLogger
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
    threshold = config["evaluation"].get("threshold", 0.5)
    predictor = load_predictor(
        model_dir=model_dir,
        max_length=config["detector"]["max_length"],
        batch_size=config["detector"]["batch_size"],
        device_name=config["project"].get("device", "auto"),
    )

    texts = frame["text"].astype(str).tolist()
    labels = frame["label"].astype(int).tolist()
    scores = predictor.score_texts(texts)
    metrics = compute_classification_metrics(labels, scores, threshold=threshold)
    metrics["num_samples"] = len(frame)

    output_dir = ensure_dir(
        (Path(output_path).parent if output_path else resolve_path(config["evaluation"]["metrics_path"]))
    )
    prediction_frame = pd.DataFrame(build_prediction_rows(texts, labels, scores, threshold=threshold))
    if config["evaluation"].get("log_predictions", True):
        prediction_frame.to_csv(output_dir / f"{Path(split_path).stem}_predictions.csv", index=False)

    if rewrites_path:
        rewrite_frame = pd.read_csv(resolve_path(rewrites_path))
        if {"original_text", "rewritten_text", "detector_confidence"}.issubset(rewrite_frame.columns):
            metrics["evasion_rate"] = compute_evasion_rate(
                rewrite_frame["detector_confidence"].astype(float).tolist(),
                evasion_threshold=config["agent"]["evasion_threshold"],
            )

    image_paths = {}
    if config["evaluation"].get("save_plots", True):
        image_paths = generate_classification_plots(
            labels=labels,
            probabilities=scores,
            predictions=prediction_frame["predicted_label"].astype(int).tolist(),
            output_dir=resolve_path(config["evaluation"]["plots_path"]),
            prefix=Path(split_path).stem,
        )

    if output_path:
        save_json(metrics, output_path)

    logger = ExperimentLogger(
        config=config,
        run_name=f"eval_{Path(model_dir).name}_{Path(split_path).stem}",
        job_type="evaluation",
        tags=["evaluation"],
    )
    logger.log_metrics(metrics, prefix="evaluation")
    logger.log_dataframe(f"{Path(split_path).stem}_predictions", prediction_frame)
    logger.log_images_from_paths(image_paths)
    logger.finish()

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
