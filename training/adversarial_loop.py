from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch

from agents.adversarial_agent import AdversarialAgent, SuccessfulEvasionExample
from evaluation.eval_pipeline import evaluate_detector_checkpoint
from evaluation.metrics import compute_evasion_rate, compute_rewrite_quality, compute_robustness_delta
from models.detector.train_detector import score_texts, train_detector
from models.seqgan.train_seqgan import generate_fake_texts, train_seqgan
from training.utils import configure_logging, ensure_dir, load_config, resolve_path, save_json, set_seed


LOGGER = logging.getLogger(__name__)


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def select_hard_samples(
    fake_texts: List[str],
    confidences: List[float],
    high_conf_threshold: float,
    top_k: int,
) -> tuple[List[str], List[float]]:
    ranked = sorted(zip(fake_texts, confidences), key=lambda item: item[1], reverse=True)
    selected = [item for item in ranked if item[1] >= high_conf_threshold][:top_k]
    if not selected:
        selected = ranked[:top_k]
    hard_samples = [text for text, _ in selected]
    hard_scores = [score for _, score in selected]
    return hard_samples, hard_scores


def build_augmented_split(
    train_frame: pd.DataFrame,
    adversarial_samples: List[str],
) -> pd.DataFrame:
    if not adversarial_samples:
        return train_frame.copy()

    adversarial_frame = pd.DataFrame(
        {
            "title": [""] * len(adversarial_samples),
            "text": adversarial_samples,
            "label": [1] * len(adversarial_samples),
        }
    )
    return pd.concat([train_frame, adversarial_frame], ignore_index=True)


def run_adversarial_loop(
    config: dict,
    detector_checkpoint: Optional[str | Path] = None,
    seqgan_checkpoint: Optional[str | Path] = None,
    agent_checkpoint: Optional[str | Path] = None,
) -> dict:
    set_seed(config["project"]["seed"])
    data_config = config["data"]
    loop_config = config["loop"]
    detector_config = config["detector"]

    train_path = resolve_path(Path(data_config["processed_path"]) / data_config["train_file"])
    val_path = resolve_path(Path(data_config["processed_path"]) / data_config["val_file"])
    test_path = resolve_path(Path(data_config["processed_path"]) / data_config["test_file"])
    train_frame = pd.read_csv(train_path)

    if detector_checkpoint is None:
        detector_checkpoint = resolve_path(detector_config["checkpoint_dir"]) / "baseline"
        if not detector_checkpoint.exists():
            LOGGER.info("Baseline detector checkpoint missing; training baseline detector first.")
            train_detector(config=config, output_dir=detector_checkpoint)

    if seqgan_checkpoint is None:
        seqgan_checkpoint = resolve_path(config["seqgan"]["checkpoint_dir"])
        if not (Path(seqgan_checkpoint) / "seqgan.pt").exists():
            LOGGER.info("SeqGAN checkpoint missing; training SeqGAN first.")
            train_seqgan(config=config, output_dir=seqgan_checkpoint)

    metrics_dir = ensure_dir(resolve_path(config["evaluation"]["metrics_path"]))
    round_data_dir = ensure_dir(resolve_path(loop_config["round_data_dir"]))

    baseline_metrics = evaluate_detector_checkpoint(
        config=config,
        model_dir=detector_checkpoint,
        split_path=test_path,
        output_path=metrics_dir / "baseline_metrics.json",
    )
    baseline_auc = baseline_metrics.get("auc", float("nan"))

    history = [baseline_metrics | {"round": 0}]
    current_detector_checkpoint = detector_checkpoint
    current_agent_checkpoint = agent_checkpoint

    for round_index in range(1, loop_config["num_rounds"] + 1):
        LOGGER.info("Starting adversarial round %s", round_index)
        round_dir = ensure_dir(round_data_dir / f"round_{round_index:02d}")

        fake_texts = generate_fake_texts(
            checkpoint_dir=seqgan_checkpoint,
            num_samples=loop_config["fake_pool_size"],
            device_name=config["project"].get("device", "auto"),
        )

        detector_scores = score_texts(
            model_dir=current_detector_checkpoint,
            texts=fake_texts,
            config=config,
        )
        release_cuda_memory()

        hard_samples, hard_scores = select_hard_samples(
            fake_texts=fake_texts,
            confidences=detector_scores,
            high_conf_threshold=config["agent"]["high_conf_threshold"],
            top_k=loop_config["hard_sample_top_k"],
        )

        agent = AdversarialAgent(config=config, checkpoint_dir=current_agent_checkpoint)
        rewritten_samples = agent.rewrite(hard_samples)
        del agent
        release_cuda_memory()

        rewritten_scores = score_texts(
            model_dir=current_detector_checkpoint,
            texts=rewritten_samples,
            config=config,
        )
        release_cuda_memory()

        rewrite_frame = pd.DataFrame(
            {
                "original_text": hard_samples,
                "rewritten_text": rewritten_samples,
                "original_detector_confidence": hard_scores,
                "detector_confidence": rewritten_scores,
            }
        )
        rewrite_frame.to_csv(round_dir / "rewrites.csv", index=False)

        augmented_train = build_augmented_split(train_frame, rewritten_samples)
        augmented_train_path = round_dir / "train_augmented.csv"
        augmented_train.to_csv(augmented_train_path, index=False)

        detector_round_dir = ensure_dir(resolve_path(detector_config["checkpoint_dir"]) / f"round_{round_index:02d}")
        train_detector(
            config=config,
            train_path=augmented_train_path,
            val_path=val_path,
            output_dir=detector_round_dir,
            init_checkpoint=current_detector_checkpoint,
        )
        current_detector_checkpoint = detector_round_dir
        release_cuda_memory()

        round_metrics = evaluate_detector_checkpoint(
            config=config,
            model_dir=current_detector_checkpoint,
            split_path=test_path,
            rewrites_path=round_dir / "rewrites.csv",
            output_path=metrics_dir / f"round_{round_index:02d}_metrics.json",
        )
        round_metrics["round"] = round_index
        round_metrics["evasion_rate"] = compute_evasion_rate(
            rewrite_frame["detector_confidence"].astype(float).tolist(),
            evasion_threshold=config["agent"]["evasion_threshold"],
        )
        round_metrics["rewrite_quality"] = compute_rewrite_quality(
            rewrite_frame["original_text"].astype(str).tolist(),
            rewrite_frame["rewritten_text"].astype(str).tolist(),
        )
        round_metrics["robustness_delta"] = compute_robustness_delta(
            round_metrics.get("auc", float("nan")),
            baseline_auc,
        )
        history.append(round_metrics)

        successful_examples = [
            {
                "original_text": row["original_text"],
                "adversarial_text": row["rewritten_text"],
                "detector_confidence": row["detector_confidence"],
            }
            for _, row in rewrite_frame.iterrows()
            if row["detector_confidence"] < config["agent"]["evasion_threshold"]
        ]
        if successful_examples:
            agent = AdversarialAgent(config=config, checkpoint_dir=current_agent_checkpoint)
            current_agent_checkpoint = ensure_dir(resolve_path(config["agent"]["checkpoint_dir"]) / f"round_{round_index:02d}")
            agent.finetune(
                examples=[
                    SuccessfulEvasionExample(
                        original_text=example["original_text"],
                        adversarial_text=example["adversarial_text"],
                        detector_confidence=example["detector_confidence"],
                    )
                    for example in successful_examples
                ],
                output_dir=current_agent_checkpoint,
            )
            del agent
            release_cuda_memory()

        save_json({"history": history}, metrics_dir / "history.json")

    summary = {"baseline": baseline_metrics, "history": history}
    LOGGER.info("Adversarial loop finished.")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the adversarial training loop.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--detector-checkpoint", default=None, help="Optional baseline detector checkpoint.")
    parser.add_argument("--seqgan-checkpoint", default=None, help="Optional SeqGAN checkpoint directory.")
    parser.add_argument("--agent-checkpoint", default=None, help="Optional agent adapter checkpoint directory.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    run_adversarial_loop(
        config=config,
        detector_checkpoint=args.detector_checkpoint,
        seqgan_checkpoint=args.seqgan_checkpoint,
        agent_checkpoint=args.agent_checkpoint,
    )


if __name__ == "__main__":
    main()
