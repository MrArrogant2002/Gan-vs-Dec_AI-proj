from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from evaluation.metrics import build_prediction_rows, compute_classification_metrics
from evaluation.visualization import generate_classification_plots, plot_training_history
from training.experiment_logger import ExperimentLogger
from training.utils import (
    configure_logging,
    ensure_dir,
    get_device,
    load_config,
    maybe_autocast,
    resolve_path,
    save_json,
    set_seed,
)


LOGGER = logging.getLogger(__name__)


class SequenceClassificationDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[int]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def read_split(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "text" not in frame.columns or "label" not in frame.columns:
        raise ValueError(f"{path} must contain 'text' and 'label' columns.")
    return frame


def create_dataloader(
    frame: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = SequenceClassificationDataset(
        texts=frame["text"].astype(str).tolist(),
        labels=frame["label"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_inference_dataloader(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
) -> DataLoader:
    dataset = SequenceClassificationDataset(
        texts=list(texts),
        labels=None,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def maybe_enable_gradient_checkpointing(model: PreTrainedModel, enabled: bool) -> None:
    if enabled and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False


@dataclass
class DetectorPredictor:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    max_length: int
    batch_size: int

    @torch.inference_mode()
    def score_texts(self, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []

        loader = create_inference_dataloader(
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )
        self.model.eval()
        scores: List[float] = []
        for batch in loader:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            outputs = self.model(**batch)
            probabilities = torch.softmax(outputs.logits, dim=-1)[:, 1]
            scores.extend(probabilities.detach().cpu().tolist())
        return scores


def load_predictor(
    model_dir: str | Path,
    max_length: int,
    batch_size: int,
    device_name: str = "auto",
) -> DetectorPredictor:
    device = get_device(device_name)
    resolved_dir = resolve_path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(resolved_dir)
    model = AutoModelForSequenceClassification.from_pretrained(resolved_dir)
    model.to(device)
    model.eval()
    return DetectorPredictor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )


def evaluate_model(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    return_predictions: bool = False,
) -> dict | tuple[dict, List[int], List[float]]:
    model.eval()
    losses: List[float] = []
    labels: List[int] = []
    probabilities: List[float] = []

    with torch.inference_mode():
        for batch in dataloader:
            labels.extend(batch["labels"].tolist())
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(outputs.loss.detach().cpu().item())
            scores = torch.softmax(outputs.logits, dim=-1)[:, 1]
            probabilities.extend(scores.detach().cpu().tolist())

    metrics = compute_classification_metrics(labels, probabilities, threshold=threshold)
    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
    if return_predictions:
        return metrics, labels, probabilities
    return metrics


def train_detector(
    config: dict,
    train_path: Optional[str | Path] = None,
    val_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    init_checkpoint: Optional[str | Path] = None,
) -> dict:
    detector_config = config["detector"]
    data_config = config["data"]
    project_config = config["project"]

    set_seed(project_config["seed"])
    device = get_device(project_config.get("device", "auto"))

    train_path = resolve_path(train_path or Path(data_config["processed_path"]) / data_config["train_file"])
    val_path = resolve_path(val_path or Path(data_config["processed_path"]) / data_config["val_file"])
    output_dir = ensure_dir(output_dir or resolve_path(detector_config["checkpoint_dir"]) / "baseline")

    train_frame = read_split(train_path)
    val_frame = read_split(val_path)

    model_source = str(resolve_path(init_checkpoint)) if init_checkpoint else detector_config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=detector_config["num_labels"],
    )
    maybe_enable_gradient_checkpointing(model, detector_config.get("gradient_checkpointing", False))
    model.to(device)

    train_loader = create_dataloader(
        train_frame,
        tokenizer,
        max_length=detector_config["max_length"],
        batch_size=detector_config["batch_size"],
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_frame,
        tokenizer,
        max_length=detector_config["max_length"],
        batch_size=detector_config["batch_size"],
        shuffle=False,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=detector_config["lr"],
        weight_decay=detector_config["weight_decay"],
    )
    total_steps = math.ceil(len(train_loader) / detector_config["grad_accum_steps"]) * detector_config["epochs_per_round"]
    warmup_steps = int(total_steps * detector_config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    use_amp = detector_config.get("fp16", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    logger = ExperimentLogger(
        config=config,
        run_name=f"detector_{Path(output_dir).name}",
        job_type="detector_train",
        tags=["detector", "biobert"],
    )

    best_metrics: Optional[dict] = None
    best_score = float("-inf")
    global_step = 0
    epoch_history: List[dict] = []
    best_prediction_payload: Optional[dict] = None

    LOGGER.info("Training detector on %s using %s", train_path, device)
    for epoch in range(detector_config["epochs_per_round"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"detector epoch {epoch + 1}", leave=False)
        train_losses: List[float] = []

        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with maybe_autocast(use_amp, device):
                outputs = model(**batch)
                loss = outputs.loss / detector_config["grad_accum_steps"]
            train_losses.append(loss.item() * detector_config["grad_accum_steps"])

            scaler.scale(loss).backward()

            if step % detector_config["grad_accum_steps"] == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            progress.set_postfix(loss=f"{loss.item() * detector_config['grad_accum_steps']:.4f}")

        val_metrics, val_labels, val_probabilities = evaluate_model(
            model,
            val_loader,
            device,
            threshold=config["evaluation"].get("threshold", 0.5),
            return_predictions=True,
        )
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": float(sum(train_losses) / max(len(train_losses), 1)),
            **val_metrics,
        }
        epoch_history.append(epoch_metrics)
        LOGGER.info("Detector epoch %s validation metrics: %s", epoch + 1, val_metrics)
        current_score = val_metrics.get("f1", 0.0)
        logger.log_metrics(epoch_metrics, step=epoch + 1, prefix="detector")

        if current_score >= best_score:
            best_score = current_score
            best_metrics = val_metrics
            best_prediction_payload = {
                "labels": val_labels,
                "probabilities": val_probabilities,
            }
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            save_json(
                {
                    "metrics": best_metrics,
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "global_step": global_step,
                    "epoch_history": epoch_history,
                },
                Path(output_dir) / "training_summary.json",
            )

    image_paths = {
        "detector_training_curve": plot_training_history(
            epoch_history,
            Path(output_dir) / "detector_training_curve.png",
            title="Detector Training History",
        )
    }
    if best_prediction_payload:
        prediction_frame = pd.DataFrame(
            build_prediction_rows(
                texts=val_frame["text"].astype(str).tolist(),
                labels=best_prediction_payload["labels"],
                probabilities=best_prediction_payload["probabilities"],
                threshold=config["evaluation"].get("threshold", 0.5),
            )
        )
        prediction_frame.to_csv(Path(output_dir) / "val_predictions.csv", index=False)
        image_paths.update(
            generate_classification_plots(
                labels=best_prediction_payload["labels"],
                probabilities=best_prediction_payload["probabilities"],
                predictions=prediction_frame["predicted_label"].astype(int).tolist(),
                output_dir=Path(output_dir),
                prefix="val",
            )
        )
        logger.log_dataframe("detector_val_predictions", prediction_frame)

    logger.log_images_from_paths(image_paths)
    summary = {
        "output_dir": str(output_dir),
        "best_metrics": best_metrics or {},
        "steps": global_step,
        "epoch_history": epoch_history,
    }
    logger.log_metrics(summary["best_metrics"], prefix="detector_best")
    logger.finish()
    LOGGER.info("Detector training complete: %s", summary)
    return summary


def score_texts(
    model_dir: str | Path,
    texts: Sequence[str],
    config: dict,
) -> List[float]:
    predictor = load_predictor(
        model_dir=model_dir,
        max_length=config["detector"]["max_length"],
        batch_size=config["detector"]["batch_size"],
        device_name=config["project"].get("device", "auto"),
    )
    return predictor.score_texts(texts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BioBERT detector.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--train-path", default=None, help="Optional override for the train CSV.")
    parser.add_argument("--val-path", default=None, help="Optional override for the validation CSV.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for checkpoints.")
    parser.add_argument("--init-checkpoint", default=None, help="Optional checkpoint to continue fine-tuning from.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    train_detector(
        config=config,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        init_checkpoint=args.init_checkpoint,
    )


if __name__ == "__main__":
    main()
