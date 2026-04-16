from __future__ import annotations

import argparse
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from evaluation.metrics import average_metrics
from evaluation.visualization import plot_training_history
from models.seqgan.discriminator import DiscriminatorConfig, SeqGANDiscriminator
from models.seqgan.generator import GeneratorConfig, SeqGANGenerator
from models.seqgan.rollout import RolloutPolicy
from training.experiment_logger import ExperimentLogger
from training.utils import configure_logging, ensure_dir, get_device, load_config, resolve_path, save_json, set_seed


LOGGER = logging.getLogger(__name__)

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


@dataclass
class Vocabulary:
    stoi: Dict[str, int]
    itos: Dict[int, str]

    @property
    def pad_token_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.stoi[BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.stoi[EOS_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    def encode(self, text: str, seq_len: int) -> List[int]:
        tokens = [BOS_TOKEN] + text.lower().split()[: seq_len - 2] + [EOS_TOKEN]
        token_ids = [self.stoi.get(token, self.unk_token_id) for token in tokens]
        if len(token_ids) < seq_len:
            token_ids += [self.pad_token_id] * (seq_len - len(token_ids))
        return token_ids[:seq_len]

    def decode(self, token_ids: Sequence[int]) -> str:
        tokens: List[str] = []
        for token_id in token_ids:
            token = self.itos.get(int(token_id), UNK_TOKEN)
            if token in {PAD_TOKEN, BOS_TOKEN}:
                continue
            if token == EOS_TOKEN:
                break
            tokens.append(token)
        return " ".join(tokens)

    def to_dict(self) -> dict:
        return {"stoi": self.stoi, "itos": {str(key): value for key, value in self.itos.items()}}

    @classmethod
    def from_dict(cls, payload: dict) -> "Vocabulary":
        return cls(
            stoi={key: int(value) for key, value in payload["stoi"].items()},
            itos={int(key): value for key, value in payload["itos"].items()},
        )


class GeneratorPretrainDataset(Dataset):
    def __init__(self, sequences: Sequence[Sequence[int]]) -> None:
        self.sequences = [torch.tensor(sequence, dtype=torch.long) for sequence in sequences]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict:
        sequence = self.sequences[index]
        return {
            "input_ids": sequence[:-1],
            "target_ids": sequence[1:],
        }


class SequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[Sequence[int]], labels: Sequence[int]) -> None:
        self.sequences = [torch.tensor(sequence, dtype=torch.long) for sequence in sequences]
        self.labels = [torch.tensor(label, dtype=torch.float32) for label in labels]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


def build_vocab(texts: Sequence[str], vocab_size: int) -> Vocabulary:
    counts: Dict[str, int] = {}
    for text in texts:
        for token in text.lower().split():
            counts[token] = counts.get(token, 0) + 1

    most_common = sorted(counts.items(), key=lambda item: item[1], reverse=True)[: max(vocab_size - len(SPECIAL_TOKENS), 0)]
    stoi = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for token, _ in most_common:
        if token not in stoi:
            stoi[token] = len(stoi)
    itos = {index: token for token, index in stoi.items()}
    return Vocabulary(stoi=stoi, itos=itos)


def load_training_texts(
    train_path: str | Path,
    label_column: str = "label",
    positive_label: int = 1,
) -> List[str]:
    frame = pd.read_csv(train_path)
    if "text" not in frame.columns:
        raise ValueError(f"{train_path} must contain a 'text' column.")
    if label_column in frame.columns:
        frame = frame[frame[label_column].astype(int) == int(positive_label)].copy()
    return frame["text"].dropna().astype(str).tolist()


def prepare_sequences(texts: Sequence[str], vocab: Vocabulary, seq_len: int) -> List[List[int]]:
    return [vocab.encode(text, seq_len) for text in texts]


def pretrain_generator(
    generator: SeqGANGenerator,
    dataloader: DataLoader,
    optimizer: Adam,
    epochs: int,
    device: torch.device,
    gradient_clip_norm: float,
    logger: ExperimentLogger | None = None,
) -> List[dict]:
    history: List[dict] = []
    generator.train()
    for epoch in range(epochs):
        losses: List[float] = []
        for batch in tqdm(dataloader, desc=f"seqgan generator pretrain {epoch + 1}", leave=False):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            loss = generator.pretrain_loss(input_ids, target_ids)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), gradient_clip_norm)
            optimizer.step()
            losses.append(loss.item())
        epoch_metrics = {"epoch": epoch + 1, "generator_pretrain_loss": float(sum(losses) / max(len(losses), 1))}
        history.append(epoch_metrics)
        if logger:
            logger.log_metrics(epoch_metrics, step=epoch + 1, prefix="seqgan_generator_pretrain")
    return history


def pretrain_discriminator(
    discriminator: SeqGANDiscriminator,
    generator: SeqGANGenerator,
    real_sequences: Sequence[Sequence[int]],
    batch_size: int,
    epochs: int,
    device: torch.device,
    bos_token_id: int,
    seq_len: int,
    lr: float,
    logger: ExperimentLogger | None = None,
) -> List[dict]:
    optimizer = Adam(discriminator.parameters(), lr=lr)
    history: List[dict] = []
    sampled_real_count = min(len(real_sequences), batch_size * 20)

    for epoch in range(epochs):
        sampled_reals = random.sample(list(real_sequences), k=sampled_real_count)
        fake_sequences = generator.sample(
            batch_size=sampled_real_count,
            seq_len=seq_len,
            bos_token_id=bos_token_id,
            device=device,
        ).cpu().tolist()
        sequences = sampled_reals + fake_sequences
        labels = [1] * len(sampled_reals) + [0] * len(fake_sequences)
        dataset = SequenceDataset(sequences, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses: List[float] = []
        discriminator.train()
        for input_ids, label_batch in tqdm(dataloader, desc=f"seqgan discriminator pretrain {epoch + 1}", leave=False):
            input_ids = input_ids.to(device)
            label_batch = label_batch.to(device)
            loss = discriminator.loss(input_ids, label_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_metrics = {"epoch": epoch + 1, "discriminator_pretrain_loss": float(sum(losses) / max(len(losses), 1))}
        history.append(epoch_metrics)
        if logger:
            logger.log_metrics(epoch_metrics, step=epoch + 1, prefix="seqgan_discriminator_pretrain")
    return history


def adversarial_train(
    generator: SeqGANGenerator,
    discriminator: SeqGANDiscriminator,
    rollout: RolloutPolicy,
    config: dict,
    device: torch.device,
    bos_token_id: int,
    real_sequences: Sequence[Sequence[int]],
    logger: ExperimentLogger | None = None,
) -> List[dict]:
    seqgan_config = config["seqgan"]
    g_optimizer = Adam(generator.parameters(), lr=seqgan_config["lr_g"])
    d_optimizer = Adam(discriminator.parameters(), lr=seqgan_config["lr_d"])
    history: List[dict] = []

    for epoch in range(seqgan_config["adversarial_epochs"]):
        generator.train()
        sampled_ids = generator.sample(
            batch_size=seqgan_config["batch_size"],
            seq_len=seqgan_config["seq_len"],
            bos_token_id=bos_token_id,
            device=device,
            temperature=seqgan_config["temperature"],
        )
        rewards = rollout.get_rewards(
            sampled_ids=sampled_ids,
            rollout_num=seqgan_config["rollout_num"],
            discriminator=discriminator,
            temperature=seqgan_config["temperature"],
        )
        g_loss = generator.policy_gradient_loss(sampled_ids, rewards)
        g_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), seqgan_config["gradient_clip_norm"])
        g_optimizer.step()

        discriminator.train()
        fake_batch = sampled_ids.detach()
        sampled_reals = random.sample(list(real_sequences), k=min(len(real_sequences), fake_batch.size(0)))
        real_batch = torch.tensor(sampled_reals, dtype=torch.long, device=device)
        min_batch = min(real_batch.size(0), fake_batch.size(0))
        mixed_input = torch.cat([real_batch[:min_batch], fake_batch[:min_batch]], dim=0)
        labels = torch.cat(
            [
                torch.ones(min_batch, device=device),
                torch.zeros(min_batch, device=device),
            ],
            dim=0,
        )
        d_loss = discriminator.loss(mixed_input, labels)
        d_optimizer.zero_grad(set_to_none=True)
        d_loss.backward()
        d_optimizer.step()

        rollout.update_params(generator)
        epoch_metrics = {
            "epoch": epoch + 1,
            "generator_adversarial_loss": float(g_loss.detach().cpu().item()),
            "discriminator_adversarial_loss": float(d_loss.detach().cpu().item()),
        }
        history.append(epoch_metrics)
        if logger:
            logger.log_metrics(epoch_metrics, step=epoch + 1, prefix="seqgan_adversarial")
    return history


def compute_bleu_like_score(references: Sequence[str], candidates: Sequence[str]) -> float:
    if not references or not candidates:
        return 0.0
    scores: List[float] = []
    for reference, candidate in zip(references, candidates):
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()
        if not candidate_tokens:
            scores.append(0.0)
            continue
        overlap = sum(1 for token in candidate_tokens if token in reference_tokens)
        scores.append(overlap / len(candidate_tokens))
    return float(sum(scores) / len(scores))


def compute_perplexity_like(generator_losses: Sequence[float]) -> float:
    if not generator_losses:
        return math.nan
    return float(math.exp(sum(generator_losses) / len(generator_losses)))


def save_seqgan_checkpoint(
    output_dir: Path,
    generator: SeqGANGenerator,
    discriminator: SeqGANDiscriminator,
    vocab: Vocabulary,
    config: dict,
    metrics: dict,
) -> None:
    ensure_dir(output_dir)
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "generator_config": asdict(generator.config),
            "discriminator_config": asdict(discriminator.config),
            "vocab": vocab.to_dict(),
            "metrics": metrics,
            "seqgan_config": config["seqgan"],
        },
        output_dir / "seqgan.pt",
    )
    save_json(metrics, output_dir / "training_summary.json")


def load_seqgan_bundle(checkpoint_dir: str | Path, device_name: str = "auto") -> dict:
    device = get_device(device_name)
    checkpoint_path = resolve_path(checkpoint_dir) / "seqgan.pt"
    payload = torch.load(checkpoint_path, map_location=device)

    generator_config = GeneratorConfig(**payload["generator_config"])
    discriminator_config = DiscriminatorConfig(**payload["discriminator_config"])
    vocab = Vocabulary.from_dict(payload["vocab"])

    generator = SeqGANGenerator(generator_config)
    discriminator = SeqGANDiscriminator(discriminator_config)
    generator.load_state_dict(payload["generator_state_dict"])
    discriminator.load_state_dict(payload["discriminator_state_dict"])
    generator.to(device).eval()
    discriminator.to(device).eval()

    return {
        "generator": generator,
        "discriminator": discriminator,
        "vocab": vocab,
        "config": payload["seqgan_config"],
        "metrics": payload.get("metrics", {}),
        "device": device,
    }


@torch.inference_mode()
def generate_fake_texts(
    checkpoint_dir: str | Path,
    num_samples: int,
    device_name: str = "auto",
) -> List[str]:
    bundle = load_seqgan_bundle(checkpoint_dir, device_name=device_name)
    generator: SeqGANGenerator = bundle["generator"]
    vocab: Vocabulary = bundle["vocab"]
    seqgan_config: dict = bundle["config"]
    device: torch.device = bundle["device"]

    sampled = generator.sample(
        batch_size=num_samples,
        seq_len=seqgan_config["seq_len"],
        bos_token_id=vocab.bos_token_id,
        eos_token_id=vocab.eos_token_id,
        temperature=seqgan_config.get("temperature", 1.0),
        device=device,
    )
    return [vocab.decode(row.tolist()) for row in sampled]


def train_seqgan(
    config: dict,
    train_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    set_seed(config["project"]["seed"])
    seqgan_config = config["seqgan"]
    device = get_device(config["project"].get("device", "auto"))
    logger = ExperimentLogger(
        config=config,
        run_name=f"seqgan_{Path(output_dir or seqgan_config['checkpoint_dir']).name}",
        job_type="seqgan_train",
        tags=["seqgan"],
    )

    data_config = config["data"]
    generator_train_file = data_config.get("generator_train_file", data_config["train_file"])
    train_path = resolve_path(train_path or Path(data_config["processed_path"]) / generator_train_file)
    output_dir = ensure_dir(output_dir or resolve_path(seqgan_config["checkpoint_dir"]))

    texts = load_training_texts(
        train_path,
        label_column=data_config.get("label_column", "label"),
        positive_label=int(data_config.get("positive_label", 1)),
    )
    vocab = build_vocab(texts, vocab_size=seqgan_config["vocab_size"])
    sequences = prepare_sequences(texts, vocab=vocab, seq_len=seqgan_config["seq_len"])

    generator = SeqGANGenerator(
        GeneratorConfig(
            vocab_size=len(vocab.stoi),
            embed_dim=seqgan_config["embed_dim"],
            hidden_dim=seqgan_config["hidden_dim"],
            num_layers=seqgan_config["num_layers"],
            pad_token_id=vocab.pad_token_id,
        )
    ).to(device)
    discriminator = SeqGANDiscriminator(
        DiscriminatorConfig(
            vocab_size=len(vocab.stoi),
            embed_dim=seqgan_config["embed_dim"],
            hidden_dim=seqgan_config["hidden_dim"],
            num_layers=seqgan_config["num_layers"],
            pad_token_id=vocab.pad_token_id,
        )
    ).to(device)

    generator_dataset = GeneratorPretrainDataset(sequences)
    generator_loader = DataLoader(generator_dataset, batch_size=seqgan_config["batch_size"], shuffle=True)
    generator_optimizer = Adam(generator.parameters(), lr=seqgan_config["lr_g"])
    generator_history = pretrain_generator(
        generator=generator,
        dataloader=generator_loader,
        optimizer=generator_optimizer,
        epochs=seqgan_config["pretrain_epochs_g"],
        device=device,
        gradient_clip_norm=seqgan_config["gradient_clip_norm"],
        logger=logger,
    )

    discriminator_history = pretrain_discriminator(
        discriminator=discriminator,
        generator=generator,
        real_sequences=sequences,
        batch_size=seqgan_config["batch_size"],
        epochs=seqgan_config["pretrain_epochs_d"],
        device=device,
        bos_token_id=vocab.bos_token_id,
        seq_len=seqgan_config["seq_len"],
        lr=seqgan_config["lr_d"],
        logger=logger,
    )

    rollout = RolloutPolicy(generator)
    adversarial_history = adversarial_train(
        generator=generator,
        discriminator=discriminator,
        rollout=rollout,
        config=config,
        device=device,
        bos_token_id=vocab.bos_token_id,
        real_sequences=sequences,
        logger=logger,
    )

    generated_texts = generate_fake_texts(output_dir, num_samples=min(32, len(texts))) if (output_dir / "seqgan.pt").exists() else [
        vocab.decode(row.tolist())
        for row in generator.sample(
            batch_size=min(32, len(texts)),
            seq_len=seqgan_config["seq_len"],
            bos_token_id=vocab.bos_token_id,
            eos_token_id=vocab.eos_token_id,
            device=device,
        )
    ]
    reference_texts = texts[: len(generated_texts)]
    metrics = {
        "generator": average_metrics(generator_history),
        "discriminator": average_metrics(discriminator_history),
        "adversarial": average_metrics(adversarial_history),
        "bleu_like": compute_bleu_like_score(reference_texts, generated_texts),
        "perplexity_like": compute_perplexity_like(
            [row["generator_pretrain_loss"] for row in generator_history if "generator_pretrain_loss" in row]
        ),
    }

    save_seqgan_checkpoint(
        output_dir=Path(output_dir),
        generator=generator,
        discriminator=discriminator,
        vocab=vocab,
        config=config,
        metrics=metrics,
    )
    image_paths = {
        "seqgan_generator_pretrain_curve": plot_training_history(
            generator_history,
            Path(output_dir) / "seqgan_generator_pretrain_curve.png",
            title="SeqGAN Generator Pretrain",
        ),
        "seqgan_discriminator_pretrain_curve": plot_training_history(
            discriminator_history,
            Path(output_dir) / "seqgan_discriminator_pretrain_curve.png",
            title="SeqGAN Discriminator Pretrain",
        ),
        "seqgan_adversarial_curve": plot_training_history(
            adversarial_history,
            Path(output_dir) / "seqgan_adversarial_curve.png",
            title="SeqGAN Adversarial Training",
        ),
    }
    logger.log_metrics(metrics["generator"], prefix="seqgan_generator_summary")
    logger.log_metrics(metrics["discriminator"], prefix="seqgan_discriminator_summary")
    logger.log_metrics(metrics["adversarial"], prefix="seqgan_adversarial_summary")
    logger.log_metrics(
        {"bleu_like": metrics["bleu_like"], "perplexity_like": metrics["perplexity_like"]},
        prefix="seqgan_eval",
    )
    logger.log_images_from_paths(image_paths)
    logger.finish()
    LOGGER.info("SeqGAN training complete: %s", metrics)
    return {"output_dir": str(output_dir), "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SeqGAN generator and discriminator.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--train-path", default=None, help="Optional override for the train CSV.")
    parser.add_argument("--output-dir", default=None, help="Optional checkpoint output directory.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    train_seqgan(config=config, train_path=args.train_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
