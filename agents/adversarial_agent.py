from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from agents.prompts import build_generation_prompt, build_rewrite_prompt
from evaluation.visualization import plot_training_history
from training.experiment_logger import ExperimentLogger
from training.utils import (
    configure_logging,
    count_trainable_parameters,
    ensure_dir,
    get_device,
    load_config,
    maybe_autocast,
    resolve_path,
    save_json,
    set_seed,
)


LOGGER = logging.getLogger(__name__)

try:
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except Exception:  # pragma: no cover - dependency is available only on target system
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


class PromptResponseDataset(Dataset):
    def __init__(self, tokenizer, examples: Sequence[tuple[str, str]], max_length: int) -> None:
        self.records = []
        self.tokenizer = tokenizer
        for prompt, response in examples:
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

            input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
            input_ids = input_ids[:max_length]
            attention_mask = [1] * len(input_ids)
            labels = [-100] * min(len(prompt_ids), len(input_ids))
            remaining_response = input_ids[len(labels) :]
            labels.extend(remaining_response)

            pad_length = max_length - len(input_ids)
            if pad_length > 0:
                input_ids += [tokenizer.pad_token_id] * pad_length
                attention_mask += [0] * pad_length
                labels += [-100] * pad_length

            self.records.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


@dataclass
class SuccessfulEvasionExample:
    original_text: str
    adversarial_text: str
    detector_confidence: float


def _load_quantization_config(use_4bit: bool) -> Optional["BitsAndBytesConfig"]:
    if not use_4bit or not torch.cuda.is_available():
        return None

    try:
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    except Exception as exc:  # pragma: no cover - dependency is available only on target system
        LOGGER.warning("4-bit loading requested but unavailable, falling back to standard weights: %s", exc)
        return None


class AdversarialAgent:
    def __init__(self, config: dict, checkpoint_dir: Optional[str | Path] = None) -> None:
        self.config = config
        self.agent_config = config["agent"]
        self.device = get_device(config["project"].get("device", "auto"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.agent_config["model_name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = _load_quantization_config(self.agent_config.get("use_4bit", False))

        model_kwargs = {}
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(self.agent_config["model_name"], **model_kwargs)

        if checkpoint_dir and resolve_path(checkpoint_dir).exists() and PeftModel is not None:
            try:
                self.model = PeftModel.from_pretrained(self.model, resolve_path(checkpoint_dir))
            except Exception:
                pass

        if self.device.type == "cuda" and "device_map" not in model_kwargs:
            self.model.to(self.device)

        self.model.eval()

    def _generate(self, prompts: Sequence[str], max_new_tokens: Optional[int] = None) -> List[str]:
        if not prompts:
            return []

        encoded = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=self.agent_config["max_length"],
            return_tensors="pt",
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=self.agent_config["temperature"],
                top_p=self.agent_config["top_p"],
                max_new_tokens=max_new_tokens or self.agent_config["max_new_tokens"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated: List[str] = []
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        for row_index, generated_ids in enumerate(outputs):
            new_token_ids = generated_ids[int(prompt_lengths[row_index]) :]
            generated.append(self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip())
        return generated

    def rewrite(self, texts: Sequence[str]) -> List[str]:
        prompts = [build_rewrite_prompt(text) for text in texts]
        return self._generate(prompts)

    def generate(self, topics: Sequence[str]) -> List[str]:
        prompts = [build_generation_prompt(topic) for topic in topics]
        return self._generate(prompts)

    def collect_successful_examples(
        self,
        originals: Sequence[str],
        adversarial_texts: Sequence[str],
        detector_confidences: Sequence[float],
    ) -> List[SuccessfulEvasionExample]:
        examples: List[SuccessfulEvasionExample] = []
        for original, rewritten, confidence in zip(originals, adversarial_texts, detector_confidences):
            if confidence < self.agent_config["evasion_threshold"]:
                examples.append(
                    SuccessfulEvasionExample(
                        original_text=original,
                        adversarial_text=rewritten,
                        detector_confidence=confidence,
                    )
                )
        return examples

    def finetune(
        self,
        examples: Sequence[SuccessfulEvasionExample],
        output_dir: str | Path,
    ) -> dict:
        if not examples:
            LOGGER.info("No successful evasion examples collected; skipping agent fine-tuning.")
            return {"output_dir": str(output_dir), "trainable_parameters": 0, "examples": 0}

        if LoraConfig is None or get_peft_model is None:
            raise ImportError("peft is required to fine-tune the adversarial agent.")

        output_dir = ensure_dir(output_dir)
        set_seed(self.config["project"]["seed"])
        quantization_config = _load_quantization_config(self.agent_config.get("use_4bit", False))

        model_kwargs = {}
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(self.agent_config["model_name"], **model_kwargs)
        if quantization_config is not None and prepare_model_for_kbit_training is not None:
            model = prepare_model_for_kbit_training(model)
        elif self.device.type == "cuda" and "device_map" not in model_kwargs:
            model.to(self.device)

        if self.agent_config.get("gradient_checkpointing", False) and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        lora_config = LoraConfig(
            r=self.agent_config["lora_r"],
            lora_alpha=self.agent_config["lora_alpha"],
            lora_dropout=self.agent_config["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        trainable_parameters = count_trainable_parameters(model)

        dataset = PromptResponseDataset(
            tokenizer=self.tokenizer,
            examples=[(build_rewrite_prompt(item.original_text), item.adversarial_text) for item in examples],
            max_length=self.agent_config["max_length"],
        )
        dataloader = DataLoader(dataset, batch_size=self.agent_config["batch_size"], shuffle=True)
        optimizer = AdamW(model.parameters(), lr=self.agent_config["lr"])
        logger = ExperimentLogger(
            config=self.config,
            run_name=f"agent_{Path(output_dir).name}",
            job_type="agent_finetune",
            tags=["agent", "biogpt", "lora"],
        )

        use_amp = self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_history: List[dict] = []

        for epoch in range(self.agent_config["finetune_epochs"]):
            progress = tqdm(dataloader, desc=f"agent finetune {epoch + 1}", leave=False)
            epoch_losses: List[float] = []
            for step, batch in enumerate(progress, start=1):
                batch = {key: value.to(model.device) for key, value in batch.items()}
                with maybe_autocast(use_amp, self.device):
                    outputs = model(**batch)
                    loss = outputs.loss / self.agent_config["grad_accum_steps"]
                epoch_losses.append(loss.item() * self.agent_config["grad_accum_steps"])
                scaler.scale(loss).backward()

                if step % self.agent_config["grad_accum_steps"] == 0 or step == len(dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                progress.set_postfix(loss=f"{loss.item() * self.agent_config['grad_accum_steps']:.4f}")
            epoch_metrics = {
                "epoch": epoch + 1,
                "finetune_loss": float(sum(epoch_losses) / max(len(epoch_losses), 1)),
            }
            epoch_history.append(epoch_metrics)
            logger.log_metrics(epoch_metrics, step=epoch + 1, prefix="agent")

        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        image_paths = {
            "agent_finetune_curve": plot_training_history(
                epoch_history,
                Path(output_dir) / "agent_finetune_curve.png",
                title="BioGPT Agent Fine-Tuning",
            )
        }
        summary = {
            "output_dir": str(output_dir),
            "trainable_parameters": trainable_parameters,
            "examples": len(examples),
            "epoch_history": epoch_history,
        }
        save_json(summary, Path(output_dir) / "finetune_summary.json")
        logger.log_metrics(
            {
                "trainable_parameters": trainable_parameters,
                "examples": len(examples),
            },
            prefix="agent_summary",
        )
        logger.log_images_from_paths(image_paths)
        logger.finish()
        LOGGER.info("Agent fine-tuning complete: %s", summary)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BioGPT adversarial agent.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--mode", choices=["rewrite", "generate"], required=True, help="Agent mode to run.")
    parser.add_argument("--text", nargs="+", required=True, help="Input abstract(s) or topic(s).")
    parser.add_argument("--checkpoint-dir", default=None, help="Optional fine-tuned checkpoint directory.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    agent = AdversarialAgent(config=config, checkpoint_dir=args.checkpoint_dir)

    if args.mode == "rewrite":
        outputs = agent.rewrite(args.text)
    else:
        outputs = agent.generate(args.text)

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
