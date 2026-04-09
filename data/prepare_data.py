from __future__ import annotations

import argparse
import hashlib
import html
import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from training.utils import configure_logging, ensure_dir, load_config, resolve_path, set_seed


LOGGER = logging.getLogger(__name__)

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def discover_text_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"No text column found in columns={list(frame.columns)}")


def normalize_text(text: str) -> str:
    cleaned = html.unescape(str(text))
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def truncate_text(
    text: str,
    max_length: int,
    tokenizer: Optional[AutoTokenizer] = None,
) -> str:
    if tokenizer is None:
        words = text.split()
        return " ".join(words[:max_length])

    token_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def load_source(path: Path) -> pd.DataFrame:
    if path.is_dir():
        frames: List[pd.DataFrame] = []
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in {".csv", ".json", ".jsonl", ".txt"}:
                frames.append(load_source(child))
        if not frames:
            raise FileNotFoundError(f"No supported source files found under {path}")
        return pd.concat(frames, ignore_index=True)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".txt":
        return pd.DataFrame({"text": [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]})
    raise ValueError(f"Unsupported source format: {path}")


def standardize_frame(
    frame: pd.DataFrame,
    text_candidates: Sequence[str],
    label: int,
    title_column: str,
) -> pd.DataFrame:
    text_column = discover_text_column(frame, text_candidates)
    standardized = pd.DataFrame()
    standardized["text"] = frame[text_column].astype(str)
    standardized["title"] = frame[title_column].astype(str) if title_column in frame.columns else ""
    standardized["label"] = label
    return standardized


def clean_frame(
    frame: pd.DataFrame,
    min_words: int,
    max_length: int,
    tokenizer: Optional[AutoTokenizer] = None,
) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned["text"] = cleaned["text"].fillna("").map(normalize_text)
    cleaned = cleaned[cleaned["text"].astype(bool)]
    cleaned["word_count"] = cleaned["text"].map(lambda value: len(value.split()))
    cleaned = cleaned[cleaned["word_count"] >= min_words]
    cleaned["text"] = cleaned["text"].map(lambda value: truncate_text(value, max_length=max_length, tokenizer=tokenizer))
    cleaned["text_hash"] = cleaned["text"].map(text_hash)
    cleaned = cleaned.drop_duplicates(subset="text_hash")
    cleaned = cleaned.drop(columns=["word_count", "text_hash"])
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def load_tokenizer_if_enabled(config: dict) -> Optional[AutoTokenizer]:
    if not config["data"].get("use_model_tokenizer", False):
        return None

    tokenizer_name = config["data"].get("tokenizer_name") or config["detector"]["model_name"]
    try:
        LOGGER.info("Loading tokenizer %s for truncation.", tokenizer_name)
        return AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as exc:  # pragma: no cover - runtime/network dependent
        LOGGER.warning("Falling back to whitespace truncation because tokenizer loading failed: %s", exc)
        return None


def split_frame(
    frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("train/val/test splits must sum to 1.0")

    train_frame, temp_frame = train_test_split(
        frame,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=frame["label"],
    )
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_frame, test_frame = train_test_split(
        temp_frame,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_frame["label"],
    )
    return (
        train_frame.reset_index(drop=True),
        val_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
    )


def prepare_data(config: dict) -> dict:
    data_config = config["data"]
    set_seed(config["project"]["seed"])

    pubmed_path = resolve_path(data_config["pubmed_path"])
    fake_path = resolve_path(data_config["fake_path"])
    output_dir = ensure_dir(resolve_path(data_config["processed_path"]))

    if not pubmed_path.exists():
        raise FileNotFoundError(f"Real dataset not found: {pubmed_path}")
    if not fake_path.exists():
        raise FileNotFoundError(f"Fake dataset not found: {fake_path}")

    tokenizer = load_tokenizer_if_enabled(config)
    text_candidates = data_config.get("text_columns", ["abstract", "text", "content"])
    title_column = data_config.get("title_column", "title")

    LOGGER.info("Loading raw data.")
    real_frame = standardize_frame(load_source(pubmed_path), text_candidates, label=0, title_column=title_column)
    fake_frame = standardize_frame(load_source(fake_path), text_candidates, label=1, title_column=title_column)
    merged = pd.concat([real_frame, fake_frame], ignore_index=True)
    merged = clean_frame(
        merged,
        min_words=data_config["min_words"],
        max_length=data_config["max_length"],
        tokenizer=tokenizer,
    )

    LOGGER.info("Split cleaned dataset into train/val/test.")
    train_frame, val_frame, test_frame = split_frame(
        merged,
        train_ratio=data_config["train_split"],
        val_ratio=data_config["val_split"],
        test_ratio=data_config["test_split"],
        seed=config["project"]["seed"],
    )

    split_paths = {
        "train": output_dir / data_config["train_file"],
        "val": output_dir / data_config["val_file"],
        "test": output_dir / data_config["test_file"],
    }
    train_frame.to_csv(split_paths["train"], index=False)
    val_frame.to_csv(split_paths["val"], index=False)
    test_frame.to_csv(split_paths["test"], index=False)

    summary = {
        "train_samples": len(train_frame),
        "val_samples": len(val_frame),
        "test_samples": len(test_frame),
        "output_dir": str(output_dir),
    }
    LOGGER.info("Prepared dataset: %s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare medical text train/val/test splits.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    prepare_data(config)


if __name__ == "__main__":
    main()
