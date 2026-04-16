from __future__ import annotations

import argparse
import hashlib
import html
import logging
import math
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


def discover_optional_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


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


def normalize_binary_labels(
    values: pd.Series,
    positive_label: int,
    negative_label: int,
) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    if numeric_values.notna().all():
        return numeric_values.astype(int)

    positive_tokens = {
        str(positive_label).strip().lower(),
        "1",
        "true",
        "fake",
        "synthetic",
        "generated",
        "yes",
    }
    negative_tokens = {
        str(negative_label).strip().lower(),
        "0",
        "false",
        "real",
        "human",
        "authentic",
        "no",
    }

    normalized = values.astype(str).str.strip().str.lower().map(
        lambda value: positive_label if value in positive_tokens else negative_label if value in negative_tokens else math.nan
    )
    return normalized.astype("float")


def standardize_frame(
    frame: pd.DataFrame,
    text_candidates: Sequence[str],
    title_column: str,
    positive_label: int,
    negative_label: int,
    label_candidates: Sequence[str] | None = None,
    fallback_label: Optional[int] = None,
) -> pd.DataFrame:
    text_column = discover_text_column(frame, text_candidates)
    standardized = pd.DataFrame()
    standardized["text"] = frame[text_column].astype(str)
    standardized["title"] = frame[title_column].astype(str) if title_column in frame.columns else ""
    label_column = discover_optional_column(frame, label_candidates or [])
    if label_column is not None:
        standardized["label"] = normalize_binary_labels(
            frame[label_column],
            positive_label=positive_label,
            negative_label=negative_label,
        )
        standardized = standardized.dropna(subset=["label"]).copy()
        standardized["label"] = standardized["label"].astype(int)
    elif fallback_label is not None:
        standardized["label"] = fallback_label
    else:
        raise ValueError(
            "No label column found and no fallback_label provided. "
            f"Columns={list(frame.columns)} candidates={list(label_candidates or [])}"
        )
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
    detector_source_path = resolve_path(data_config.get("detector_source_path", data_config.get("fake_path", "")))
    sentence_path_value = data_config.get("sentence_path")
    sentence_path = resolve_path(sentence_path_value) if sentence_path_value else None
    output_dir = ensure_dir(resolve_path(data_config["processed_path"]))
    generator_train_path = output_dir / data_config.get("generator_train_file", "generator_train_fake.csv")
    pubmed_reference_path = output_dir / data_config.get("pubmed_reference_file", "pubmed_reference.csv")

    if not pubmed_path.exists():
        raise FileNotFoundError(f"Real dataset not found: {pubmed_path}")
    if not detector_source_path.exists():
        raise FileNotFoundError(f"Detector dataset not found: {detector_source_path}")
    if data_config.get("use_sentence_dataset", False) and sentence_path is not None and not sentence_path.exists():
        raise FileNotFoundError(f"Sentence dataset not found: {sentence_path}")

    tokenizer = load_tokenizer_if_enabled(config)
    text_candidates = data_config.get("text_columns", ["abstract", "text", "content"])
    title_column = data_config.get("title_column", "title")
    label_candidates = data_config.get("source_label_candidates", [data_config.get("label_column", "label")])
    positive_label = int(data_config.get("positive_label", 1))
    negative_label = int(data_config.get("negative_label", 0))
    article_min_words = int(data_config.get("article_min_words", data_config.get("min_words", 50)))
    sentence_min_words = int(data_config.get("sentence_min_words", data_config.get("min_words", 50)))
    pubmed_reference_min_words = int(data_config.get("pubmed_reference_min_words", data_config.get("min_words", 50)))

    LOGGER.info("Loading labeled detector dataset from %s", detector_source_path)
    article_frame = clean_frame(
        standardize_frame(
            load_source(detector_source_path),
            text_candidates=text_candidates,
            title_column=title_column,
            positive_label=positive_label,
            negative_label=negative_label,
            label_candidates=label_candidates,
        ),
        min_words=article_min_words,
        max_length=data_config["max_length"],
        tokenizer=tokenizer,
    )
    detector_frames = [article_frame]
    if data_config.get("use_sentence_dataset", False) and sentence_path is not None:
        LOGGER.info("Merging optional sentence dataset from %s", sentence_path)
        detector_frames.append(
            clean_frame(
                standardize_frame(
                    load_source(sentence_path),
                    text_candidates=text_candidates,
                    title_column=title_column,
                    positive_label=positive_label,
                    negative_label=negative_label,
                    label_candidates=label_candidates,
                ),
                min_words=sentence_min_words,
                max_length=data_config["max_length"],
                tokenizer=tokenizer,
            )
        )

    detector_frame = pd.concat(detector_frames, ignore_index=True)
    detector_frame = detector_frame.drop_duplicates(subset="text").reset_index(drop=True)

    LOGGER.info("Split cleaned detector dataset into train/val/test.")
    train_frame, val_frame, test_frame = split_frame(
        detector_frame,
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

    generator_train = train_frame[train_frame["label"] == positive_label].reset_index(drop=True)
    generator_train.to_csv(generator_train_path, index=False)

    LOGGER.info("Preparing PubMed reference dataset from %s", pubmed_path)
    pubmed_reference = standardize_frame(
        load_source(pubmed_path),
        text_candidates=text_candidates,
        title_column=title_column,
        positive_label=positive_label,
        negative_label=negative_label,
        fallback_label=negative_label,
    )
    pubmed_reference = clean_frame(
        pubmed_reference,
        min_words=pubmed_reference_min_words,
        max_length=data_config["max_length"],
        tokenizer=tokenizer,
    )
    max_pubmed_reference_samples = data_config.get("max_pubmed_reference_samples")
    if max_pubmed_reference_samples:
        pubmed_reference = pubmed_reference.sample(
            min(len(pubmed_reference), int(max_pubmed_reference_samples)),
            random_state=config["project"]["seed"],
        ).reset_index(drop=True)
    pubmed_reference.to_csv(pubmed_reference_path, index=False)

    summary = {
        "train_samples": len(train_frame),
        "val_samples": len(val_frame),
        "test_samples": len(test_frame),
        "generator_train_samples": len(generator_train),
        "pubmed_reference_samples": len(pubmed_reference),
        "detector_label_distribution": train_frame["label"].value_counts().sort_index().to_dict(),
        "detector_source_path": str(detector_source_path),
        "sentence_dataset_used": bool(data_config.get("use_sentence_dataset", False) and sentence_path is not None),
        "article_min_words": article_min_words,
        "sentence_min_words": sentence_min_words,
        "pubmed_reference_min_words": pubmed_reference_min_words,
        "generator_train_path": str(generator_train_path),
        "pubmed_reference_path": str(pubmed_reference_path),
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
