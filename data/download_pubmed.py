from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def download_pubmed(output_path: str | Path, max_samples: int = 2000) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    streamed_dataset = load_dataset("slinusc/PubMedAbstractsSubset", split="train", streaming=True)
    rows = []
    for item in streamed_dataset:
        title = item.get("title")
        abstract = item.get("abstract")
        if not title or not abstract:
            continue
        rows.append({"title": str(title), "abstract": str(abstract)})
        if len(rows) >= max_samples:
            break

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("PubMed download returned no usable rows.")
    frame["label"] = 0
    frame = frame.rename(columns={"abstract": "text"})
    frame.to_csv(output_path, index=False)
    return {"output_path": str(output_path), "rows": len(frame)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the PubMed abstracts subset.")
    parser.add_argument(
        "--output-path",
        default="data/raw/pubmed_abstracts/pubmed_real.csv",
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum number of usable PubMed rows to stream and save.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = download_pubmed(args.output_path, max_samples=args.max_samples)
    print(summary)


if __name__ == "__main__":
    main()
