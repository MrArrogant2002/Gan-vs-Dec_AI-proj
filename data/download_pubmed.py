from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def download_pubmed(output_path: str | Path) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("slinusc/PubMedAbstractsSubset")
    frame = pd.DataFrame(dataset["train"])[["title", "abstract"]].dropna()
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = download_pubmed(args.output_path)
    print(summary)


if __name__ == "__main__":
    main()
