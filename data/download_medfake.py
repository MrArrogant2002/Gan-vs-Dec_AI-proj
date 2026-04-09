from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


MED_MMHL_REPO = "https://github.com/Yanshen-Sun/Med-MMHL.git"


def clone_med_mmhl(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if output_dir.exists() and any(output_dir.iterdir()):
        return {"output_dir": str(output_dir), "status": "already_exists"}

    subprocess.run(
        ["git", "clone", MED_MMHL_REPO, str(output_dir)],
        check=True,
    )
    return {"output_dir": str(output_dir), "status": "cloned"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Med-MMHL fallback dataset repository.")
    parser.add_argument(
        "--output-dir",
        default="data/raw/med_mmhl",
        help="Destination directory for the cloned repository.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = clone_med_mmhl(args.output_dir)
    print(summary)


if __name__ == "__main__":
    main()
