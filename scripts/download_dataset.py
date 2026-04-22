from __future__ import annotations

import argparse
import logging
from pathlib import Path

from common import append_summary, ensure_dir, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SpaceNet dataset directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/spacenet"),
        help="Directory where manually downloaded SpaceNet files should be placed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    ensure_dir(args.output)

    logging.warning(
        "Automatic SpaceNet download is not implemented yet. "
        "Place the raw dataset under %s and rerun inspect_dataset.py.",
        args.output.resolve(),
    )

    append_summary(
        "Dataset Download",
        [
            "SpaceNet automatic downloader is intentionally left as a placeholder.",
            f"Expected raw dataset location: {args.output.resolve()}",
            "Reason: dataset access path and annotation structure still require validation on real samples.",
        ],
    )


if __name__ == "__main__":
    main()
