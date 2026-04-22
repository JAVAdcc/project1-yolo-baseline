from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

from common import append_summary, ensure_dir, seed_everything, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split YOLO-format dataset into train/val/test.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/yolo_format"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--external-test-input",
        type=Path,
        default=None,
        help="Optional YOLO-format dataset root whose images/all and labels/all will be used as the test split.",
    )
    parser.add_argument("--reset-splits", action="store_true", help="Delete existing train/val/test split directories before moving files.")
    return parser.parse_args()


def reset_split_dirs(images_root: Path, labels_root: Path) -> None:
    for split in ["train", "val", "test"]:
        image_dir = images_root / split
        label_dir = labels_root / split
        if image_dir.exists():
            shutil.rmtree(image_dir)
        if label_dir.exists():
            shutil.rmtree(label_dir)


def list_staging_images(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.glob("*") if path.is_file())


def populate_split(
    image_paths: list[Path],
    source_labels_dir: Path,
    target_images_dir: Path,
    target_labels_dir: Path,
    *,
    move_files: bool,
) -> None:
    ensure_dir(target_images_dir)
    ensure_dir(target_labels_dir)
    transfer = shutil.move if move_files else shutil.copy2
    for image_path in image_paths:
        label_path = source_labels_dir / f"{image_path.stem}.txt"
        transfer(str(image_path), str(target_images_dir / image_path.name))
        if label_path.exists():
            transfer(str(label_path), str(target_labels_dir / label_path.name))


def main() -> None:
    args = parse_args()
    setup_logging()
    seed_everything(args.seed)

    images_root = args.input / "images"
    labels_root = args.input / "labels"
    staging_images = images_root / "all"
    staging_labels = labels_root / "all"
    if args.reset_splits:
        logging.warning("Removing existing split directories under %s", args.input.resolve())
        reset_split_dirs(images_root, labels_root)
    unsplit_images = sorted([p for p in staging_images.glob("*") if p.is_file()])

    if not unsplit_images:
        logging.warning(
            "No unsplit images found under %s. This is expected until real data conversion is ready.",
            staging_images.resolve(),
        )
        append_summary(
            "Dataset Split",
            [
                f"No unsplit images found under {staging_images.resolve()}",
                "Split step skipped. This is expected before data conversion is implemented.",
            ],
        )
        return

    shuffled_images = list(unsplit_images)
    random.Random(args.seed).shuffle(shuffled_images)

    total = len(shuffled_images)
    if args.external_test_input is None:
        train_end = int(total * args.train_ratio)
        val_end = train_end + int(total * args.val_ratio)
        buckets = {
            "train": shuffled_images[:train_end],
            "val": shuffled_images[train_end:val_end],
        }
    else:
        train_val_ratio = args.train_ratio + args.val_ratio
        if train_val_ratio <= 0:
            raise ValueError("train_ratio + val_ratio must be positive when using an external test dataset.")
        normalized_train_ratio = args.train_ratio / train_val_ratio
        train_end = int(total * normalized_train_ratio)
        buckets = {
            "train": shuffled_images[:train_end],
            "val": shuffled_images[train_end:],
        }
    summary_lines = [f"Seed used for shuffle: {args.seed}", f"Reset split directories: {args.reset_splits}"]

    if args.external_test_input is None:
        buckets["test"] = shuffled_images[val_end:]
    else:
        external_images_root = args.external_test_input / "images"
        external_labels_root = args.external_test_input / "labels"
        external_staging_images = external_images_root / "all"
        external_staging_labels = external_labels_root / "all"
        external_test_images = list_staging_images(external_staging_images)
        if not external_test_images:
            raise FileNotFoundError(
                f"No external test images found under {external_staging_images.resolve()}"
            )
        buckets["test"] = external_test_images
        summary_lines.append(f"External test input: {args.external_test_input.resolve()}")
        summary_lines.append(
            "Test split populated from external dataset staging directory; train/val ratios were renormalized to consume the full training dataset."
        )

    for split, image_paths in buckets.items():
        if split == "test" and args.external_test_input is not None:
            populate_split(
                image_paths,
                external_staging_labels,
                images_root / split,
                labels_root / split,
                move_files=False,
            )
            continue
        populate_split(
            image_paths,
            staging_labels,
            images_root / split,
            labels_root / split,
            move_files=True,
        )

    append_summary(
        "Dataset Split",
        [f"{split}: {len(paths)} samples" for split, paths in buckets.items()] + summary_lines,
    )


if __name__ == "__main__":
    main()
