from __future__ import annotations

import argparse
import ast
import logging
from collections import Counter
from pathlib import Path

import h5py

from common import append_summary, is_image_file, is_label_like_file, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw SpaceNet dataset files.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/spacenet"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    if not args.input.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input}")

    files = [path for path in args.input.rglob("*") if path.is_file()]
    suffix_counter = Counter(path.suffix.lower() or "<no_suffix>" for path in files)
    image_files = [path for path in files if is_image_file(path)]
    label_files = [path for path in files if is_label_like_file(path)]

    sample_label_preview = "N/A"
    if label_files:
        sample = label_files[0]
        try:
            sample_label_preview = sample.read_text(encoding="utf-8", errors="ignore")[:500].strip()
        except Exception as exc:
            sample_label_preview = f"Failed to read {sample.name}: {exc}"

    h5_files = [path for path in files if path.suffix.lower() in {".h5", ".hdf5"}]
    h5_summary_lines = []
    if h5_files:
        h5_path = h5_files[0]
        with h5py.File(h5_path, "r") as handle:
            keys = list(handle.keys())
            h5_summary_lines.append(f"H5 file: {h5_path.name}")
            h5_summary_lines.append(f"H5 keys: {keys}")
            if "waveforms" in handle:
                h5_summary_lines.append(
                    f"waveforms shape: {handle['waveforms'].shape}, dtype: {handle['waveforms'].dtype}"
                )
            if "labels" in handle:
                labels = handle["labels"]
                sample_raw = labels[0]
                if isinstance(sample_raw, bytes):
                    sample_raw = sample_raw.decode("utf-8", errors="replace")
                bands = ast.literal_eval(sample_raw)
                h5_summary_lines.append(f"sample label[0]: {sample_raw}")
                h5_summary_lines.append(f"sample band count: {len(bands)}")

    logging.info("Scanned %d files from %s", len(files), args.input.resolve())
    logging.info("Image files: %d", len(image_files))
    logging.info("Label-like files: %d", len(label_files))
    logging.info("Suffix histogram: %s", dict(suffix_counter))

    append_summary(
        "Dataset Inspection",
        [
            f"Input directory: {args.input.resolve()}",
            f"Total files: {len(files)}",
            f"Image files: {len(image_files)}",
            f"Label-like files: {len(label_files)}",
            f"Suffix histogram: {dict(suffix_counter)}",
            f"Sample label preview: {sample_label_preview or 'empty'}",
            "If the dataset only provides frequency ranges without time ranges, conversion will use a fallback full-width box strategy.",
            *h5_summary_lines,
        ],
    )


if __name__ == "__main__":
    main()
