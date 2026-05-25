from __future__ import annotations

import argparse
import ast
import logging
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from common import PROJECT_ROOT, append_summary, dump_yaml, ensure_dir, setup_logging, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw SpaceNet annotations to YOLO format.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/spacenet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/yolo_format"))
    parser.add_argument("--dataset-file", type=str, default="train.h5")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N samples. 0 means all.")
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--reset-output", action="store_true", help="Delete existing output directory before conversion.")
    parser.add_argument(
        "--skip-dataset-yaml-update",
        action="store_true",
        help="Do not overwrite configs/dataset.yaml after conversion; useful for auxiliary datasets such as an external test set.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose image and label files already exist, useful for resume.",
    )
    return parser.parse_args()


def initialize_yolo_layout(output_dir: Path) -> None:
    for relative in [
        "images/all",
        "images/train",
        "images/val",
        "images/test",
        "labels/all",
        "labels/train",
        "labels/val",
        "labels/test",
    ]:
        ensure_dir(output_dir / relative)


def maybe_reset_output(output_dir: Path, reset_output: bool) -> None:
    if reset_output and output_dir.exists():
        logging.warning("Removing existing output directory before conversion: %s", output_dir.resolve())
        shutil.rmtree(output_dir)


def parse_band_label(raw_label: bytes | str) -> list[tuple[float, float]]:
    if isinstance(raw_label, bytes):
        raw_label = raw_label.decode("utf-8", errors="replace")
    bands = ast.literal_eval(raw_label)
    return [(float(lo), float(hi)) for lo, hi in bands]


def estimate_frequency_bounds(label_dataset: h5py.Dataset) -> tuple[float, float]:
    freq_min = float("inf")
    freq_max = float("-inf")
    for raw in label_dataset:
        for lo, hi in parse_band_label(raw):
            freq_min = min(freq_min, lo)
            freq_max = max(freq_max, hi)
    return freq_min, freq_max


def waveform_to_spectrogram(waveform: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    if waveform.shape[0] < n_fft:
        raise ValueError(f"Waveform length {waveform.shape[0]} is shorter than n_fft={n_fft}")
    frame_count = 1 + (waveform.shape[0] - n_fft) // hop_length
    frames = np.lib.stride_tricks.sliding_window_view(waveform, n_fft)[::hop_length]
    frames = frames[:frame_count]
    window = np.hanning(n_fft).astype(np.float32)
    stft = np.fft.fftshift(np.fft.fft(frames * window[None, :], axis=1), axes=1)
    power = np.log1p(np.abs(stft).astype(np.float32))
    power = power.T
    power = np.flipud(power)
    power -= power.min()
    max_value = power.max()
    if max_value > 0:
        power /= max_value
    return (power * 255.0).clip(0, 255).astype(np.uint8)


def bands_to_yolo_lines(
    bands: list[tuple[float, float]],
    freq_min: float,
    freq_max: float,
) -> tuple[list[str], int]:
    freq_span = freq_max - freq_min
    if freq_span <= 0:
        raise ValueError("Invalid frequency span.")

    yolo_lines = []
    seen = set()
    duplicate_count = 0
    for lo, hi in bands:
        lo = max(freq_min, min(freq_max, lo))
        hi = max(freq_min, min(freq_max, hi))
        if hi <= lo:
            continue
        band_center = (lo + hi) / 2.0
        band_height = (hi - lo) / freq_span
        rel_center_from_bottom = (band_center - freq_min) / freq_span
        y_center = 1.0 - rel_center_from_bottom
        line = f"0 0.5 {y_center:.6f} 1.0 {band_height:.6f}"
        if line in seen:
            duplicate_count += 1
            continue
        seen.add(line)
        yolo_lines.append(line)
    return yolo_lines, duplicate_count


def update_dataset_yaml(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    try:
        relative_output = output_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        logging.info(
            "Skipping configs/dataset.yaml update because output directory is outside project root: %s",
            output_dir,
        )
        return

    config = {
        "path": relative_output.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "signal"},
    }
    dump_yaml(Path("configs/dataset.yaml"), config)


def main() -> None:
    args = parse_args()
    setup_logging()

    if not args.input.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {args.input}")

    maybe_reset_output(args.output, args.reset_output)
    initialize_yolo_layout(args.output)
    dataset_path = args.input / args.dataset_file
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    with h5py.File(dataset_path, "r") as handle:
        if "waveforms" not in handle or "labels" not in handle:
            raise KeyError("Expected `waveforms` and `labels` datasets in the H5 file.")

        waveforms = handle["waveforms"]
        labels = handle["labels"]
        if len(waveforms) != len(labels):
            raise ValueError("waveforms and labels length mismatch.")

        freq_min, freq_max = estimate_frequency_bounds(labels)
        sample_count = len(labels) if args.limit <= 0 else min(args.limit, len(labels))
        logging.info(
            "Converting %d samples from %s with freq range [%.3f, %.3f]",
            sample_count,
            dataset_path.resolve(),
            freq_min,
            freq_max,
        )

        image_dir = args.output / "images" / "all"
        label_dir = args.output / "labels" / "all"
        converted = 0
        skipped = 0
        resumed = 0
        removed_duplicates = 0
        for index in tqdm(range(sample_count), desc="convert_h5_to_yolo"):
            try:
                image_name = f"sample_{index:06d}.png"
                label_name = f"sample_{index:06d}.txt"
                image_path = image_dir / image_name
                label_path = label_dir / label_name
                if args.skip_existing and image_path.exists() and label_path.exists():
                    resumed += 1
                    continue

                waveform = waveforms[index]
                bands = parse_band_label(labels[index])
                image = waveform_to_spectrogram(waveform, n_fft=args.n_fft, hop_length=args.hop_length)
                yolo_lines, duplicate_count = bands_to_yolo_lines(bands, freq_min=freq_min, freq_max=freq_max)
                removed_duplicates += duplicate_count
                if not yolo_lines:
                    skipped += 1
                    continue

                Image.fromarray(image).save(image_path)
                label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
                converted += 1
            except Exception as exc:
                skipped += 1
                logging.warning("Skipping sample %d due to error: %s", index, exc)

    write_json(Path("reports/class_mapping.json"), {"signal": 0})
    if args.skip_dataset_yaml_update:
        logging.info("Skipping configs/dataset.yaml update by request.")
    else:
        update_dataset_yaml(args.output)

    append_summary(
        "YOLO Conversion",
        [
            f"Input directory: {args.input.resolve()}",
            f"Output directory initialized at: {args.output.resolve()}",
            "Unsplit staging directories prepared at images/all and labels/all.",
            f"Dataset file: {dataset_path.resolve()}",
            f"Sample limit: {args.limit if args.limit > 0 else 'all'}",
            "Current H5 parsing assumes labels are frequency-band lists and creates full-width YOLO boxes.",
            "Current class mapping is single-class detection: signal -> 0.",
            f"Converted samples: {converted}",
            f"Resumed samples skipped because outputs already existed: {resumed}",
            f"Removed duplicate label entries: {removed_duplicates}",
            f"Skipped samples: {skipped}",
            f"Skipped dataset.yaml update: {args.skip_dataset_yaml_update}",
        ],
    )


if __name__ == "__main__":
    main()
