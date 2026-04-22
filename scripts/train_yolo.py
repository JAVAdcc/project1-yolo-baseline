from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from common import (
    IMAGE_SUFFIXES,
    WEIGHTS_DIR,
    append_summary,
    build_resolved_data_yaml,
    count_files,
    setup_logging,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO baseline.")
    parser.add_argument("--data", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--model", type=str, default="weights/yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="yolo11n_spacenet_baseline")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--rect", action="store_true")
    parser.add_argument("--amp", type=str, default="true")
    return parser.parse_args()


def resolve_model(model_arg: str) -> str:
    candidate = Path(model_arg)
    if candidate.exists():
        return str(candidate)
    if not candidate.is_absolute():
        local_candidate = WEIGHTS_DIR / candidate.name
        if local_candidate.exists():
            return str(local_candidate)
    return model_arg


def parse_amp_flag(raw_value: str) -> bool:
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def validate_dataset_config(data_cfg: dict) -> dict:
    names = data_cfg.get("names", {})
    if not names:
        raise ValueError(
            "Dataset config has no class names yet. Real SpaceNet data has not been connected."
        )

    root = Path(data_cfg["path"])
    train_dir = root / data_cfg["train"]
    val_dir = root / data_cfg["val"]
    if count_files(train_dir, IMAGE_SUFFIXES) == 0:
        raise ValueError(f"Training images are missing: {train_dir}")
    if count_files(val_dir, IMAGE_SUFFIXES) == 0:
        raise ValueError(f"Validation images are missing: {val_dir}")
    return data_cfg


def read_results_csv(save_dir: Path) -> dict:
    results_path = save_dir / "results.csv"
    if not results_path.exists():
        return {}
    with results_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    latest = rows[-1]
    parsed = {}
    for key, value in latest.items():
        key = key.strip()
        value = value.strip()
        try:
            parsed[key] = float(value)
        except ValueError:
            parsed[key] = value
    return parsed


def is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def candidate_batches(requested_batch: int) -> list[int]:
    candidates: list[int] = []
    for batch in [requested_batch, 8]:
        if batch > 0 and batch not in candidates:
            candidates.append(batch)
    return candidates


def main() -> None:
    args = parse_args()
    setup_logging()
    import torch
    from ultralytics import YOLO

    data_cfg, resolved_data_path = build_resolved_data_yaml(args.data)
    validate_dataset_config(data_cfg)
    model_path = resolve_model(args.model)
    data_root = Path(data_cfg["path"])
    train_dir = data_root / data_cfg["train"]
    val_dir = data_root / data_cfg["val"]
    train_count = count_files(train_dir, IMAGE_SUFFIXES)
    val_count = count_files(val_dir, IMAGE_SUFFIXES)

    logging.info("Using model: %s", model_path)
    logging.info("Dataset root: %s", data_root)
    logging.info("Train/val images: %d / %d", train_count, val_count)

    results = None
    model_info = None
    actual_batch = args.batch
    amp_enabled = parse_amp_flag(args.amp)
    batch_candidates = candidate_batches(args.batch)
    for batch_size in batch_candidates:
        actual_batch = batch_size
        try:
            logging.info("Starting training attempt with batch=%d", batch_size)
            model = YOLO(model_path)
            if model_info is None:
                model_info = model.info(detailed=False, verbose=False)
            results = model.train(
                data=str(resolved_data_path),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=batch_size,
                device=args.device or None,
                project=args.project,
                name=args.name,
                workers=args.workers,
                patience=args.patience,
                cache=args.cache,
                rect=args.rect,
                amp=amp_enabled,
            )
            break
        except RuntimeError as exc:
            if not is_oom_error(exc) or batch_size == batch_candidates[-1]:
                raise
            logging.warning(
                "Training failed with batch=%d due to OOM. Clearing CUDA cache and retrying with a smaller batch.",
                batch_size,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if results is None:
        raise RuntimeError("Training did not start successfully after batch-size fallback attempts.")

    save_dir = Path(results.save_dir)
    csv_metrics = read_results_csv(save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"
    train_report = {
        "model": model_path,
        "data": str(resolved_data_path),
        "save_dir": str(save_dir),
        "train_images": train_count,
        "val_images": val_count,
        "classes": len(data_cfg["names"]),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": actual_batch,
        "workers": args.workers,
        "patience": args.patience,
        "cache": args.cache,
        "rect": args.rect,
        "amp": amp_enabled,
        "best_weights": str(best_weights),
        "last_weights": str(last_weights),
        "best_weights_mb": round(best_weights.stat().st_size / (1024 * 1024), 3) if best_weights.exists() else None,
        "last_weights_mb": round(last_weights.stat().st_size / (1024 * 1024), 3) if last_weights.exists() else None,
        "results_csv": str(save_dir / "results.csv"),
        "results": csv_metrics,
    }
    write_json(save_dir / "train_report.json", train_report)

    append_summary(
        "Training",
        [
            f"Model: {model_path}",
            f"Classes: {len(data_cfg['names'])}",
            f"Train images: {train_count}",
            f"Val images: {val_count}",
            f"Epochs: {args.epochs}",
            f"Image size: {args.imgsz}",
            f"Batch size: {actual_batch}",
            f"Training output: {save_dir}",
            f"Best weights: {best_weights}",
            f"Last weights: {last_weights}",
            f"Model info: {model_info}",
            f"Train report: {save_dir / 'train_report.json'}",
        ],
    )


if __name__ == "__main__":
    main()
