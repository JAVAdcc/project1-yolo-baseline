from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import append_summary, read_json, setup_logging, summarize_yolo_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export baseline summary insights.")
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument("--latency", type=Path, default=Path("reports/latency.json"))
    parser.add_argument("--mapping", type=Path, default=Path("reports/class_mapping.json"))
    parser.add_argument("--weights", type=Path, default=Path("runs/train/yolo11n_spacenet_baseline/weights/best.pt"))
    parser.add_argument("--train-dir", type=Path, default=Path("runs/train/smoke32_yolo11n_run2"))
    parser.add_argument("--predictions", type=Path, default=Path("reports/figures/predictions_smoke32/prediction_manifest.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/yolo_format"))
    return parser.parse_args()


def load_latest_row(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def main() -> None:
    args = parse_args()
    setup_logging()

    metrics = read_json(args.metrics, default={})
    latency = read_json(args.latency, default={})
    mapping = read_json(args.mapping, default={})
    predictions = read_json(args.predictions, default={})
    latest_train_metrics = load_latest_row(args.train_dir / "results.csv")
    train_label_stats = summarize_yolo_labels(args.data_root / "labels" / "train")
    val_label_stats = summarize_yolo_labels(args.data_root / "labels" / "val")
    test_label_stats = summarize_yolo_labels(args.data_root / "labels" / "test")
    model_size_mb = None
    if args.weights.exists():
        model_size_mb = args.weights.stat().st_size / (1024 * 1024)

    class_name_map = {
        int(class_id): class_name for class_name, class_id in mapping.items()
    }
    class_histogram_named = {
        class_name_map.get(int(class_id), class_id): count
        for class_id, count in train_label_stats.get("class_histogram", {}).items()
    }
    imbalance_hint = "unavailable"
    if class_histogram_named:
        counts = sorted(class_histogram_named.values())
        imbalance_hint = round(counts[-1] / counts[0], 4) if counts[0] > 0 else "infinite"

    lines = [
        f"Known classes: {len(mapping)}",
        f"Class mapping: {mapping}",
        f"Metrics available: {bool(metrics)}",
        f"mAP50: {metrics.get('map50', 'unavailable')}",
        f"mAP50-95: {metrics.get('map50_95', 'unavailable')}",
        f"Precision: {metrics.get('precision', 'unavailable')}",
        f"Recall: {metrics.get('recall', 'unavailable')}",
        f"Latency available: {bool(latency)}",
        f"Latency mean (ms): {latency.get('mean_ms', 'unavailable')}",
        f"Latency median (ms): {latency.get('median_ms', 'unavailable')}",
        f"Latency device: {latency.get('device', 'unavailable')}",
        f"Best model size (MB): {model_size_mb:.2f}" if model_size_mb is not None else "Best model size (MB): unavailable",
        f"Training directory: {args.train_dir.resolve()}" if args.train_dir.exists() else f"Training directory missing: {args.train_dir}",
        f"Latest training row available: {bool(latest_train_metrics)}",
        f"Latest training row: {latest_train_metrics}" if latest_train_metrics else "Latest training row: unavailable",
        f"Train label stats: {train_label_stats}",
        f"Val label stats: {val_label_stats}",
        f"Test label stats: {test_label_stats}",
        f"Named train class histogram: {class_histogram_named or 'unavailable'}",
        f"Class imbalance ratio (max/min): {imbalance_hint}",
        f"Prediction manifest available: {bool(predictions)}",
        f"Predicted class histogram: {predictions.get('class_histogram', {})}" if predictions else "Predicted class histogram: unavailable",
        f"Zero-detection images: {predictions.get('zero_detection_images', [])}" if predictions else "Zero-detection images: unavailable",
        "Current failure-case export is based on the available prediction manifest and should be refreshed after each new full-data training run.",
    ]

    append_summary("Export Summary", lines)


if __name__ == "__main__":
    main()
