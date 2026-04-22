from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import (
    IMAGE_SUFFIXES,
    append_summary,
    build_resolved_data_yaml,
    setup_logging,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained YOLO model.")
    parser.add_argument("--data", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--benchmark-limit", type=int, default=16)
    parser.add_argument("--benchmark-device", type=str, default="cpu")
    return parser.parse_args()


def benchmark_inference(weights_path: Path, image_paths: list[Path], device: str) -> dict:
    if not image_paths:
        return {}
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    benchmark_device = device or None
    use_cuda_timing = benchmark_device not in {None, "", "cpu"} and torch.cuda.is_available()

    # Warm up a few runs so the reported latency is closer to steady-state inference.
    warmup_images = image_paths[: min(3, len(image_paths))]
    for image_path in warmup_images:
        model.predict(
            source=str(image_path),
            verbose=False,
            save=False,
            conf=0.25,
            device=benchmark_device,
            max_det=300,
        )
    if use_cuda_timing:
        torch.cuda.synchronize()

    latencies_ms = []
    for image_path in image_paths:
        if use_cuda_timing:
            torch.cuda.synchronize()
        start = time.perf_counter()
        model.predict(
            source=str(image_path),
            verbose=False,
            save=False,
            conf=0.25,
            device=benchmark_device,
            max_det=300,
        )
        if use_cuda_timing:
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    ordered = sorted(latencies_ms)
    count = len(ordered)
    return {
        "device": device or "default",
        "samples": count,
        "warmup_samples": len(warmup_images),
        "mean_ms": round(sum(ordered) / count, 4),
        "median_ms": round(ordered[count // 2], 4),
        "min_ms": round(ordered[0], 4),
        "max_ms": round(ordered[-1], 4),
    }


def main() -> None:
    args = parse_args()
    setup_logging()
    from ultralytics import YOLO

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights file does not exist: {args.weights}")

    model = YOLO(str(args.weights))
    data_cfg, resolved_data_path = build_resolved_data_yaml(args.data, prefix="yolo_val_data_cfg_")
    metrics = model.val(data=str(resolved_data_path), split=args.split)
    image_dir = Path(data_cfg["path"]) / data_cfg[args.split]
    benchmark_sources = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )[: args.benchmark_limit] if image_dir.exists() else []
    latency = benchmark_inference(args.weights, benchmark_sources, args.benchmark_device)
    output = {
        "split": args.split,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "latency": latency,
    }

    write_json(Path("reports/metrics.json"), output)
    write_json(Path("reports/latency.json"), latency)
    append_summary(
        "Validation",
        [
            f"Split: {args.split}",
            f"mAP50: {output['map50']:.6f}",
            f"mAP50-95: {output['map50_95']:.6f}",
            f"Precision: {output['precision']:.6f}",
            f"Recall: {output['recall']:.6f}",
            f"Latency benchmark device: {latency.get('device', 'unavailable')}",
            f"Latency mean (ms): {latency.get('mean_ms', 'unavailable')}",
            f"Latency median (ms): {latency.get('median_ms', 'unavailable')}",
        ],
    )


if __name__ == "__main__":
    main()
