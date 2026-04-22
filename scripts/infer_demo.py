from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

from common import append_summary, ensure_dir, setup_logging, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on demo images.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("data/processed/yolo_format/images/test"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/predictions"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    from ultralytics import YOLO

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights file does not exist: {args.weights}")
    if not args.source.exists():
        raise FileNotFoundError(f"Source path does not exist: {args.source}")

    ensure_dir(args.output)
    model = YOLO(str(args.weights))

    if args.source.is_dir():
        candidate_images = sorted(path for path in args.source.iterdir() if path.is_file())
        if not candidate_images:
            raise FileNotFoundError(f"No image files found under {args.source}")
        selected_sources = [str(path) for path in candidate_images[: args.limit]]
    else:
        selected_sources = [str(args.source)]

    logging.info("Running inference on %d sample(s)", len(selected_sources))
    results = model.predict(
        source=selected_sources,
        conf=args.conf,
        save=True,
        project=str(args.output.parent),
        name=args.output.name,
        exist_ok=True,
        max_det=300,
    )
    class_hist = Counter()
    per_image = []
    for result in results:
        names = result.names
        cls_values = result.boxes.cls.tolist() if result.boxes is not None else []
        confidences = result.boxes.conf.tolist() if result.boxes is not None else []
        predicted_labels = [names[int(cls_id)] for cls_id in cls_values]
        class_hist.update(predicted_labels)
        per_image.append(
            {
                "image": str(result.path),
                "detections": len(predicted_labels),
                "labels": predicted_labels,
                "confidences": [round(float(score), 6) for score in confidences],
            }
        )
    manifest = {
        "weights": str(args.weights),
        "source": str(args.source),
        "conf": args.conf,
        "requested_limit": args.limit,
        "actual_count": len(selected_sources),
        "class_histogram": dict(class_hist),
        "zero_detection_images": [item["image"] for item in per_image if item["detections"] == 0],
        "images": per_image,
    }
    write_json(args.output / "prediction_manifest.json", manifest)

    append_summary(
        "Inference",
        [
            f"Prediction outputs: {args.output.resolve()}",
            f"Requested sample limit: {args.limit}",
            f"Actual sample count: {len(selected_sources)}",
            f"Result objects returned: {len(results)}",
            f"Confidence threshold: {args.conf}",
            f"Prediction manifest: {args.output / 'prediction_manifest.json'}",
        ],
    )


if __name__ == "__main__":
    main()
