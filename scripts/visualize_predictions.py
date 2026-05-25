from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import ensure_dir, write_json


CLASS_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#800000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create side-by-side GT/prediction visualizations.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("data/processed/yolo_advanced/images/test"))
    parser.add_argument("--labels", type=Path, default=Path("data/processed/yolo_advanced/labels/test"))
    parser.add_argument("--output", type=Path, default=Path("reports/figures/advanced_predictions"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def load_yolo_label(path: Path, image_width: int, image_height: int) -> list[dict]:
    boxes = []
    if not path.exists():
        return boxes
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        class_id, x_center, y_center, width, height = line.split()
        class_id = int(float(class_id))
        x_center = float(x_center) * image_width
        y_center = float(y_center) * image_height
        width = float(width) * image_width
        height = float(height) * image_height
        boxes.append(
            {
                "class_id": class_id,
                "conf": None,
                "xyxy": [
                    x_center - width / 2,
                    y_center - height / 2,
                    x_center + width / 2,
                    y_center + height / 2,
                ],
            }
        )
    return boxes


def draw_boxes(image: Image.Image, boxes: list[dict], names: dict[int, str], title: str) -> Image.Image:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, output.width - 1, 24], fill="#111111")
    draw.text((8, 6), title, fill="#ffffff", font=font)
    for box in boxes:
        class_id = int(box["class_id"])
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        x1, y1, x2, y2 = box["xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = names.get(class_id, str(class_id))
        if box.get("conf") is not None:
            label = f"{label} {box['conf']:.2f}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_x = max(0, min(int(x1), output.width - text_width - 4))
        label_y = max(25, int(y1) - text_height - 6)
        draw.rectangle(
            [label_x, label_y, label_x + text_width + 4, label_y + text_height + 4],
            fill=color,
        )
        draw.text((label_x + 2, label_y + 2), label, fill="#000000", font=font)
    return output


def combine(left: Image.Image, right: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "#ffffff")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    ensure_dir(args.output)
    model = YOLO(str(args.weights))
    names = {int(key): value for key, value in model.names.items()}
    images = sorted(args.images.glob("*.png"))[: args.candidate_limit]
    if not images:
        raise FileNotFoundError(f"No PNG images found under {args.images}")

    selected = []
    manifest = []
    for image_path in images:
        result = model.predict(source=str(image_path), conf=args.conf, save=False, verbose=False, max_det=300)[0]
        pred_boxes = []
        if result.boxes is not None:
            for xyxy, class_id, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                pred_boxes.append({"class_id": int(class_id), "conf": float(conf), "xyxy": xyxy})
        if not pred_boxes:
            continue

        with Image.open(image_path) as image:
            base = image.convert("RGB")
            gt_boxes = load_yolo_label(args.labels / f"{image_path.stem}.txt", base.width, base.height)
            gt_view = draw_boxes(base, gt_boxes, names, "Ground truth")
            pred_view = draw_boxes(base, pred_boxes, names, "Prediction")
            combined = combine(gt_view, pred_view)
        out_path = args.output / f"{image_path.stem}_gt_pred.png"
        combined.save(out_path)
        selected.append(out_path)
        manifest.append(
            {
                "source": str(image_path),
                "output": str(out_path),
                "ground_truth_boxes": len(gt_boxes),
                "predicted_boxes": len(pred_boxes),
                "predicted_classes": [names[int(item["class_id"])] for item in pred_boxes],
                "confidences": [round(float(item["conf"]), 4) for item in pred_boxes],
            }
        )
        if len(selected) >= args.limit:
            break

    write_json(args.output / "visualization_manifest.json", manifest)
    if len(selected) < args.limit:
        raise RuntimeError(f"Only created {len(selected)} visualizations from {len(images)} candidates.")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
