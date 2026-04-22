from __future__ import annotations

import json
import logging
import random
import tempfile
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_SUFFIXES = {".txt", ".json", ".xml", ".csv", ".yaml", ".yml"}


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_summary(title: str, lines: list[str]) -> None:
    ensure_dir(REPORTS_DIR)
    summary_path = REPORTS_DIR / "baseline_summary.md"
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {title}\n\n")
        for line in lines:
            handle.write(f"- {line}\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def is_label_like_file(path: Path) -> bool:
    return path.suffix.lower() in LABEL_SUFFIXES


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_dataset_config(data_path: Path) -> dict[str, Any]:
    data_cfg = load_yaml(data_path)
    root = resolve_project_path(data_cfg["path"])
    resolved_cfg = dict(data_cfg)
    resolved_cfg["path"] = str(root)
    return resolved_cfg


def build_resolved_data_yaml(data_path: Path, prefix: str = "yolo_data_cfg_") -> tuple[dict[str, Any], Path]:
    data_cfg = load_dataset_config(data_path)
    tmp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    resolved_path = tmp_dir / data_path.name
    dump_yaml(resolved_path, data_cfg)
    return data_cfg, resolved_path


def count_files(directory: Path, suffixes: set[str] | None = None) -> int:
    if not directory.exists():
        return 0
    files = [path for path in directory.iterdir() if path.is_file()]
    if suffixes is None:
        return len(files)
    return sum(1 for path in files if path.suffix.lower() in suffixes)


def read_yolo_label_file(path: Path) -> list[dict[str, float]]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO label format in {path}:{line_no}: {line}")
            class_id, x_center, y_center, width, height = fields
            records.append(
                {
                    "class_id": int(float(class_id)),
                    "x_center": float(x_center),
                    "y_center": float(y_center),
                    "width": float(width),
                    "height": float(height),
                }
            )
    return records


def summarize_yolo_labels(labels_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "label_files": 0,
        "boxes": 0,
        "class_histogram": {},
        "width": {},
        "height": {},
        "aspect_ratio": {},
    }
    if not labels_dir.exists():
        return summary

    widths: list[float] = []
    heights: list[float] = []
    aspect_ratios: list[float] = []
    class_histogram: dict[int, int] = {}

    label_files = sorted(path for path in labels_dir.glob("*.txt") if path.is_file())
    for label_path in label_files:
        records = read_yolo_label_file(label_path)
        summary["label_files"] += 1
        for record in records:
            summary["boxes"] += 1
            class_id = int(record["class_id"])
            class_histogram[class_id] = class_histogram.get(class_id, 0) + 1
            width = float(record["width"])
            height = float(record["height"])
            widths.append(width)
            heights.append(height)
            if height > 0:
                aspect_ratios.append(width / height)

    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        ordered = sorted(values)
        size = len(ordered)
        return {
            "min": round(ordered[0], 6),
            "mean": round(sum(ordered) / size, 6),
            "median": round(ordered[size // 2], 6),
            "max": round(ordered[-1], 6),
        }

    summary["class_histogram"] = {str(key): value for key, value in sorted(class_histogram.items())}
    summary["width"] = stats(widths)
    summary["height"] = stats(heights)
    summary["aspect_ratio"] = stats(aspect_ratios)
    return summary
