from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import struct
import zipfile
from collections import Counter
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image
from tqdm import tqdm

from common import PROJECT_ROOT, append_summary, dump_yaml, ensure_dir, setup_logging, write_json
from convert_to_yolo import waveform_to_spectrogram


ADVANCED_CLASS_NAMES = {
    0: "WIFI 20MHz QPSK",
    1: "WIFI 20MHz 16QAM",
    2: "WIFI 20MHz 64QAM",
    3: "WIFI 40MHz QPSK",
    4: "WIFI 40MHz 16QAM",
    5: "WIFI 40MHz 64QAM",
    6: "BLE LE1M",
    7: "BLE LE2M",
    8: "Zigbee",
    9: "LoRa 250KHZ",
    10: "SRRC QPSK",
    11: "SRRC 16QAM",
    12: "AM",
    13: "FM",
}


class RangeReader(io.RawIOBase):
    """Seekable reader for a stored zip entry inside another zip file."""

    def __init__(self, path: Path, start: int, size: int):
        self._handle: BinaryIO = path.open("rb")
        self._start = start
        self._size = size
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == io.SEEK_END:
            new_pos = self._size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if new_pos < 0:
            raise ValueError("Cannot seek before start of range.")
        self._pos = min(new_pos, self._size)
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._pos
        size = min(size, self._size - self._pos)
        if size <= 0:
            return b""
        self._handle.seek(self._start + self._pos)
        data = self._handle.read(size)
        self._pos += len(data)
        return data

    def close(self) -> None:
        try:
            self._handle.close()
        finally:
            super().close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert advanced SpaceNet bin/json data to YOLO format.")
    parser.add_argument("--input-zip", type=Path, default=Path("data/raw/SpaceNet.zip"))
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory containing extracted .bin/.json files.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/yolo_advanced"))
    parser.add_argument(
        "--archives",
        nargs="*",
        default=["train"],
        help="Inner zip filename filters, for example `train`, `test.zip`, or `train-0-1499.zip`.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N samples across selected inputs.")
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Resize spectrograms to a square image of this size. Use 0 to keep native spectrogram dimensions.",
    )
    parser.add_argument("--reset-output", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-dataset-yaml-update", action="store_true")
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


def local_zip_data_offset(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> int:
    zf.fp.seek(info.header_offset)
    header = zf.fp.read(30)
    fields = struct.unpack("<IHHHHHIIIHH", header)
    if fields[0] != 0x04034B50:
        raise ValueError(f"Invalid local file header for {info.filename}")
    name_len = fields[-2]
    extra_len = fields[-1]
    return info.header_offset + 30 + name_len + extra_len


def archive_matches(filename: str, filters: list[str]) -> bool:
    normalized = filename.replace("\\", "/").lower()
    if not normalized.endswith(".zip"):
        return False
    return any(filter_text.lower() in normalized for filter_text in filters)


def selected_inner_archives(input_zip: Path, filters: list[str]) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(input_zip) as outer:
        infos = [info for info in outer.infolist() if archive_matches(info.filename, filters)]
    if not infos:
        raise FileNotFoundError(f"No inner zip archives matched filters {filters} in {input_zip}")
    return sorted(infos, key=lambda info: info.filename)


def open_stored_inner_zip(input_zip: Path, outer_info: zipfile.ZipInfo) -> zipfile.ZipFile:
    if outer_info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(
            f"Inner archive {outer_info.filename} is compressed inside the outer zip; extract it before conversion."
        )
    outer = zipfile.ZipFile(input_zip)
    start = local_zip_data_offset(outer, outer_info)
    outer.close()
    reader = RangeReader(input_zip, start, outer_info.file_size)
    return zipfile.ZipFile(reader)


def read_complex_float16(raw: bytes) -> np.ndarray:
    values = np.frombuffer(raw, dtype=np.float16)
    if values.size < 2:
        raise ValueError("Signal file is too short.")
    if values.size % 2:
        values = values[:-1]
    return values[0::2].astype(np.float32) + 1j * values[1::2].astype(np.float32)


def signal_to_yolo_line(signal: dict, observation_range: list[float], duration_ms: float) -> str | None:
    freq_min, freq_max = map(float, observation_range)
    freq_span = freq_max - freq_min
    if freq_span <= 0 or duration_ms <= 0:
        raise ValueError("Invalid observation range or duration.")

    class_id = int(signal["class"])
    if class_id not in ADVANCED_CLASS_NAMES:
        raise ValueError(f"Unknown advanced class id: {class_id}")

    start_freq = max(freq_min, min(freq_max, float(signal["start_frequency"])))
    end_freq = max(freq_min, min(freq_max, float(signal["end_frequency"])))
    start_time = max(0.0, min(duration_ms, float(signal["start_time"])))
    end_time = max(0.0, min(duration_ms, float(signal["end_time"])))
    if end_freq <= start_freq or end_time <= start_time:
        return None

    x_center = ((start_time + end_time) / 2.0) / duration_ms
    width = (end_time - start_time) / duration_ms
    freq_center = (start_freq + end_freq) / 2.0
    rel_center_from_bottom = (freq_center - freq_min) / freq_span
    y_center = 1.0 - rel_center_from_bottom
    height = (end_freq - start_freq) / freq_span
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def label_to_yolo_lines(label: dict, duration_ms: float) -> tuple[list[str], Counter[int], int]:
    observation_range = label["observation_range"]
    lines: list[str] = []
    class_counts: Counter[int] = Counter()
    clipped_or_skipped = 0
    seen: set[str] = set()
    for signal in label.get("signals", []):
        line = signal_to_yolo_line(signal, observation_range, duration_ms)
        if line is None:
            clipped_or_skipped += 1
            continue
        if line in seen:
            clipped_or_skipped += 1
            continue
        seen.add(line)
        lines.append(line)
        class_counts[int(line.split()[0])] += 1
    return lines, class_counts, clipped_or_skipped


def update_dataset_yaml(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    try:
        relative_output = output_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        logging.info("Skipping configs/dataset.yaml update because output is outside project root: %s", output_dir)
        return
    dump_yaml(
        Path("configs/dataset.yaml"),
        {
            "path": relative_output.as_posix(),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": ADVANCED_CLASS_NAMES,
        },
    )


def stable_sample_stem(source_name: str, bin_name: str) -> str:
    source_stem = Path(source_name).stem.lower().replace("-", "_")
    return f"{source_stem}_{Path(bin_name).stem}"


def convert_one_sample(
    *,
    waveform: np.ndarray,
    label: dict,
    stem: str,
    output_dir: Path,
    n_fft: int,
    hop_length: int,
    image_size: int,
    skip_existing: bool,
) -> tuple[bool, Counter[int], int]:
    image_path = output_dir / "images" / "all" / f"{stem}.png"
    label_path = output_dir / "labels" / "all" / f"{stem}.txt"
    if skip_existing and image_path.exists() and label_path.exists():
        return False, Counter(), 0

    obs_start, obs_end = map(float, label["observation_range"])
    sample_rate_hz = (obs_end - obs_start) * 1e6
    duration_ms = waveform.shape[0] / sample_rate_hz * 1000.0
    yolo_lines, class_counts, skipped_boxes = label_to_yolo_lines(label, duration_ms)
    if not yolo_lines:
        return False, Counter(), skipped_boxes

    pil_image = Image.fromarray(waveform_to_spectrogram(waveform, n_fft=n_fft, hop_length=hop_length))
    if image_size > 0:
        pil_image = pil_image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    pil_image.save(image_path)
    label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
    return True, class_counts, skipped_boxes


def convert_from_directory(args: argparse.Namespace) -> tuple[int, int, Counter[int]]:
    assert args.input_dir is not None
    json_paths = sorted(args.input_dir.rglob("*.json"), key=lambda path: str(path))
    converted = 0
    skipped = 0
    class_counts: Counter[int] = Counter()
    selected = json_paths if args.limit <= 0 else json_paths[: args.limit]
    for json_path in tqdm(selected, desc="convert_advanced_dir"):
        bin_path = json_path.with_suffix(".bin")
        if not bin_path.exists():
            skipped += 1
            logging.warning("Missing bin file for label: %s", json_path)
            continue
        try:
            label = json.loads(json_path.read_text(encoding="utf-8"))
            waveform = read_complex_float16(bin_path.read_bytes())
            stem = stable_sample_stem(json_path.parent.name or "advanced", bin_path.stem)
            ok, counts, skipped_boxes = convert_one_sample(
                waveform=waveform,
                label=label,
                stem=stem,
                output_dir=args.output,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                image_size=args.image_size,
                skip_existing=args.skip_existing,
            )
            converted += int(ok)
            skipped += skipped_boxes + int(not ok)
            class_counts.update(counts)
        except Exception as exc:
            skipped += 1
            logging.warning("Skipping %s due to error: %s", json_path, exc)
    return converted, skipped, class_counts


def convert_from_zip(args: argparse.Namespace) -> tuple[int, int, Counter[int]]:
    archives = selected_inner_archives(args.input_zip, args.archives)
    converted = 0
    skipped = 0
    class_counts: Counter[int] = Counter()

    for outer_info in archives:
        if args.limit > 0 and converted >= args.limit:
            break
        logging.info("Reading inner archive: %s", outer_info.filename)
        with open_stored_inner_zip(args.input_zip, outer_info) as inner:
            json_infos = sorted(
                [info for info in inner.infolist() if info.filename.lower().endswith(".json")],
                key=lambda info: info.filename,
            )
            for json_info in tqdm(json_infos, desc=Path(outer_info.filename).name):
                if args.limit > 0 and converted >= args.limit:
                    break
                bin_name = str(Path(json_info.filename).with_suffix(".bin")).replace("\\", "/")
                try:
                    label = json.loads(inner.read(json_info).decode("utf-8"))
                    waveform = read_complex_float16(inner.read(bin_name))
                    stem = stable_sample_stem(Path(outer_info.filename).name, bin_name)
                    ok, counts, skipped_boxes = convert_one_sample(
                        waveform=waveform,
                        label=label,
                        stem=stem,
                        output_dir=args.output,
                        n_fft=args.n_fft,
                        hop_length=args.hop_length,
                        image_size=args.image_size,
                        skip_existing=args.skip_existing,
                    )
                    converted += int(ok)
                    skipped += skipped_boxes + int(not ok)
                    class_counts.update(counts)
                except Exception as exc:
                    skipped += 1
                    logging.warning("Skipping %s inside %s due to error: %s", json_info.filename, outer_info.filename, exc)
    return converted, skipped, class_counts


def main() -> None:
    args = parse_args()
    setup_logging()
    maybe_reset_output(args.output, args.reset_output)
    initialize_yolo_layout(args.output)

    if args.input_dir is not None:
        if not args.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
        converted, skipped, class_counts = convert_from_directory(args)
        input_description = str(args.input_dir.resolve())
    else:
        if not args.input_zip.exists():
            raise FileNotFoundError(f"Input zip does not exist: {args.input_zip}")
        converted, skipped, class_counts = convert_from_zip(args)
        input_description = f"{args.input_zip.resolve()} archives={args.archives}"

    mapping = {name: class_id for class_id, name in ADVANCED_CLASS_NAMES.items()}
    write_json(Path("reports/class_mapping.json"), mapping)
    write_json(
        Path("reports/advanced_class_counts.json"),
        {ADVANCED_CLASS_NAMES[class_id]: count for class_id, count in sorted(class_counts.items())},
    )
    if args.skip_dataset_yaml_update:
        logging.info("Skipping configs/dataset.yaml update by request.")
    else:
        update_dataset_yaml(args.output)

    append_summary(
        "Advanced YOLO Conversion",
        [
            f"Input: {input_description}",
            f"Output: {args.output.resolve()}",
            f"Sample limit: {args.limit if args.limit > 0 else 'all'}",
            f"Output image size: {args.image_size if args.image_size > 0 else 'native'}",
            "Advanced parser reads float16 interleaved IQ .bin files and object-level JSON labels.",
            "YOLO boxes use JSON time bounds for x and observation frequency range for y.",
            f"Classes: {len(ADVANCED_CLASS_NAMES)}",
            f"Converted samples: {converted}",
            f"Skipped samples/boxes: {skipped}",
            f"Class counts: {dict(sorted(class_counts.items()))}",
        ],
    )
    logging.info("Converted %d samples; skipped %d samples/boxes.", converted, skipped)


if __name__ == "__main__":
    main()
