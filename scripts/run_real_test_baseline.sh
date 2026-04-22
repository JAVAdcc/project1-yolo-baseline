#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/x/miniconda3/envs/evo-rl/bin/python}"

TRAIN_RAW_DIR="${ROOT_DIR}/data/raw/spacenet"
TRAIN_H5="${TRAIN_RAW_DIR}/train_full.h5"
TEST_H5="${TRAIN_RAW_DIR}/test_full.h5"

MAIN_OUTPUT="${ROOT_DIR}/data/processed/yolo_format"
EXTERNAL_TEST_OUTPUT="${ROOT_DIR}/data/processed/yolo_format_test_external"
RUN_NAME="${RUN_NAME:-realtest_yolo11n_e50_gpu3}"
DEVICE="${DEVICE:-3}"

cd "${ROOT_DIR}"

if [[ ! -f "${TRAIN_H5}" ]]; then
  echo "Missing training H5: ${TRAIN_H5}" >&2
  exit 1
fi

if [[ ! -f "${TEST_H5}" ]]; then
  echo "Missing test H5: ${TEST_H5}" >&2
  exit 1
fi

echo "[1/6] Convert training H5 into YOLO staging output"
"${PYTHON_BIN}" scripts/convert_to_yolo.py \
  --input data/raw/spacenet \
  --dataset-file train_full.h5 \
  --output data/processed/yolo_format \
  --reset-output

echo "[2/6] Convert real test H5 into external YOLO staging output"
"${PYTHON_BIN}" scripts/convert_to_yolo.py \
  --input data/raw/spacenet \
  --dataset-file test_full.h5 \
  --output data/processed/yolo_format_test_external \
  --reset-output \
  --skip-dataset-yaml-update

echo "[3/6] Split train/val from full training set and populate test from external real test set"
"${PYTHON_BIN}" scripts/split_dataset.py \
  --input data/processed/yolo_format \
  --external-test-input data/processed/yolo_format_test_external \
  --train-ratio 0.8 \
  --val-ratio 0.2 \
  --seed 42 \
  --reset-splits

echo "[4/6] Train YOLO baseline on full training set"
"${PYTHON_BIN}" scripts/train_yolo.py \
  --data configs/dataset.yaml \
  --model weights/yolo11n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --device "${DEVICE}" \
  --workers 8 \
  --patience 20 \
  --name "${RUN_NAME}"

echo "[5/6] Validate best checkpoint on the real test set"
"${PYTHON_BIN}" scripts/validate_yolo.py \
  --data configs/dataset.yaml \
  --weights "runs/train/${RUN_NAME}/weights/best.pt" \
  --split test \
  --benchmark-device cpu \
  --benchmark-limit 16

echo "[6/6] Export prediction samples and summary"
"${PYTHON_BIN}" scripts/infer_demo.py \
  --weights "runs/train/${RUN_NAME}/weights/best.pt" \
  --source data/processed/yolo_format/images/test \
  --conf 0.25 \
  --limit 10 \
  --output "reports/figures/${RUN_NAME}_predictions"

"${PYTHON_BIN}" scripts/export_summary.py \
  --metrics reports/metrics.json \
  --latency reports/latency.json \
  --mapping reports/class_mapping.json \
  --weights "runs/train/${RUN_NAME}/weights/best.pt" \
  --train-dir "runs/train/${RUN_NAME}" \
  --predictions "reports/figures/${RUN_NAME}_predictions/prediction_manifest.json" \
  --data-root data/processed/yolo_format

echo "Pipeline completed: ${RUN_NAME}"
