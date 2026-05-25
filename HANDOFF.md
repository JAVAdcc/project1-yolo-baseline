# Project 1 YOLO Baseline Handoff

## 1. Repository and Artifacts

GitHub public repository:

- https://github.com/JAVAdcc/project1-yolo-baseline

Hugging Face model repository:

- https://huggingface.co/javadcc/project1-yolo-baseline-yolo11n

Main local workspace:

- `/Users/javadcc/code/ssh/4_3090/yolo_project`

3090 remote workspace:

- `/home/x/workspaces/yekai/wym/project1_yolo_baseline`

3090 SSH access used during the project:

- `ssh x@10.60.18.143`
- Previous Tailscale address: `100.108.12.34`
- Older SSH alias: `3090_4`

The GitHub repository intentionally excludes raw datasets, processed datasets, training runs, pretrained weights, and large generated outputs. The trained YOLO checkpoint is hosted on Hugging Face instead.

## 2. Current Baseline Status

The current YOLO baseline is complete and runnable end to end:

- H5 waveform data can be converted into spectrogram images.
- Frequency-range labels can be converted into YOLO detection labels.
- The dataset can be split into train, val, and test.
- YOLO training, validation, inference, and summary export scripts are implemented.
- A `yolo11n` baseline has been trained and evaluated on an independent real test set.

Important result files:

- Local model: `artifacts/realtest_yolo11n_e50_gpu3/best.pt`
- Local training CSV: `artifacts/realtest_yolo11n_e50_gpu3/results.csv`
- Local metrics: `artifacts/realtest_yolo11n_e50_gpu3/metrics.json`
- Local report draft: `artifacts/realtest_yolo11n_e50_gpu3/notion_report_realtest.md`
- HF checkpoint: `best.pt` in `javadcc/project1-yolo-baseline-yolo11n`
- HF training log: `results.csv` in `javadcc/project1-yolo-baseline-yolo11n`

## 3. Current Dataset Assumption

The baseline was built against the basic SpaceNet H5 files used so far:

- `train_full.h5`
- `test_full.h5`

The parsed H5 structure was:

- `waveforms`: complex waveform array
- `labels`: string array containing frequency-band lists

Example label shape:

```text
[[2407.0, 2417.0], [2408.875, 2409.125], ...]
```

This label format only states where signal energy exists in frequency. It does not expose per-signal protocol names, modulation names, or class IDs. Therefore the current conversion maps all objects into a single YOLO class:

```text
signal -> 0
```

The current baseline is therefore a single-class detector. It detects signal regions, not signal categories.

## 4. Current Label Conversion Logic

The current conversion logic is implemented in:

- `scripts/convert_to_yolo.py`

The current assumption is:

- One frequency band becomes one YOLO bounding box.
- Because no time start/end is available, each box covers the full horizontal image width.
- `x_center = 0.5`
- `width = 1.0`
- `y_center` and `height` are computed from the frequency range.
- Every box uses class ID `0`.

This is a conservative baseline. It is correct for "signal present in this frequency band" detection, but it is not a full semantic signal-recognition solution.

## 5. Current Training Result

Training setup:

- Model: `yolo11n.pt`
- Image size: `640`
- Batch size: `16`
- Epochs requested: `50`
- Early stopping patience: `20`
- Device: single RTX 3090, GPU3

Observed GPU memory:

- Around `2.4 GB` to `2.7 GB` during `yolo11n`, `imgsz=640`, `batch=16` training.

Practical estimate:

- 8 GB GPU should be enough for `yolo11n`, `imgsz=640`, `batch=8`.
- 8 GB GPU will likely also handle `yolo11n`, `imgsz=640`, `batch=16`, but start with `batch=8` for safety.
- Dataset size affects disk usage, conversion time, and total training time, but not single-step GPU memory much.
- More classes add a small YOLO head cost, but batch size and input resolution dominate memory.

Best validation result:

- Best fitness and best `mAP50-95` occurred at epoch 12.
- Precision: `0.87967`
- Recall: `0.63125`
- mAP50: `0.69653`
- mAP50-95: `0.59204`

Independent real test result:

- Precision: `0.59541`
- Recall: `0.32303`
- mAP50: `0.35692`
- mAP50-95: `0.23215`

The validation-to-test gap suggests that the full-width single-class approximation is a useful baseline but not the final modeling approach.

## 6. How To Reproduce Current Single-Class Baseline

Expected remote raw data layout:

```text
data/raw/spacenet/train_full.h5
data/raw/spacenet/test_full.h5
```

Full pipeline script:

```bash
bash scripts/run_real_test_baseline.sh
```

The script performs:

- Convert `train_full.h5` to YOLO staging output.
- Convert `test_full.h5` to a separate external-test YOLO staging output.
- Split training data into train/val and populate test from the external test set.
- Train `yolo11n`.
- Validate on the real test split.
- Export prediction samples and summary.

Manual commands are documented in `README.md`.

## 7. Plan For Multi-Class YOLO Expansion

You found that an advanced SpaceNet dataset may contain signal-category fields. The next milestone should be to migrate that dataset and upgrade the converter from single-class to multi-class.

Target behavior:

- Each signal object should produce one YOLO line.
- The YOLO `class_id` should come from the advanced dataset's signal category.
- `configs/dataset.yaml` should list all class names.
- `reports/class_mapping.json` should map class names to stable integer IDs.
- Validation reports should include per-class metrics.

Recommended implementation steps:

1. Inspect the advanced H5 file before writing conversion code.

   Check the full HDF5 tree, dataset shapes, dtypes, and attrs. Do not assume field names. Look for fields such as `classes`, `types`, `protocols`, `modulations`, `signals`, `annotations`, `metadata`, or structured arrays.

2. Add a new parser path in `scripts/convert_to_yolo.py`.

   The current `parse_band_label()` returns only `(freq_start, freq_end)`. For multi-class data, replace or extend it with a parser that returns structured objects such as:

   ```python
   {
       "class_name": "...",
       "freq_start": ...,
       "freq_end": ...,
       "time_start": optional,
       "time_end": optional,
   }
   ```

3. Preserve the current fallback behavior.

   If the H5 file only has frequency bands and no class field, keep the existing single-class `signal` fallback. This keeps old experiments reproducible.

4. Generate class mapping automatically.

   Scan the training labels first, sort or otherwise stabilize class names, then write:

   ```text
   reports/class_mapping.json
   configs/dataset.yaml
   ```

5. Validate mapping consistency.

   Fail fast if:

   - A label references an unknown class.
   - `dataset.yaml` class count does not match the label files.
   - A YOLO label has class ID outside `[0, nc - 1]`.

6. Run a small smoke test.

   First convert and train on a small subset, for example 32 or 128 samples. Confirm that:

   - Multiple class IDs appear in label files.
   - Ultralytics sees `nc > 1`.
   - Training starts without class-index errors.
   - Prediction legends show multiple class names.

7. Run full training only after smoke checks pass.

   Suggested first 8 GB config:

   ```bash
   python scripts/train_yolo.py \
     --data configs/dataset.yaml \
     --model weights/yolo11n.pt \
     --epochs 50 \
     --imgsz 640 \
     --batch 8 \
     --workers 4
   ```

8. Update summary and report scripts.

   `scripts/export_summary.py` should report:

   - Per-class sample counts
   - Per-class precision and recall if available
   - Class imbalance ratio
   - Zero-detection images
   - Examples from frequent and rare classes

## 8. Multi-Class Risks To Watch

Potential issues in the advanced dataset:

- Class names may be stored at waveform level, not object level.
- One waveform may contain multiple signals with different classes.
- Frequency bands and classes may be separate arrays that need index alignment.
- Some signals may have class names but no time bounds.
- Some labels may be duplicated or overlapping.
- Rare classes may make mAP unstable.

Do not infer classes from frequency bands alone unless the dataset documentation explicitly defines that mapping. If class labels are not object-aligned, document the limitation before training.

## 9. Files Most Likely To Change For Multi-Class

Primary files:

- `scripts/inspect_dataset.py`
- `scripts/convert_to_yolo.py`
- `scripts/split_dataset.py`
- `scripts/export_summary.py`
- `configs/dataset.yaml`
- `README.md`

Likely useful additions:

- `scripts/inspect_h5_schema.py`
- `reports/advanced_dataset_schema.md`
- A small smoke-test H5 subset for parser validation

## 10. Current Public Repo Hygiene

The public GitHub repo should not include:

- Raw H5 datasets
- Processed PNG datasets
- Training run directories
- Large model weights
- Remote wheel caches

The trained checkpoint is already on Hugging Face. Keep GitHub focused on code, configs, reports, and small example artifacts.

