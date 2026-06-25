# SpaceNet Advanced Multi-Class YOLO Baseline Experiment Report

## 1. Experiment Objective

This experiment upgrades the previous single-class YOLO baseline from basic SpaceNet frequency-band detection to multi-class signal detection on the advanced SpaceNet dataset.

The previous baseline used basic HDF5 files and mapped all objects to one class:

```text
signal -> 0
```

The advanced dataset provides object-level labels with signal class, frequency range, and time range. Therefore this experiment trains a 14-class YOLO detector. Each object is converted into one YOLO bounding box over a spectrogram image.

## 2. Dataset Usage Confirmation

The advanced dataset was used in full for this run.

Source archive:

```text
E:\Downloads\SpaceNet.zip
```

The archive contains two dataset families:

- Basic library: `train.h5`, `test.h5`
- Advanced library: `.bin` signal files paired with `.json` labels inside inner zip shards

This experiment uses the advanced library only. The basic HDF5 dataset was not mixed into this run.

Advanced train shards used:

```text
train-0-1499.zip
train-1500-2999.zip
train-3000-4499.zip
train-4500-5999.zip
train-6000-7499.zip
```

Advanced test shard used:

```text
test.zip
```

Conversion result:

| Source | Samples converted | Notes |
|---|---:|---|
| Advanced train shards | 7,500 | Full train set converted |
| Advanced test shard | 2,500 | Full test set converted |

Some boxes were skipped during conversion:

| Source | Skipped boxes |
|---|---:|
| Train conversion | 129 |
| Test conversion | 56 |

These are invalid, clipped-to-zero, or duplicate boxes after normalization. They are not skipped samples. The final image/label sample counts are complete.

Final YOLO split:

| Split | Images | Labels | Boxes |
|---|---:|---:|---:|
| Train | 5,833 | 5,833 | 27,010 |
| Val | 1,667 | 1,667 | 7,616 |
| Test | 2,500 | 2,500 | 19,962 |

The train/val split is produced from the official advanced train shards. The test split is populated from the official advanced `test.zip`.

## 3. Class Mapping

The advanced dataset defines 14 classes:

| ID | Class |
|---:|---|
| 0 | WIFI 20MHz QPSK |
| 1 | WIFI 20MHz 16QAM |
| 2 | WIFI 20MHz 64QAM |
| 3 | WIFI 40MHz QPSK |
| 4 | WIFI 40MHz 16QAM |
| 5 | WIFI 40MHz 64QAM |
| 6 | BLE LE1M |
| 7 | BLE LE2M |
| 8 | Zigbee |
| 9 | LoRa 250KHZ |
| 10 | SRRC QPSK |
| 11 | SRRC 16QAM |
| 12 | AM |
| 13 | FM |

The mapping is saved at:

```text
reports/class_mapping.json
```

## 4. Data Conversion Method

The conversion script is:

```text
scripts/convert_advanced_to_yolo.py
```

The script reads the outer `SpaceNet.zip` directly and opens the stored inner zip shards by byte range. This avoids fully extracting the 114 GB source archive.

Each advanced sample consists of:

- `<id>.bin`: float16 interleaved IQ samples, stored as `I0, Q0, I1, Q1, ...`
- `<id>.json`: object-level annotations

Observed JSON schema:

```json
{
  "signals": [
    {
      "signal_id": 0,
      "start_frequency": 2417.97385,
      "end_frequency": 2418.02615,
      "start_time": 32.0,
      "end_time": 80.0,
      "class": 9
    }
  ],
  "observation_range": [2401.0, 2431.0]
}
```

YOLO box conversion:

- `class_id` comes directly from `signals[*].class`.
- `x_center` and `width` are computed from `start_time` and `end_time`.
- `y_center` and `height` are computed from `start_frequency` and `end_frequency`.
- The frequency normalization range is `observation_range`.
- The waveform duration is inferred from complex sample count and observation bandwidth.

Spectrogram generation:

- STFT-like FFT spectrogram
- `n_fft = 1024`
- `hop_length = 512`
- Output image resized to `640 x 640`

The resize step is important. Native spectrograms can be extremely wide because the advanced dataset has variable sampling duration. YOLO labels are normalized, so resizing preserves label geometry while keeping disk and training I/O bounded.

## 5. Training Configuration

Environment:

| Item | Value |
|---|---|
| OS | Windows |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| VRAM | 8 GB |
| PyTorch | `2.12.0+cu126` |
| Ultralytics | `8.4.53` |

Training command:

```powershell
.\.venv\Scripts\python.exe scripts\train_yolo.py `
  --data configs\dataset.yaml `
  --model weights\yolo11n.pt `
  --epochs 50 `
  --imgsz 640 `
  --batch 8 `
  --workers 4 `
  --device 0 `
  --name yolo11n_spacenet_advanced_e50 `
  --patience 20
```

Model:

| Item | Value |
|---|---|
| Architecture | YOLO11n |
| Classes | 14 |
| Input size | 640 |
| Batch size | 8 |
| Epochs completed | 50 |
| AMP | Enabled |
| Best checkpoint | `runs/train/yolo11n_spacenet_advanced_e50/weights/best.pt` |
| Best checkpoint size | 5.224 MB |

During training, GPU memory usage stayed around 1.5 to 1.6 GB. No out-of-memory failure occurred.

## 6. Validation Result

Best validation epoch:

| Metric | Value |
|---|---:|
| Best epoch | 50 |
| Precision | 0.66402 |
| Recall | 0.71243 |
| mAP50 | 0.67993 |
| mAP50-95 | 0.60239 |

The validation curve was still improving near the final epoch, so this run is a complete baseline but not necessarily the absolute best attainable model.

## 7. Independent Test Result

The best checkpoint was evaluated on the official advanced test split.

Test command:

```powershell
.\.venv\Scripts\python.exe scripts\validate_yolo.py `
  --data configs\dataset.yaml `
  --weights runs\train\yolo11n_spacenet_advanced_e50\weights\best.pt `
  --split test `
  --benchmark-device 0 `
  --benchmark-limit 32
```

Overall test metrics:

| Metric | Value |
|---|---:|
| Precision | 0.64590 |
| Recall | 0.67134 |
| mAP50 | 0.63589 |
| mAP50-95 | 0.54298 |
| Macro IoU | 0.48185 |
| real-test precision | 0.64590 |

Macro IoU is computed from per-class precision and recall using `IoU = TP / (TP + FP + FN) = P * R / (P + R - P * R)`, then averaged over the 14 classes.

Latency benchmark:

| Metric | Value |
|---|---:|
| Device | GPU 0 |
| Samples | 32 |
| Mean latency | 17.2154 ms/image |
| Median latency | 16.8399 ms/image |
| Min latency | 12.5335 ms/image |
| Max latency | 31.0238 ms/image |

## 8. Per-Class Test Metrics

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| WIFI 20MHz QPSK | 0.4737 | 0.7207 | 0.6083 | 0.5327 |
| WIFI 20MHz 16QAM | 0.4395 | 0.6907 | 0.4866 | 0.4222 |
| WIFI 20MHz 64QAM | 0.5280 | 0.6750 | 0.6629 | 0.5724 |
| WIFI 40MHz QPSK | 0.7099 | 0.6963 | 0.7760 | 0.6892 |
| WIFI 40MHz 16QAM | 0.5328 | 0.7222 | 0.7012 | 0.6144 |
| WIFI 40MHz 64QAM | 0.5628 | 0.7505 | 0.7156 | 0.6328 |
| BLE LE1M | 0.8847 | 0.9052 | 0.9411 | 0.8074 |
| BLE LE2M | 0.8748 | 0.9348 | 0.9513 | 0.8301 |
| Zigbee | 0.9026 | 0.9283 | 0.9582 | 0.8846 |
| LoRa 250KHZ | 0.7163 | 0.2897 | 0.3131 | 0.2064 |
| SRRC QPSK | 0.5473 | 0.8121 | 0.6292 | 0.5107 |
| SRRC 16QAM | 0.5646 | 0.8677 | 0.7200 | 0.6525 |
| AM | 0.6345 | 0.3021 | 0.3002 | 0.1745 |
| FM | 0.6711 | 0.1033 | 0.1386 | 0.0719 |

Strong classes:

- Zigbee: mAP50-95 = 0.8846
- BLE LE2M: mAP50-95 = 0.8301
- BLE LE1M: mAP50-95 = 0.8074

Weak classes:

- FM: mAP50-95 = 0.0719, recall = 0.1033
- AM: mAP50-95 = 0.1745, recall = 0.3021
- LoRa 250KHZ: mAP50-95 = 0.2064, recall = 0.2897

This suggests the model is already learning strong geometric and spectral patterns for dense digital protocols, but narrow or low-contrast analog/LoRa-like patterns are still difficult.

## 9. Visualization Results

Prediction visualizations are saved in:

```text
reports/figures/advanced_predictions/
```

Each visualization is a side-by-side image:

- Left: ground truth boxes
- Right: model prediction boxes with class name and confidence

Generated examples:

| Image | GT boxes | Predicted boxes |
|---|---:|---:|
| `test_0_gt_pred.png` | 6 | 6 |
| `test_1_gt_pred.png` | 10 | 6 |
| `test_10_gt_pred.png` | 8 | 10 |
| `test_100_gt_pred.png` | 6 | 4 |
| `test_1000_gt_pred.png` | 6 | 3 |
| `test_1001_gt_pred.png` | 6 | 3 |
| `test_1002_gt_pred.png` | 10 | 7 |
| `test_1003_gt_pred.png` | 10 | 6 |

Representative visualization files:

```text
reports/figures/advanced_predictions/test_0_gt_pred.png
reports/figures/advanced_predictions/test_1_gt_pred.png
reports/figures/advanced_predictions/test_10_gt_pred.png
reports/figures/advanced_predictions/test_1002_gt_pred.png
```

Qualitative observation:

- High-confidence detections are usually well aligned for Zigbee and BLE.
- Wi-Fi boxes are generally localized, but modulation variants can still be confused.
- Some dense scenes are under-detected, especially when several signals overlap.
- AM/FM/LoRa have low recall on the independent test set; the model often misses them or predicts only the stronger neighboring signals.

## 10. Artifacts

Important files:

| Artifact | Path |
|---|---|
| Converter | `scripts/convert_advanced_to_yolo.py` |
| Training script | `scripts/train_yolo.py` |
| Validation script | `scripts/validate_yolo.py` |
| Visualization script | `scripts/visualize_predictions.py` |
| Dataset config | `configs/dataset.yaml` |
| Class mapping | `reports/class_mapping.json` |
| Training run | `runs/train/yolo11n_spacenet_advanced_e50/` |
| Best weights | `runs/train/yolo11n_spacenet_advanced_e50/weights/best.pt` |
| Test metrics | `reports/metrics.json` |
| Latency metrics | `reports/latency.json` |
| Visualization outputs | `reports/figures/advanced_predictions/` |

Generated dataset directories:

| Dataset | Path | Size |
|---|---|---:|
| Advanced train/val/test YOLO dataset | `data/processed/yolo_advanced` | 2.296 GB |
| Advanced test staging dataset | `data/processed/yolo_advanced_test` | 0.564 GB |

## 11. Conclusion

This run completes the multi-class YOLO baseline on the advanced SpaceNet dataset.

The experiment successfully:

- Used the full advanced train and test datasets.
- Converted object-level JSON labels into 14-class YOLO labels.
- Used both frequency and time bounds instead of the previous full-width frequency-only baseline.
- Trained YOLO11n locally on an 8 GB laptop GPU without relying on the remote 3090.
- Produced a working checkpoint and independent test metrics.
- Exported qualitative prediction visualizations.

The baseline is complete and reproducible. The most important next improvements should focus on weak classes, especially FM, AM, and LoRa 250KHZ. Practical directions include class-aware sampling, longer training, larger YOLO variants, class-specific augmentation, and targeted error analysis on missed narrowband or low-contrast signals.
