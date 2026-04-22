# Baseline Summary

## Status

- Project initialized
- YOLO training framework prepared
- Dataset integration pending real SpaceNet samples

## Notes

- Current implementation leaves dataset parsing extensible on purpose.
- No training metrics are available until dataset conversion is completed.

## YOLO Conversion

- Input directory: /Users/javadcc/Downloads
- Output directory initialized at: /private/tmp/yolo_h5_smoke
- Unsplit staging directories prepared at images/all and labels/all.
- Dataset file: /Users/javadcc/Downloads/train.h5
- Sample limit: 3
- Current H5 parsing assumes labels are frequency-band lists and creates full-width YOLO boxes.
- Current class mapping is single-class detection: signal -> 0.
- Converted samples: 3
- Skipped samples: 0

## YOLO Conversion

- Input directory: /home/x/workspaces/yekai/wym/project1_yolo_baseline/data/raw/spacenet
- Output directory initialized at: /home/x/workspaces/yekai/wym/project1_yolo_baseline/data/processed/yolo_format
- Unsplit staging directories prepared at images/all and labels/all.
- Dataset file: /home/x/workspaces/yekai/wym/project1_yolo_baseline/data/raw/spacenet/train_smoke32.h5
- Sample limit: all
- Current H5 parsing assumes labels are frequency-band lists and creates full-width YOLO boxes.
- Current class mapping is single-class detection: signal -> 0.
- Converted samples: 32
- Skipped samples: 0

## Dataset Split

- train: 22 samples
- val: 6 samples
- test: 4 samples
- Seed used for shuffle: 42

## Training

- Model: weights/yolo11n.pt
- Classes: 1
- Epochs: 1
- Image size: 640
- Batch size: 4
- Training output: runs/train/smoke32_yolo11n_run2
- Model info: None

## Validation

- Split: val
- mAP50: 0.010095
- mAP50-95: 0.003941
- Precision: 0.004444
- Recall: 0.363636

## Inference

- Prediction outputs: /home/x/workspaces/yekai/wym/project1_yolo_baseline/reports/figures/predictions_smoke32
- Requested sample limit: 4
- Actual sample count: 4
- Result objects returned: 4
- Confidence threshold: 0.25

## Validation

- Split: val
- mAP50: 0.010095
- mAP50-95: 0.003941
- Precision: 0.004444
- Recall: 0.363636
- Latency benchmark device: cpu
- Latency mean (ms): 37.1134
- Latency median (ms): 25.8793

## Inference

- Prediction outputs: /home/x/workspaces/yekai/wym/project1_yolo_baseline/reports/figures/predictions_smoke32
- Requested sample limit: 4
- Actual sample count: 4
- Result objects returned: 4
- Confidence threshold: 0.25
- Prediction manifest: reports/figures/predictions_smoke32/prediction_manifest.json

## Export Summary

- Known classes: 1
- Class mapping: {'signal': 0}
- Metrics available: True
- mAP50: 0.010094562475526384
- mAP50-95: 0.003941394237914916
- Precision: 0.0044444444444444444
- Recall: 0.36363636363636365
- Latency available: True
- Latency mean (ms): 37.1134
- Latency median (ms): 25.8793
- Latency device: cpu
- Best model size (MB): 5.18
- Training directory: /home/x/workspaces/yekai/wym/project1_yolo_baseline/runs/train/smoke32_yolo11n_run2
- Latest training row available: True
- Latest training row: {'                  epoch': '                      1', '         train/box_loss': '                 3.0478', '         train/cls_loss': '                 3.8795', '         train/dfl_loss': '                 2.6833', '   metrics/precision(B)': '                0.00389', '      metrics/recall(B)': '                0.31818', '       metrics/mAP50(B)': '                0.00857', '    metrics/mAP50-95(B)': '                0.00328', '           val/box_loss': '                 3.0281', '           val/cls_loss': '                 2.9143', '           val/dfl_loss': '                 2.6611', '                 lr/pg0': '                 0.0001', '                 lr/pg1': '                 0.0001', '                 lr/pg2': '                 0.0001'}
- Train label stats: {'label_files': 22, 'boxes': 61, 'class_histogram': {'0': 61}, 'width': {'min': 1.0, 'mean': 1.0, 'median': 1.0, 'max': 1.0}, 'height': {'min': 0.002465, 'mean': 0.095178, 'median': 0.024653, 'max': 0.493066}, 'aspect_ratio': {'min': 2.028126, 'mean': 116.598018, 'median': 40.563015, 'max': 405.679513}}
- Val label stats: {'label_files': 6, 'boxes': 22, 'class_histogram': {'0': 22}, 'width': {'min': 1.0, 'mean': 1.0, 'median': 1.0, 'max': 1.0}, 'height': {'min': 0.002465, 'mean': 0.119317, 'median': 0.024653, 'max': 0.493066}, 'aspect_ratio': {'min': 2.028126, 'mean': 136.428734, 'median': 40.563015, 'max': 405.679513}}
- Test label stats: {'label_files': 4, 'boxes': 12, 'class_histogram': {'0': 12}, 'width': {'min': 1.0, 'mean': 1.0, 'median': 1.0, 'max': 1.0}, 'height': {'min': 0.003082, 'mean': 0.121726, 'median': 0.123267, 'max': 0.493066}, 'aspect_ratio': {'min': 2.028126, 'mean': 70.471614, 'median': 40.563015, 'max': 324.464633}}
- Named train class histogram: {'signal': 61}
- Class imbalance ratio (max/min): 1.0
- Prediction manifest available: True
- Predicted class histogram: {}
- Zero-detection images: ['data/processed/yolo_format/images/test/sample_000000.png', 'data/processed/yolo_format/images/test/sample_000007.png', 'data/processed/yolo_format/images/test/sample_000008.png', 'data/processed/yolo_format/images/test/sample_000023.png']
- Failure-case export remains provisional until full train.h5 reaches the remote machine.

## Dataset Inspection

- Input directory: /home/x/workspaces/yekai/wym/project1_yolo_baseline/data/raw/spacenet
- Total files: 2
- Image files: 0
- Label-like files: 0
- Suffix histogram: {'.h5': 2}
- Sample label preview: N/A
- If the dataset only provides frequency ranges without time ranges, conversion will use a fallback full-width box strategy.
- H5 file: train_smoke32.h5
- H5 keys: ['labels', 'waveforms']
- waveforms shape: (32, 100000), dtype: complex64
- sample label[0]: [[2427.0, 2447.0], [2422.0, 2432.0], [2422.0, 2432.0], [2427.0, 2447.0]]
- sample band count: 4

## YOLO Conversion

- Input directory: /home/x/workspaces/yekai/wym/project1_yolo_baseline/data/raw/spacenet
- Output directory initialized at: /home/x/workspaces/yekai/wym/project1_yolo_baseline/data/processed/yolo_format_probe
- Unsplit staging directories prepared at images/all and labels/all.
- Dataset file: /home/x/workspaces/yekai/wym/project1_yolo_baseline/data/raw/spacenet/train_full.h5
- Sample limit: 64
- Current H5 parsing assumes labels are frequency-band lists and creates full-width YOLO boxes.
- Current class mapping is single-class detection: signal -> 0.
- Converted samples: 64
- Resumed samples skipped because outputs already existed: 0
- Skipped samples: 0

## Dataset Split

- train: 44 samples
- val: 12 samples
- test: 8 samples
- Seed used for shuffle: 42
- Reset split directories: True

## YOLO Conversion

- Input directory: /home/x/workspaces/yekai/wym/project1_yolo_baseline/data/raw/spacenet
- Output directory initialized at: /home/x/workspaces/yekai/wym/project1_yolo_baseline/data/processed/yolo_format
- Unsplit staging directories prepared at images/all and labels/all.
- Dataset file: /home/x/workspaces/yekai/wym/project1_yolo_baseline/data/raw/spacenet/train_full.h5
- Sample limit: all
- Current H5 parsing assumes labels are frequency-band lists and creates full-width YOLO boxes.
- Current class mapping is single-class detection: signal -> 0.
- Converted samples: 24000
- Resumed samples skipped because outputs already existed: 0
- Skipped samples: 0

## Dataset Split

- train: 16800 samples
- val: 4800 samples
- test: 2400 samples
- Seed used for shuffle: 42
- Reset split directories: True

## Training

- Model: weights/yolo11n.pt
- Classes: 1
- Train images: 16800
- Val images: 4800
- Epochs: 1
- Image size: 640
- Batch size: 16
- Training output: runs/train/full_signal_yolo11n_e1_gpu3
- Best weights: runs/train/full_signal_yolo11n_e1_gpu3/weights/best.pt
- Last weights: runs/train/full_signal_yolo11n_e1_gpu3/weights/last.pt
- Model info: None
- Train report: runs/train/full_signal_yolo11n_e1_gpu3/train_report.json

## Validation

- Split: test
- mAP50: 0.428864
- mAP50-95: 0.284898
- Precision: 0.650410
- Recall: 0.377708
- Latency benchmark device: cpu
- Latency mean (ms): 31.8765
- Latency median (ms): 28.648

## Inference

- Prediction outputs: /home/x/workspaces/yekai/wym/project1_yolo_baseline/reports/figures/predictions_full_e1
- Requested sample limit: 10
- Actual sample count: 10
- Result objects returned: 10
- Confidence threshold: 0.25
- Prediction manifest: reports/figures/predictions_full_e1/prediction_manifest.json

## Export Summary

- Known classes: 1
- Class mapping: {'signal': 0}
- Metrics available: True
- mAP50: 0.428863827599024
- mAP50-95: 0.2848980394207282
- Precision: 0.650409706275379
- Recall: 0.3777075227561407
- Latency available: True
- Latency mean (ms): 31.8765
- Latency median (ms): 28.648
- Latency device: cpu
- Best model size (MB): 5.18
- Training directory: /home/x/workspaces/yekai/wym/project1_yolo_baseline/runs/train/full_signal_yolo11n_e1_gpu3
- Latest training row available: True
- Latest training row: {'                  epoch': '                      1', '         train/box_loss': '                 1.4037', '         train/cls_loss': '                 1.9508', '         train/dfl_loss': '                 1.2249', '   metrics/precision(B)': '                0.65016', '      metrics/recall(B)': '                0.38067', '       metrics/mAP50(B)': '                0.42868', '    metrics/mAP50-95(B)': '                0.28678', '           val/box_loss': '                 1.3181', '           val/cls_loss': '                 1.5714', '           val/dfl_loss': '                0.99469', '                 lr/pg0': '             0.00066603', '                 lr/pg1': '             0.00066603', '                 lr/pg2': '             0.00066603'}
- Train label stats: {'label_files': 16800, 'boxes': 49968, 'class_histogram': {'0': 49968}, 'width': {'min': 1.0, 'mean': 1.0, 'median': 1.0, 'max': 1.0}, 'height': {'min': 7.4e-05, 'mean': 0.109014, 'median': 0.02454, 'max': 0.490798}, 'aspect_ratio': {'min': 2.037498, 'mean': 121.247408, 'median': 40.749796, 'max': 13513.513514}}
- Val label stats: {'label_files': 4800, 'boxes': 14238, 'class_histogram': {'0': 14238}, 'width': {'min': 1.0, 'mean': 1.0, 'median': 1.0, 'max': 1.0}, 'height': {'min': 7.4e-05, 'mean': 0.109762, 'median': 0.02454, 'max': 0.490798}, 'aspect_ratio': {'min': 2.037498, 'mean': 124.557891, 'median': 40.749796, 'max': 13513.513514}}
- Test label stats: {'label_files': 2400, 'boxes': 7175, 'class_histogram': {'0': 7175}, 'width': {'min': 1.0, 'mean': 1.0, 'median': 1.0, 'max': 1.0}, 'height': {'min': 7.4e-05, 'mean': 0.111215, 'median': 0.02454, 'max': 0.490798}, 'aspect_ratio': {'min': 2.037498, 'mean': 124.769207, 'median': 40.749796, 'max': 13513.513514}}
- Named train class histogram: {'signal': 49968}
- Class imbalance ratio (max/min): 1.0
- Prediction manifest available: True
- Predicted class histogram: {'signal': 19}
- Zero-detection images: ['data/processed/yolo_format/images/test/sample_000092.png']
- Failure-case export remains provisional until full train.h5 reaches the remote machine.

## Validation

- Split: test
- mAP50: 0.428864
- mAP50-95: 0.284898
- Precision: 0.650410
- Recall: 0.377708
- Latency benchmark device: cpu
- Latency mean (ms): 32.0043
- Latency median (ms): 26.8587

## Training

- Model: weights/yolo11n.pt
- Classes: 1
- Train images: 16800
- Val images: 4800
- Epochs: 50
- Image size: 640
- Batch size: 16
- Training output: runs/train/full_signal_yolo11n_e50_gpu3
- Best weights: runs/train/full_signal_yolo11n_e50_gpu3/weights/best.pt
- Last weights: runs/train/full_signal_yolo11n_e50_gpu3/weights/last.pt
- Model info: None
- Train report: runs/train/full_signal_yolo11n_e50_gpu3/train_report.json

## Validation

- Split: test
- mAP50: 0.710959
- mAP50-95: 0.614534
- Precision: 0.890763
- Recall: 0.634968
- Latency benchmark device: cpu
- Latency mean (ms): 27.4203
- Latency median (ms): 27.6562

## Inference

- Prediction outputs: /home/x/workspaces/yekai/wym/project1_yolo_baseline/reports/figures/predictions_full_e50_best
- Requested sample limit: 10
- Actual sample count: 10
- Result objects returned: 10
- Confidence threshold: 0.25
- Prediction manifest: reports/figures/predictions_full_e50_best/prediction_manifest.json

## Export Summary

- Known classes: 1
- Class mapping: {'signal': 0}
- Metrics available: True
- mAP50: 0.710959088404169
- mAP50-95: 0.6145341857718785
- Precision: 0.8907632905344315
- Recall: 0.634967596506058
- Latency available: True
- Latency mean (ms): 27.4203
- Latency median (ms): 27.6562
- Latency device: cpu
- Best model size (MB): 5.19
- Training directory: /home/x/workspaces/yekai/wym/project1_yolo_baseline/runs/train/full_signal_yolo11n_e50_gpu3
- Latest training row available: True
- Latest training row: {'                  epoch': '                     38', '         train/box_loss': '                0.42886', '         train/cls_loss': '                0.50328', '         train/dfl_loss': '                0.82404', '   metrics/precision(B)': '                0.44503', '      metrics/recall(B)': '                0.39797', '       metrics/mAP50(B)': '                 0.3477', '    metrics/mAP50-95(B)': '                0.19009', '           val/box_loss': '                 1.7342', '           val/cls_loss': '                 1.7491', '           val/dfl_loss': '                 1.2204', '                 lr/pg0': '               0.002674', '                 lr/pg1': '               0.002674', '                 lr/pg2': '               0.002674'}
- Train label stats: {'label_files': 16800, 'boxes': 49467, 'class_histogram': {'0': 49467}, 'width': {'min': 1.0, 'mean': 1.0, 'median': 1.0, 'max': 1.0}, 'height': {'min': 7.4e-05, 'mean': 0.107886, 'median': 0.02454, 'max': 0.490798}, 'aspect_ratio': {'min': 2.037498, 'mean': 122.196733, 'median': 40.749796, 'max': 13513.513514}}
- Val label stats: {'label_files': 4800, 'boxes': 14104, 'class_histogram': {'0': 14104}, 'width': {'min': 1.0, 'mean': 1.0, 'median': 1.0, 'max': 1.0}, 'height': {'min': 7.4e-05, 'mean': 0.108669, 'median': 0.02454, 'max': 0.490798}, 'aspect_ratio': {'min': 2.037498, 'mean': 125.4012, 'median': 40.749796, 'max': 13513.513514}}
- Test label stats: {'label_files': 2400, 'boxes': 7098, 'class_histogram': {'0': 7098}, 'width': {'min': 1.0, 'mean': 1.0, 'median': 1.0, 'max': 1.0}, 'height': {'min': 7.4e-05, 'mean': 0.10967, 'median': 0.02454, 'max': 0.490798}, 'aspect_ratio': {'min': 2.037498, 'mean': 125.808655, 'median': 40.749796, 'max': 13513.513514}}
- Named train class histogram: {'signal': 49467}
- Class imbalance ratio (max/min): 1.0
- Prediction manifest available: False
- Predicted class histogram: unavailable
- Zero-detection images: unavailable
- Failure-case export remains provisional until full train.h5 reaches the remote machine.
