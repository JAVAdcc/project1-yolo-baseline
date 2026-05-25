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

## Dataset Split

- train: 4 samples
- val: 1 samples
- test: 2 samples
- Seed used for shuffle: 42
- Reset split directories: True
- External test input: /private/var/folders/3m/zgq8279n64bdpwfbl6zmlpz80000gn/T/tmpb1r61f7_/ext
- Test split populated from external dataset staging directory; training dataset contributed only train/val samples.

## Dataset Split

- train: 7 samples
- val: 3 samples
- test: 3 samples
- Seed used for shuffle: 42
- Reset split directories: True
- External test input: /private/var/folders/3m/zgq8279n64bdpwfbl6zmlpz80000gn/T/tmp3jxuyb10/ext
- Test split populated from external dataset staging directory; train/val ratios were renormalized to consume the full training dataset.

## Advanced YOLO Conversion

- Input: E:\Downloads\SpaceNet.zip archives=['train-0-1499.zip']
- Output: E:\code\project1-yolo-baseline\data\processed\yolo_advanced_smoke
- Sample limit: 32
- Advanced parser reads float16 interleaved IQ .bin files and object-level JSON labels.
- YOLO boxes use JSON time bounds for x and observation frequency range for y.
- Classes: 14
- Converted samples: 32
- Skipped samples/boxes: 0
- Class counts: {0: 5, 1: 9, 2: 7, 3: 2, 4: 2, 5: 3, 6: 15, 7: 7, 8: 43, 9: 27, 10: 7, 11: 7, 12: 6, 13: 7}

## Dataset Split

- train: 22 samples
- val: 6 samples
- test: 4 samples
- Seed used for shuffle: 42
- Reset split directories: True

## Training

- Model: yolo11n.pt
- Classes: 14
- Train images: 22
- Val images: 6
- Epochs: 1
- Image size: 640
- Batch size: 4
- Training output: E:\code\project1-yolo-baseline\runs\detect\runs\train\advanced_smoke_e1
- Best weights: E:\code\project1-yolo-baseline\runs\detect\runs\train\advanced_smoke_e1\weights\best.pt
- Last weights: E:\code\project1-yolo-baseline\runs\detect\runs\train\advanced_smoke_e1\weights\last.pt
- Model info: None
- Train report: E:\code\project1-yolo-baseline\runs\detect\runs\train\advanced_smoke_e1\train_report.json

## Training

- Model: weights\yolo11n.pt
- Classes: 14
- Train images: 22
- Val images: 6
- Epochs: 1
- Image size: 320
- Batch size: 4
- Training output: E:\code\project1-yolo-baseline\runs\train\advanced_smoke_e1_pathcheck
- Best weights: E:\code\project1-yolo-baseline\runs\train\advanced_smoke_e1_pathcheck\weights\best.pt
- Last weights: E:\code\project1-yolo-baseline\runs\train\advanced_smoke_e1_pathcheck\weights\last.pt
- Model info: None
- Train report: E:\code\project1-yolo-baseline\runs\train\advanced_smoke_e1_pathcheck\train_report.json

## Advanced YOLO Conversion

- Input: E:\Downloads\SpaceNet.zip archives=['train-0-1499.zip']
- Output: E:\code\project1-yolo-baseline\data\processed\yolo_advanced_smoke_resized
- Sample limit: 8
- Output image size: 640
- Advanced parser reads float16 interleaved IQ .bin files and object-level JSON labels.
- YOLO boxes use JSON time bounds for x and observation frequency range for y.
- Classes: 14
- Converted samples: 8
- Skipped samples/boxes: 0
- Class counts: {1: 2, 2: 3, 5: 1, 6: 2, 7: 3, 8: 6, 9: 9, 10: 1, 11: 1, 12: 3, 13: 2}

## Advanced YOLO Conversion

- Input: E:\Downloads\SpaceNet.zip archives=['train']
- Output: E:\code\project1-yolo-baseline\data\processed\yolo_advanced
- Sample limit: all
- Output image size: 640
- Advanced parser reads float16 interleaved IQ .bin files and object-level JSON labels.
- YOLO boxes use JSON time bounds for x and observation frequency range for y.
- Classes: 14
- Converted samples: 7500
- Skipped samples/boxes: 129
- Class counts: {0: 1567, 1: 1589, 2: 1591, 3: 716, 4: 694, 5: 727, 6: 3465, 7: 3501, 8: 6934, 9: 6892, 10: 1769, 11: 1698, 12: 1774, 13: 1709}

## Advanced YOLO Conversion

- Input: E:\Downloads\SpaceNet.zip archives=['test.zip']
- Output: E:\code\project1-yolo-baseline\data\processed\yolo_advanced_test
- Sample limit: all
- Output image size: 640
- Advanced parser reads float16 interleaved IQ .bin files and object-level JSON labels.
- YOLO boxes use JSON time bounds for x and observation frequency range for y.
- Classes: 14
- Converted samples: 2500
- Skipped samples/boxes: 56
- Class counts: {0: 852, 1: 915, 2: 933, 3: 390, 4: 378, 5: 410, 6: 1962, 7: 2033, 8: 4042, 9: 3945, 10: 1038, 11: 1025, 12: 1013, 13: 1026}

## Dataset Split

- train: 5833 samples
- val: 1667 samples
- test: 2500 samples
- Seed used for shuffle: 42
- Reset split directories: True
- External test input: E:\code\project1-yolo-baseline\data\processed\yolo_advanced_test
- Test split populated from external dataset staging directory; train/val ratios were renormalized to consume the full training dataset.

## Training

- Model: weights\yolo11n.pt
- Classes: 14
- Train images: 5833
- Val images: 1667
- Epochs: 50
- Image size: 640
- Batch size: 8
- Training output: E:\code\project1-yolo-baseline\runs\train\yolo11n_spacenet_advanced_e50
- Best weights: E:\code\project1-yolo-baseline\runs\train\yolo11n_spacenet_advanced_e50\weights\best.pt
- Last weights: E:\code\project1-yolo-baseline\runs\train\yolo11n_spacenet_advanced_e50\weights\last.pt
- Model info: None
- Train report: E:\code\project1-yolo-baseline\runs\train\yolo11n_spacenet_advanced_e50\train_report.json

## Validation

- Split: test
- mAP50: 0.635885
- mAP50-95: 0.542982
- Precision: 0.645901
- Recall: 0.671336
- Latency benchmark device: 0
- Latency mean (ms): 15.2081
- Latency median (ms): 14.7428

## Validation

- Split: test
- mAP50: 0.635885
- mAP50-95: 0.542982
- Precision: 0.645901
- Recall: 0.671336
- Latency benchmark device: 0
- Latency mean (ms): 17.2154
- Latency median (ms): 16.8399

## Export Summary

- Known classes: 14
- Class mapping: {'WIFI 20MHz QPSK': 0, 'WIFI 20MHz 16QAM': 1, 'WIFI 20MHz 64QAM': 2, 'WIFI 40MHz QPSK': 3, 'WIFI 40MHz 16QAM': 4, 'WIFI 40MHz 64QAM': 5, 'BLE LE1M': 6, 'BLE LE2M': 7, 'Zigbee': 8, 'LoRa 250KHZ': 9, 'SRRC QPSK': 10, 'SRRC 16QAM': 11, 'AM': 12, 'FM': 13}
- Metrics available: True
- mAP50: 0.6358852479582835
- mAP50-95: 0.5429824939918118
- Precision: 0.6459013548405617
- Recall: 0.6713364396069006
- Latency available: True
- Latency mean (ms): 17.2154
- Latency median (ms): 16.8399
- Latency device: 0
- Best model size (MB): 5.22
- Training directory: E:\code\project1-yolo-baseline\runs\train\yolo11n_spacenet_advanced_e50
- Latest training row available: True
- Latest training row: {'epoch': '50', 'time': '2981.81', 'train/box_loss': '0.41693', 'train/cls_loss': '0.59181', 'train/dfl_loss': '0.86613', 'metrics/precision(B)': '0.66402', 'metrics/recall(B)': '0.71243', 'metrics/mAP50(B)': '0.67993', 'metrics/mAP50-95(B)': '0.60239', 'val/box_loss': '0.64918', 'val/cls_loss': '0.81613', 'val/dfl_loss': '0.67972', 'lr/pg0': '1.65688e-05', 'lr/pg1': '1.65688e-05', 'lr/pg2': '1.65688e-05'}
- Train label stats: {'label_files': 5833, 'boxes': 27010, 'class_histogram': {'0': 1252, '1': 1256, '2': 1246, '3': 566, '4': 540, '5': 550, '6': 2680, '7': 2698, '8': 5440, '9': 5360, '10': 1342, '11': 1325, '12': 1413, '13': 1342}, 'width': {'min': 0.006667, 'mean': 0.576373, 'median': 0.566667, 'max': 1.0}, 'height': {'min': 7.5e-05, 'mean': 0.192769, 'median': 0.05, 'max': 1.0}, 'aspect_ratio': {'min': 0.006667, 'mean': 136.557796, 'median': 10.33334, 'max': 13333.333333}}
- Val label stats: {'label_files': 1667, 'boxes': 7616, 'class_histogram': {'0': 315, '1': 333, '2': 345, '3': 150, '4': 154, '5': 177, '6': 785, '7': 803, '8': 1494, '9': 1532, '10': 427, '11': 373, '12': 361, '13': 367}, 'width': {'min': 0.006667, 'mean': 0.573766, 'median': 0.5625, 'max': 1.0}, 'height': {'min': 7.5e-05, 'mean': 0.19315, 'median': 0.05, 'max': 1.0}, 'aspect_ratio': {'min': 0.0125, 'mean': 137.851653, 'median': 10.66668, 'max': 13333.333333}}
- Test label stats: {'label_files': 2500, 'boxes': 19962, 'class_histogram': {'0': 852, '1': 915, '2': 933, '3': 390, '4': 378, '5': 410, '6': 1962, '7': 2033, '8': 4042, '9': 3945, '10': 1038, '11': 1025, '12': 1013, '13': 1026}, 'width': {'min': 0.006667, 'mean': 0.536492, 'median': 0.52, 'max': 1.0}, 'height': {'min': 7.5e-05, 'mean': 0.17486, 'median': 0.04, 'max': 1.0}, 'aspect_ratio': {'min': 0.006667, 'mean': 141.641173, 'median': 11.5, 'max': 13333.333333}}
- Named train class histogram: {'WIFI 20MHz QPSK': 1252, 'WIFI 20MHz 16QAM': 1256, 'WIFI 20MHz 64QAM': 1246, 'WIFI 40MHz QPSK': 566, 'WIFI 40MHz 16QAM': 540, 'WIFI 40MHz 64QAM': 550, 'BLE LE1M': 2680, 'BLE LE2M': 2698, 'Zigbee': 5440, 'LoRa 250KHZ': 5360, 'SRRC QPSK': 1342, 'SRRC 16QAM': 1325, 'AM': 1413, 'FM': 1342}
- Class imbalance ratio (max/min): 10.0741
- Prediction manifest available: False
- Predicted class histogram: unavailable
- Zero-detection images: unavailable
- Current failure-case export is based on the available prediction manifest and should be refreshed after each new full-data training run.
