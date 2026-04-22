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
