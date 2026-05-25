# Advanced SpaceNet Dataset Schema

Source inspected: `E:\Downloads\SpaceNet.zip`

The outer zip contains two dataset families:

- Basic library: HDF5 files such as `train.h5` and `test.h5`.
- Advanced library: stored inner zip shards such as `train-0-1499.zip` and `test.zip`.

The advanced library is not HDF5. Each inner shard contains paired files:

- `<id>.bin`: float16 interleaved IQ samples, stored as `I0, Q0, I1, Q1, ...`.
- `<id>.json`: object-level annotations for the paired signal file.

Observed JSON shape:

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

YOLO conversion used in `scripts/convert_advanced_to_yolo.py`:

- `class_id` comes directly from `signals[*].class`.
- `x_center` and `width` are computed from `start_time` and `end_time` divided by the inferred waveform duration.
- `y_center` and `height` are computed from the signal frequency range relative to `observation_range`.
- The waveform duration is inferred from complex sample count and sample rate, where sample rate equals observation bandwidth in Hz.
- Spectrograms are resized to 640x640 by default to keep disk usage and training I/O bounded; YOLO labels remain valid because they are normalized.

Class mapping:

```text
0  WIFI 20MHz QPSK
1  WIFI 20MHz 16QAM
2  WIFI 20MHz 64QAM
3  WIFI 40MHz QPSK
4  WIFI 40MHz 16QAM
5  WIFI 40MHz 64QAM
6  BLE LE1M
7  BLE LE2M
8  Zigbee
9  LoRa 250KHZ
10 SRRC QPSK
11 SRRC 16QAM
12 AM
13 FM
```

Smoke check performed locally:

- Converted 32 samples from `train-0-1499.zip`.
- Generated all 14 class IDs in YOLO labels.
- Split into 22 train, 6 val, and 4 test images.
- Started YOLO11n training with `nc=14` on local CUDA.
