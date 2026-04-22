# Project 1 YOLO Baseline

用于课程 Project 1 的 YOLO baseline 工程骨架。当前版本优先完成以下目标：

- 固化工程目录与运行入口
- 准备 Ultralytics YOLO 训练/验证/推理脚本
- 支持手动接入 SpaceNet 原始数据
- 将数据检查、格式转换、训练、评测、汇总拆成独立命令

当前状态：

- YOLO 训练/验证/推理脚本已经跑通
- `train_full.h5` 与真实 `test_full.h5` 已完成接入、转换和切分
- `yolo11n` baseline 已在真实独立测试集上完成一轮正式实验
- 当前公开版本保留代码、配置、总结报告和少量示例图，不包含原始数据与大体量中间产物

## 目录结构

```text
project1_yolo_baseline/
├── README.md
├── requirements.txt
├── configs/
│   ├── dataset.yaml
│   └── train_config.yaml
├── data/
│   ├── raw/
│   │   └── spacenet/
│   └── processed/
│       └── yolo_format/
│           ├── images/
│           │   ├── train/
│           │   ├── val/
│           │   └── test/
│           └── labels/
│               ├── train/
│               ├── val/
│               └── test/
├── reports/
│   ├── figures/
│   ├── baseline_summary.md
│   ├── class_mapping.json
│   └── metrics.json
├── runs/
├── scripts/
│   ├── common.py
│   ├── download_dataset.py
│   ├── inspect_dataset.py
│   ├── convert_to_yolo.py
│   ├── split_dataset.py
│   ├── train_yolo.py
│   ├── validate_yolo.py
│   ├── infer_demo.py
│   └── export_summary.py
└── weights/
```

## 环境准备

推荐使用 Python 3.10 及以上版本。

本地或远端执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 权重准备

默认使用：

- `weights/yolo11n.pt`

可选：

- `weights/yolo11s.pt`

如果 `weights/` 下没有对应文件，训练脚本会先尝试使用本地文件；本地也不存在时，交给 Ultralytics 自行下载。由于当前 3090 服务器公网 DNS 不稳定，建议采用“本地下载后中转”的方式。

## 数据接入

SpaceNet 数据当前建议手动下载后放入：

```text
data/raw/spacenet/
```

当前已经支持以下 H5 结构：

- `waveforms`: `(N, T)` 复数波形数组
- `labels`: `(N,)` 字符串数组，每条记录是形如 `[[f_start, f_end], ...]` 的频段列表

当前采用的最小可行假设：

- 每个频段都视为一个检测目标
- 由于标签中没有时间起止范围，YOLO 框在横向上覆盖整张图
- 当前数据没有显式协议类别，因此先统一映射为单类 `signal`

这不是最终语义，而是让 baseline 先可复现、可运行的保守接法。

## 运行命令

1. 检查原始数据

```bash
python scripts/inspect_dataset.py --input data/raw/spacenet
```

2. 转换为 YOLO 格式

```bash
python scripts/convert_to_yolo.py \
  --input data/raw/spacenet \
  --output data/processed/yolo_format \
  --dataset-file train.h5
```

3. 划分数据集

```bash
python scripts/split_dataset.py \
  --input data/processed/yolo_format \
  --seed 42
```

4. 训练

在执行这一步前，需要先完成两件事：

- `data/processed/yolo_format/images/{train,val,test}` 中已经有真实样本
- `configs/dataset.yaml` 中的 `names` 已被真实类别映射替换，不再是空字典

否则训练脚本会显式报错退出，这是当前设计的预期行为。

```bash
python scripts/train_yolo.py \
  --data configs/dataset.yaml \
  --model weights/yolo11n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

5. 验证

```bash
python scripts/validate_yolo.py \
  --data configs/dataset.yaml \
  --weights runs/train/yolo11n_spacenet_baseline/weights/best.pt \
  --benchmark-device cpu \
  --benchmark-limit 16
```

6. 推理

```bash
python scripts/infer_demo.py \
  --weights runs/train/yolo11n_spacenet_baseline/weights/best.pt \
  --source data/processed/yolo_format/images/test \
  --conf 0.25
```

7. 导出总结

```bash
python scripts/export_summary.py \
  --train-dir runs/train/yolo11n_spacenet_baseline \
  --predictions reports/figures/predictions/prediction_manifest.json
```

## 当前约束

- 课程要求中的 `precision / latency / storage cost` 会纳入最终汇总
- 当前 `validate_yolo.py` 会额外输出 `reports/latency.json`，默认做一轮 CPU 推理延迟统计，GPU 可通过 `--benchmark-device 0` 补测
- 如果后续发现原始标注只有频率范围、没有时间范围，会采用“横向覆盖全图”的 fallback 假设，并在报告里明确标注
- 当前默认接法仍是基于 `train.h5` 的单类检测 baseline，不代表最终协议识别建模方式
- 当前正式 baseline 采用单卡 `GPU3` 训练；如需进一步扩展，可继续尝试多卡 DDP 或更大的模型配置

## 后续计划

- 继续深挖原始 H5 是否存在可恢复的多类别字段
- 补充更系统的误差分析，重点关注零检测样本
- 视结果决定是否补充 `yolo11s` 对比
- 如课程汇报需要，补测 GPU 推理延迟
