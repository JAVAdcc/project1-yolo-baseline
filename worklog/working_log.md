# Working Log

- Objective: 为课程 Project 1 搭建可复现 YOLO baseline，当前阶段先完成项目理解、方案规划，并准备把数据集/预训练模型拉到 4x3090 服务器。
- Completion criteria: 1) 读完项目说明和现有指引；2) 形成分阶段方案；3) 明确当前阶段的数据与权重获取路径、远端准备步骤、风险与待确认项。
- Assumptions: 当前仓库为空；数据集名称 SpaceNet 需进一步确认具体来源；服务器 SSH 信息位于 /Users/javadcc/code/ssh/4_3090/ssh.md。
- Open risks: 数据集真实下载源未确认；远端磁盘/网络/conda 环境未知；课程 PDF 可能对 baseline 评价指标有特殊要求。
- Known debt: 暂无。
- Completed checkpoints: 初始化工作日志。
- Failed attempts: 暂无。
- Current status: 正在读取课程说明与远端信息。
- Next action: 解析 PDF，读取 ssh.md，提炼正式需求与实施计划。

- Completed checkpoints: 1) 读取课程 PDF；2) 验证 SSH 可连；3) 确认远端为 4x3090；4) 确认个人工作目录 `/home/x/workspaces/yekai/wym` 可用且有 2.6T 可用空间；5) 发现远端公网 DNS 解析失败，不能依赖服务器直连 GitHub/HuggingFace。
- Current status: 正在输出执行方案，并等待 reviewer 对方案风险做一次审查。
- Next action: 向用户汇报课程正式要求与当前阶段建议执行顺序；同时明确真正 blocker 是 SpaceNet 数据源/标注格式尚未拿到。

- Completed checkpoints: 6) 下载本地权重 `weights/yolo11n.pt` 与 `weights/yolo11s.pt`；7) 初始化工程骨架与脚本入口；8) 同步工程和权重到远端 `/home/x/workspaces/yekai/wym/project1_yolo_baseline`；9) 本地 compileall 通过；10) 本地和远端 `train_yolo.py --help` 可正常运行。
- Known debt: 真实 SpaceNet 解析器尚未实现；远端尚未安装训练依赖，因为公网不可用。
- Current status: 等待 reviewer 对 milestone 做审查。
- Next action: 根据 reviewer 反馈修正，再向用户汇报当前完成度与下一步。

- Completed checkpoints: 11) 确认 `train.h5` 结构为 `waveforms(24000,100000) complex64` + `labels(24000,)`；12) 确认标签是频段列表字符串；13) 统计得到总体频率范围 `[2400.5, 2482.0]`，总框数 71381，单样本 2-4 个框；14) 将 H5->spectrogram->YOLO 的真实转换逻辑接入 `convert_to_yolo.py`；15) 本地 smoke test 成功生成 3 个样本的 PNG 和 YOLO 标签。
- Assumptions: 当前数据只有频率范围，没有时间边界，也没有协议类别，因此先做单类 `signal` 检测并使用全宽框。
- Open risks: 18GB H5 传输仍在进行；远端可复用训练环境尚未最终选定。

- Completed checkpoints: 16) 远端 `/home/x/miniconda3/envs/evo-rl` 环境补齐 `h5py`、`ultralytics`、`seaborn`；17) 将 `train_smoke32.h5` 传到远端；18) 在远端完成 32 样本 H5->spectrogram->YOLO 转换并切分为 train22/val6/test4；19) 远端 `yolo11n` 1 epoch smoke training 成功完成并生成 `best.pt`/`last.pt`；20) 远端 `validate_yolo.py` 和 `infer_demo.py` 在 smoke 数据上成功运行。
- Failed attempts: 1) 多次并行触发 convert/split/train 造成竞态；2) 本地 smoke 输出曾覆盖主 `dataset.yaml`；3) 使用 `rsync --delete` 同步代码时误删远端 `data/processed`；以上都已定位并修正。
- Open risks: 完整 `train.h5` 传到远端的链路过慢，实测约 0.13 MB/s，18GB 预计约 38 小时。

- Completed checkpoints: 21) 为 `validate_yolo.py` 增加延迟基准测试并输出 `reports/latency.json`；22) 为 `export_summary.py` 增加训练指标、模型大小、标签统计、类别不平衡提示、零检测样本索引汇总；23) 为 `infer_demo.py` 输出 `zero_detection_images` 便于后续误差分析。
- Assumptions: 在完整数据尚未到远端前，延迟和误差分析先基于 smoke 数据链路验证脚本正确性，不将其误报为最终 baseline 结果。
- Known debt: 目前延迟统计仍是逐张推理 wall-clock 时间，尚未拆分 warmup / steady-state，也未单独记录 GPU 同步后的纯前向延迟。
- Current status: 正在将增强后的训练汇总脚本同步到远端并做 smoke 级验证。
- Next action: 远端执行 validate/infer/export 三段命令，确认 `latency.json` 与增强版 `baseline_summary.md` 正常生成。

- Completed checkpoints: 24) 2026-04-04 本地 `train.h5` 与远端 `train_full.h5` 文件大小一致；25) 远端成功用 `h5py` 打开并验证 `labels`/`waveforms` 结构；26) 抽样比对索引 0、12000、23999 的标签与波形统计，结果与本地一致。
- Current status: 完整训练数据已在远端就位并通过结构性校验。
- Next action: 在远端基于 `train_full.h5` 执行完整 `inspect -> convert -> split -> train` 流程。

- Completed checkpoints: 27) 完整 `train_full.h5` 已转换为 YOLO 图像与标签；28) 正式数据切分完成，train/val/test 分别为 16800/4800/2400；29) 正式处理目录体积约 7.1G。
- Current status: 正在检查 GPU 状态并准备基于完整数据做正式训练的短程 sanity run。
- Next action: 在远端启动完整数据训练，获取单 epoch 耗时、显存占用和首轮指标。

- Completed checkpoints: 30) 在 GPU3 上完成完整数据 1 epoch 正式训练，单轮总耗时约 3 分 28 秒；31) 完整数据 test split 验证得到 mAP50=0.4289、mAP50-95=0.2849、Precision=0.6504、Recall=0.3777；32) 推理样例与 summary 已导出；33) 清理重复标签 464 行，复验后 duplicate label 告警消失。
- Known debt: 当前正式训练只跑了 1 epoch，尚不是课程最终 baseline；还需要决定是否用单卡继续 50 epoch，或切到多卡 DDP。
- Current status: 正式数据已经接入 YOLO，完整数据训练/验证/推理/汇总链路均已跑通。
- Next action: 按用户意图选择继续长跑 50 epoch baseline，或先做多卡训练配置调整。

- Completed checkpoints: 34) 已启动 50 epoch 正式 baseline `full_signal_yolo11n_e50_gpu3`，训练 PID 为 `2794670`；35) 已确认 epoch 1 完整落盘，`results.csv` 写出 `mAP50=0.39261 / mAP50-95=0.24982 / Precision=0.65029 / Recall=0.33281`；36) `best.pt` 与 `last.pt` 已生成，文件大小均约 11MB；37) 已确认 epoch 2 继续正常训练中，实时 loss 约为 `box=1.154 / cls=1.593 / dfl=1.055`，GPU3 显存约 `2.45GB`。
- Completed checkpoints: 38) 已确认 epoch 2 完整落盘，`results.csv` 新增第二行，当前 `mAP50=0.37375 / mAP50-95=0.25031 / Precision=0.52249 / Recall=0.34940`；39) 训练已推进到 `3/50`，GPU3 显存约 `2.70GB`、利用率约 `54%`；40) 修正 `scripts/export_summary.py` 中“完整数据尚未到远端”的过期描述并同步到远端；41) 在 `scripts/train_yolo.py` 中补齐 batch size OOM 自动回退逻辑（`16 -> 8`）；42) 在 `scripts/validate_yolo.py` 中补齐 GPU latency warmup 与 `cuda synchronize`，避免后续课程汇报引用到偏差过大的延迟结果。
- Open risks: 1) 训练脚本当前仍保留 Ultralytics 离线 AMP 检查告警，会尝试下载 `yolov8n.pt` 但不会中断训练；2) 训练脚本尚未实现 batch size 自动回退；3) 当前 log 主要由 tqdm 进度条组成，后续需要整理更易汇报的训练摘要。
- Known debt: AGENTS 期望在大里程碑后调用 reviewer sub-agent，但当前工具策略要求只有用户显式要求时才能 `spawn_agent`，因此本轮未实际调用，只能改为人工记录风险与证据。
- Current status: 50 epoch 正式 baseline 正在远端持续运行，已跨过首个稳定 checkpoint，当前没有发现发散、NaN、checkpoint 缺失或重复标签告警复发。
- Next action: 继续基于 `results.csv`、日志、GPU 和 checkpoint 做真实监控；同时把过期 README/summary 描述同步到远端，避免后续导出结果时带错上下文。

- Objective: 接入用户新提供的真实测试集 `/Users/javadcc/Downloads/test.h5`，在远端重建“训练集用于 train/val、真实 test.h5 用于 test”的 baseline，并输出真实 test baseline 指标与推理样例。
- Completion criteria: 1) `test.h5` 传到远端并通过结构校验；2) 数据流程改成外部真实 test split；3) 重新训练一个 baseline；4) 在真实 test 集上输出 metrics、summary 和预测图。
- Assumptions: `test.h5` 与 `train_full.h5` 的标签语义一致，仍是频段列表字符串，因此可沿用当前单类 `signal` + 全宽 bbox 的保守接法。
- Open risks: 1) 重新从 `train_full.h5` 转图会覆盖当前 `data/processed/yolo_format`；2) 若继续保留 `patience=20`，训练可能再次在 50 epoch 前 early stop；3) 4.5GB 测试集传输耗时依赖当前本地到远端链路。
- Current status: 已确认本地 `test.h5` 结构为 `labels(6000,)` + `waveforms(6000,100000)`，并已开始 rsync 到远端；正在补 `split_dataset.py` 以支持外部真实测试集。
- Next action: 完成脚本修改与验证；待 `test.h5` 传输完成后，在远端执行 `convert(train) -> convert(test) -> split(train+val + external test) -> train -> validate(test) -> infer -> export`。

- Completed checkpoints: 43) 已确认本地真实测试集 `/Users/javadcc/Downloads/test.h5` 大小约 4.5G，结构为 `labels(6000,)` + `waveforms(6000,100000)`；44) 已为 `split_dataset.py` 增加 `--external-test-input`，支持“训练集仅切 train/val，外部数据集复制为 test split”；45) 已为 `convert_to_yolo.py` 增加 `--skip-dataset-yaml-update`，避免转换独立真实测试集时污染主 `configs/dataset.yaml`；46) 上述脚本均已在本地与远端通过 `--help` 级验证。
- Failed attempts: 4) 单流 rsync 传 `test.h5` 实测稳定吞吐约 `2.2 MB/s`，35 分钟量级；5) 尝试更换 SSH cipher/关闭压缩后无明显提升；6) 尝试四路并行 64MB 探测传输后，两路只到 `43-46MB`，总吞吐未明显改善；7) 尝试 `nohup rsync` 后本地进程未稳定存活，远端文件停在约 `197MB`。
- Open risks: 当前最主要 blocker 是本地到 3090 的上传链路慢且不稳定，导致真实测试集尚未完整到远端，因此无法继续完成真实 test baseline 的训练与评测。
- Current status: 代码层面的真实 test 接入已经准备完毕；远端资源空闲、磁盘充足，但 `test_full.h5` 尚未完整到位。
- Next action: 重新选择更稳的传输方式把 `test.h5` 完整送到远端，之后按既定流水线继续完成真实 test baseline。

- Completed checkpoints: 47) 真实测试集 `test_full.h5` 已完整上传到远端并通过 `h5py` 结构校验；48) 远端 `scripts/run_real_test_baseline.sh` 已启动真实 test 全流水线；49) 2026-04-08 19:56 CST 复核到第 1 步训练集转换仍在推进，`data/processed/yolo_format/images/all` 与 `labels/all` 已各生成 14004 个文件，日志更新时间为 0 秒前。
- Completed checkpoints: 50) 2026-04-08 20:01 CST 确认第 1 步训练集转换完成，`data/processed/yolo_format/images/all` 达到 24000；51) 流水线已自动切换到第 2 步真实测试集转换，当前进程为 `convert_to_yolo.py --dataset-file test_full.h5 --output data/processed/yolo_format_test_external --skip-dataset-yaml-update`。
- Completed checkpoints: 52) 2026-04-08 20:04 CST 确认真实测试集转换完成并进入训练阶段；53) split 后主数据集规模为 `train=19200 / val=4800 / test=6000`，其中 `test` 来自外部 `test_full.h5`；54) 2026-04-08 20:08 CST 确认训练已产出首个 epoch 结果和 checkpoint，`best.pt/last.pt` 均已生成；55) 当前第 2 个 epoch 正常推进，GPU3 显存约 `2.7GB`、利用率约 `47%-52%`。
- Open risks: 1) H5 转频谱图阶段仍是 CPU 密集步骤，整体耗时较长；2) 训练尚未起跑，最终是否出现早停、OOM 或指标异常仍需继续观察。
- Open risks: 1) `patience=20` 仍可能在 50 epoch 前触发 early stopping；2) 当前单类 `signal` 假设仍是工程性近似，最终指标仅代表该保守接法的 baseline 上限下界之一。
- Current status: 正在按分钟级主动轮询远端进程、日志、产物数量和 GPU；当前无中断迹象。
- Next action: 继续盯完整训练直到结束；随后自动跟进真实 test 上的 validate、预测图导出和 summary 产物检查。

- Completed checkpoints: 56) 2026-04-08 21:14 CST 真实 test baseline 全流水线完成，包含训练、真实 test 验证、预测图导出与 summary 导出；57) Ultralytics 触发 `EarlyStopping(patience=20)`，训练实际在 `epoch 32` 结束；58) 复核 `results.csv` 后确认最佳 fitness 与最佳 `mAP50-95` 出现在 `epoch 12`，对应 `P=0.87967 / R=0.63125 / mAP50=0.69653 / mAP50-95=0.59204`，最佳 `mAP50` 出现在 `epoch 14`，为 `0.70082`；59) 使用 `best.pt` 在真实 `test`(6000 张图, 29372 个实例) 上评测得到 `Precision=0.59541 / Recall=0.32303 / mAP50=0.35692 / mAP50-95=0.23215`；60) 已将 `baseline_summary.md`、`metrics.json`、`latency.json`、`results.png`、`confusion_matrix.png` 与 10 张真实 test 预测图同步到本地 `artifacts/realtest_yolo11n_e50_gpu3/`。
- Open risks: 1) 训练阶段最佳 `val` 指标与真实 `test` 指标差距较大，说明当前“单类 + 全宽框”接法的泛化有限；2) `reports/metrics.json` 和 `reports/baseline_summary.md` 仍是通用文件名，后续并行实验时应改成 run-specific 命名以避免覆盖；3) latency 当前记录的是 CPU 推理，若课程汇报需要 GPU latency，还需额外在 CUDA 模式下重测。
- Current status: 本轮真实 test baseline 已完成，关键产物与图像已回收到本地。
- Next action: 向用户汇报结果；若需要，继续做误差分析、改进实验，或把本次结果整理进 Notion。
