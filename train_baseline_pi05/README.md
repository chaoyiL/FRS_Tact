# 独立 Pi0.5 触觉基线训练

本项目约定只新增两个相互独立的项目目录：训练目录 `train_baseline_pi05/` 与由部署交接方维护的部署目录；本目录不导入、也不修改原有训练或部署代码。它不是 FRS 方案：没有反向/前向 FRS 成对流程，也没有残差或 `x_base`。

模型固定为直接全动作解码器：冻结 Pi0.5 产生 `[B,50,20]` 粗动作，冻结触觉 ResNet18 产生 `[B,4,512]` 当前帧 token；两个 Transformer decoder layer 直接预测完整 20 维动作（包括第 9、19 维夹爪）。四个传感器顺序必须严格为：

1. `observation.images.tactile_left_0`
2. `observation.images.tactile_right_0`
3. `observation.images.tactile_left_1`
4. `observation.images.tactile_right_1`

## 配置与隔离环境

默认配置为 `configs/train_baseline_pi05.yaml`：数据集 `KaiyueChen/two_tubes_03`、Pi0.5 checkpoint、norm assets、0824 tactile encoder 与三个 `/workspace/baseline_pi05/...` cache/output 位置都在 YAML 中，部署服务器应先按本机绝对路径修改这些字段。不要修改源码来替换路径。

环境只允许是本目录的 `.venv`，Python 3.12：

```bash
bash train_baseline_pi05/scripts/setup_env.sh
```

启动脚本只使用 `train_baseline_pi05/.venv/bin/python`，并固定 `PYTHONSAFEPATH=1`、项目本地 vendored runtime 优先的 `PYTHONPATH`、`PYTHONUNBUFFERED=1` 与 `XLA_PYTHON_CLIENT_PREALLOCATE=false`：

```bash
bash train_baseline_pi05/scripts/start_train.sh --help
```

## 运行顺序

管线始终以三个独立进程依次执行 `tactile_cache`、`prepare_action_cache`、`train`。两个 JAX cache producer 必须完全退出后才启动 PyTorch 训练。

先进行只读、离线的路径检查；它不会建立目录、下载、读取视频或载入大 checkpoint：

```bash
bash train_baseline_pi05/scripts/start_train.sh \
  --config train_baseline_pi05/configs/train_baseline_pi05.yaml --check
```

小型 cache smoke 只运行两个 producer，并明确限制样本数：

```bash
train_baseline_pi05/.venv/bin/python -m train_baseline_pi05.tactile_cache \
  --config train_baseline_pi05/configs/train_baseline_pi05.yaml --max-samples 128
train_baseline_pi05/.venv/bin/python -m train_baseline_pi05.prepare_action_cache \
  --config train_baseline_pi05/configs/train_baseline_pi05.yaml --max-samples 128
```

使用完整、对齐的缓存做一步训练 smoke（`--max-steps` 不改写 YAML）：

```bash
bash train_baseline_pi05/scripts/start_train.sh \
  --config train_baseline_pi05/configs/train_baseline_pi05.yaml --max-steps 1
```

一 epoch 验证时把 YAML 的 `decoder.epochs` 临时设为 `1`，然后运行：

```bash
bash train_baseline_pi05/scripts/start_train.sh \
  --config train_baseline_pi05/configs/train_baseline_pi05.yaml
```

正式训练前恢复所需 `decoder.epochs`（默认 50），使用相同命令。恢复训练时把 YAML 的 `decoder.resume` 设为 `true`，保留现有 `last.pt`，再运行同一条正式命令。评估独立于三阶段训练管线：

```bash
train_baseline_pi05/.venv/bin/python -m train_baseline_pi05.evaluate \
  --config train_baseline_pi05/configs/train_baseline_pi05.yaml \
  --checkpoint /workspace/baseline_pi05/decoder/best.pt
```

`--max-samples` 只传给两个 cache producer，表示相同的前 N 个动作记录；触觉缓存会扩展到这些 strided records 的最大帧索引以维持对齐。`--max-steps` 只传给训练器；它们会记录到 `decoder/pipeline_run.json`，不会悄悄修改正式 YAML。

## 产物与部署交接

输出目录包含不可变 action/tactile cache manifest、`best.pt`、`last.pt` 和 `pipeline_run.json`。`best.pt` 是验证指标最优的可部署权重；`last.pt` 是可精确续训的最新状态，二者不要混用。交接部署方时提供 `best.pt`、训练 YAML、checkpoint metadata/source contract 和四传感器顺序，并继续以 `[B,50,20]` 与 `[B,4,512]` 的直接动作契约取数。

本地 CPU 测试只证明配置、缓存/检查点接口和基本逻辑；它不证明 GPU 训练可完成，也不证明相机、机器人或部署闭环成功。服务器上仍需先完成 `--check`、小缓存 smoke、一步和一 epoch 验证，再开始正式训练或机器人测试。
