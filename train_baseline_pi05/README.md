# 独立 Pi0.5 触觉基线训练

本项目约定只新增两个相互独立的项目目录：训练目录 `train_baseline_pi05/` 与由部署交接方维护的部署目录；本目录不导入、也不修改原有训练或部署代码。它不是 FRS 方案：没有反向/前向 FRS 成对流程，也没有残差或 `x_base`。

模型为直接全动作解码器：冻结 Pi0.5 产生 `[B,50,A]` 粗动作，冻结触觉 ResNet18 产生 `[B,S,512]` 当前帧 token；两个 Transformer decoder layer 直接预测完整动作。支持单臂 `A=10`、双臂 `A=20`，夹爪索引分别为 `9` 和 `9/19`（从 0 开始）。四传感器 `S=4` 的顺序为：

1. `observation.images.tactile_left_0`
2. `observation.images.tactile_right_0`
3. `observation.images.tactile_left_1`
4. `observation.images.tactile_right_1`

右臂实验也支持 `S=2`，顺序为 `tactile_left_1`、`tactile_right_1`（保留完整的 `observation.images.` 前缀）。后缀 `_1` 表示右臂相机，`left/right` 表示该夹爪的两块触觉面。通过 `decoder.tactile_keys` 选择；动作维度与触觉数量独立配置。

旧版错误使用了跨臂的 `tactile_right_0/right_1`。该传感器合同的触觉缓存和 decoder checkpoint 现在会被拒绝，不能通过重命名 metadata 或仅更换部署映射修复；需要重建触觉缓存并从头训练 decoder。右臂示例为此使用新的 `task3_right_two_face` / `task4_right_two_face` 触觉缓存与 decoder 输出目录，`resume: false`，旧产物保留用于对照。

## 配置与隔离环境

默认配置为 `configs/train_baseline_pi05.yaml`：数据集 `KaiyueChen/two_tubes_03`、Pi0.5 checkpoint、norm assets、0824 tactile encoder 与三个 `/workspace/baseline_pi05/...` cache/output 位置都在 YAML 中，部署服务器应先按本机绝对路径修改这些字段。不要修改源码来替换路径。

| 配置 | 动作/触觉 | checkpoint / norm asset |
|---|---|---|
| `train_baseline_pi05.yaml` | 双臂 20D / 四路 | 原双臂模型 / `two_tubes_0102` |
| `train_baseline_pi05_task3.yaml` | 右臂 10D / 两路 | `pi05_task3_0830_1w` / `insert_0102_train90` |
| `train_baseline_pi05_task4.yaml` | 右臂 10D / 两路 | `pi05_task4_0830_6k` / `press_0102` |

两个右臂配置是可编辑示例：设置实际的数据集根目录与动作列名（`action` 或 `actions`），相机映射仍由 `dataset.rename_map` 和 `dataset.camera_map` 控制。模型只编码配置中的相机。`source.model_action_dim` 必须匹配 checkpoint；`decoder.action_dim` 必须匹配数据集真实动作宽度。

所有示例默认 `decoder.device: cuda`；RTX 5080 16 GB 的源模型缓存 batch 默认为 4，decoder batch 为 256。缓存生成与 decoder 训练在独立进程运行。GPU 续训会在 CPU 恢复随机数状态，再恢复 CUDA 模型和优化器。

触觉缓存使用 `cache.tactile_batch_size: 32`（帧数），两路触觉对应每批 64 张图像，四路对应 128 张。冻结 ResNet18 使用整网 JIT 推理。内嵌图像数据直接读取所选触觉列并用 NumPy 预处理，视频数据继续使用 LeRobot 的视频解码；避免逐帧调用 Torch 图像转换。batch size 不改变缓存索引和传感器顺序；只有传感器顺序及其他合同字段一致的完整触觉缓存才可复用。

环境只允许是本目录的 `.venv`，Python 3.12：

```bash
bash train_baseline_pi05/scripts/setup_env.sh
```

启动脚本只使用 `train_baseline_pi05/.venv/bin/python`，并固定 `PYTHONSAFEPATH=1`、项目本地 vendored runtime 优先的 `PYTHONPATH`、`PYTHONUNBUFFERED=1` 与 `XLA_PYTHON_CLIENT_PREALLOCATE=false`：

```bash
bash train_baseline_pi05/scripts/start_train.sh --help
```

例如运行 task3（先修改 YAML 中的数据和输出路径）：

```bash
bash train_baseline_pi05/scripts/start_train.sh \
  --config train_baseline_pi05/configs/train_baseline_pi05_task3.yaml
```

动作缓存只对当前观测读取配置的相机，50 步专家动作从数值列批量读取，不再逐步解码视频。数值列视图在循环外创建一次，避免每个样本反复复制 Hugging Face/Arrow 表结构信息。缓存记录相机映射、重命名映射和动作列；修改这些字段或使用升级前的动作缓存时，需要设置新的 `cache.action_root`，避免复用不匹配的动作。

本次只修正触觉传感器选择，Pi0.5 动作缓存不依赖触觉，可保留原 `task3/action_cache`；数据集、相机映射、源模型、归一化、采样与划分合同仍须一致。本机重训命令如下，保持原 50 epochs 和其他超参数：

```bash
mkdir -p outputs/baseline_pi05/task3_right_two_face
set -o pipefail
CUDA_VISIBLE_DEVICES=0 bash train_baseline_pi05/scripts/start_train.sh \
  --config train_baseline_pi05/configs/train_baseline_pi05_task3_5080.yaml \
  2>&1 | tee outputs/baseline_pi05/task3_right_two_face/train.log
```

Pi0.5 动作缓存将完整采样函数 JIT 编译一次并复用，固定采样噪声和 10 步求解过程不变；切换 batch 形状时会首次编译该形状。进度条的 `data`、`infer`、`write` 分别显示上一批的数据准备、推理和写盘耗时。动作缓存进程将 Torch CPU 线程数设为 1，避免小图像转换中的多线程调度开销。本机 `train_baseline_pi05_task3_5080.yaml` 使用实测通过的 `action_batch_size: 16`；其余通用配置保留 4。调整 batch size 可以继续已有动作缓存，无需重新生成完整触觉缓存。

本机 task3 配置还启用 `cache.action_prefetch: true`：单个后台线程在 CPU 上运行相同的预处理，并提前准备下一批；主线程负责 GPU 推理和按顺序写盘。只预取一批，既不提前遍历全数据集，也不加载额外模型。`data` 显示该批完整预处理耗时，`wait` 显示主线程实际等待数据的时间，`pack` 显示组批和提交设备数组的耗时；预取开启时 `data` 与推理重叠，不能将这些耗时直接相加。关闭该选项可恢复串行路径，未指定时默认关闭。实际 task3 的 32 个样本已验证 CPU/GPU 预处理输出完全一致；该验证不代表任意图像尺寸下跨设备舍入都逐位相同。

## 运行顺序

管线始终以三个独立进程依次执行 `tactile_cache`、`prepare_action_cache`、`train`。两个 JAX cache producer 必须完全退出后才启动 PyTorch 训练。

两个缓存阶段均显示进度条（完成数量、速度、耗时和预计剩余时间），使用 `2>&1 | tee ...` 时仍会显示。动作缓存断点恢复从已有样本数继续显示；加载数据、加载模型、写盘验证及复用缓存也有阶段提示。首次编码/推理包含 JAX 编译，第一帧或第一批可能耗时较长。

先进行只读、离线的路径检查；它不会建立目录、下载、读取视频或载入大 checkpoint：

```bash
bash train_baseline_pi05/scripts/start_train.sh \
  --config train_baseline_pi05/configs/train_baseline_pi05.yaml --check
```

小型 smoke 必须使用独立 YAML 和独立 cache/output roots，绝不改动正式 YAML：

```bash
SMOKE_CONFIG=/workspace/baseline_pi05/smoke/train_baseline_pi05_smoke.yaml
mkdir -p "$(dirname "$SMOKE_CONFIG")"
cp train_baseline_pi05/configs/train_baseline_pi05.yaml "$SMOKE_CONFIG"
sed -i \
  -e 's|/workspace/baseline_pi05/action_cache|/workspace/baseline_pi05/smoke/action_cache|' \
  -e 's|/workspace/baseline_pi05/tactile_embedding_cache|/workspace/baseline_pi05/smoke/tactile_embedding_cache|' \
  -e 's|/workspace/baseline_pi05/decoder|/workspace/baseline_pi05/smoke/decoder|' \
  "$SMOKE_CONFIG"
```

对该 YAML 先做只读检查，再用一次完整三阶段管线同时生成匹配的 capped caches 并训练一步（两个 override 都不改写 YAML）：

```bash
bash train_baseline_pi05/scripts/start_train.sh \
  --config "$SMOKE_CONFIG" --check
bash train_baseline_pi05/scripts/start_train.sh \
  --config "$SMOKE_CONFIG" --max-samples 128 --max-steps 1
```

一 epoch smoke 仍使用同一独立 YAML 和同一 capped cache selection：

```bash
sed -i -e 's/^  epochs: 50$/  epochs: 1/' "$SMOKE_CONFIG"
bash train_baseline_pi05/scripts/start_train.sh \
  --config "$SMOKE_CONFIG" --max-samples 128
```

正式训练使用原始 YAML（或另复制一份并把三个 roots 都改为新的 formal roots），默认 50 epochs：

```bash
FORMAL_CONFIG=train_baseline_pi05/configs/train_baseline_pi05.yaml
bash train_baseline_pi05/scripts/start_train.sh --config "$FORMAL_CONFIG"
```

不要把 capped cache 扩成 full：完整训练必须使用新的 formal cache/output roots，不能以相同 root 重跑不带 `--max-samples` 的 smoke YAML。恢复训练时把 formal YAML 的 `decoder.resume` 设为 `true`，保留 formal `last.pt`，再运行同一条正式命令。评估独立于三阶段训练管线：

```bash
train_baseline_pi05/.venv/bin/python -m train_baseline_pi05.evaluate \
  --config train_baseline_pi05/configs/train_baseline_pi05.yaml \
  --checkpoint /workspace/baseline_pi05/decoder/best.pt
```

`--max-samples` 只传给两个 cache producer，表示相同的前 N 个动作记录；触觉缓存会扩展到这些 strided records 的最大帧索引以维持对齐。已完成的 tactile cache 对 encoder、dataset、sensor order、预处理、帧数与 selection contract 均不可变；相同合同只复用，不同合同必须换 root。`--max-steps` 只传给训练器；它们会记录到 `decoder/pipeline_run.json`，不会悄悄修改正式 YAML。

## 产物与部署交接

输出目录包含不可变 action/tactile cache manifest、`best.pt`、`last.pt` 和 `pipeline_run.json`。`best.pt` 是验证指标最优的权重；`last.pt` 是可精确续训的最新状态。交接部署方时提供 `best.pt`、训练 YAML、checkpoint metadata/source contract 和传感器顺序，并按实际的 `[B,50,A]` 与 `[B,S,512]` 取数。本次单臂适配仅涉及训练项目，部署端仍须支持对应的动作维度和触觉数量。

本地 CPU 测试只证明配置、缓存/检查点接口和基本逻辑；它不证明 GPU 训练可完成，也不证明相机、机器人或部署闭环成功。服务器上仍需先完成 `--check`、小缓存 smoke、一步和一 epoch 验证，再开始正式训练或机器人测试。
