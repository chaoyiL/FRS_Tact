# Flow Decoder

这个目录实现一个无条件、双向自注意力的 Flow Matching 解码器。它先使用指定的
OpenPI config/checkpoint 对数据集 observation 做动作采样，再将预测动作沿原模型速度场
从 `t=0` 反向积分到 base 端 `t=1`。解码器学习从该 `x_base` 重建同一个预测动作。

所有动作均位于 OpenPI 数据变换产生的归一化动作空间。解码器不会读取图像、触觉、
state 或语言；它只读取 `[action_horizon, action_dim]` 的序列和流时间。

## 环境

从仓库根目录运行命令。先安装项目锁定的解码器依赖：

```bash
uv sync
```

如使用 NVIDIA GPU，请确保已执行 `uv sync` 安装带 CUDA 13 的 JAX。`prepare` 还要求原仓库的
`eval_scripts/loglike_evaluate.py` 能在同一环境正常启动，因为两者共享 OpenPI 与 LeRobot 数据代码。

## 1. 准备配对缓存

```bash
uv run python -m flow_decoder.prepare \
  --config-name pi05_bi \
  --checkpoint-dir checkpoints/50000 \
  --cache-dir flow_decoder_outputs/cache
```

数据集选择与 `eval_scripts/loglike_evaluate.py` 一致：由 `config.py` 中的数据集常量和所选
config 决定，并使用 checkpoint 下的 normalization assets。程序不提供 dataset repo 覆盖。

默认参数：

- 原模型采样 10 个 Euler steps；
- 预测动作到 `x_base` 的反向积分 120 steps；
- 以 dataset index 折叠固定 seed，保证每个样本的原模型初始噪声可复现；
- episode 固定 seed 打乱后按 80%/20% 分为 train/val；
- 完整处理所有帧。

完整数据预计算可能非常慢。建议先做小规模冒烟：

```bash
uv run python -m flow_decoder.prepare \
  --config-name pi05_bi \
  --checkpoint-dir checkpoints/50000 \
  --cache-dir flow_decoder_outputs/cache_smoke \
  --max-episodes 5 \
  --frame-stride 100 \
  --max-samples 32 \
  --batch-size 2
```

缓存使用 `.npy` memmap，包含 `x_base`、`predicted_actions`、dataset/episode index、split 和
原模型正反积分误差。每个 batch 落盘后原子更新 `manifest.json`；使用完全相同的参数重跑会
从 `completed_samples` 继续。若 config、checkpoint、选择参数或样本索引不同，程序会拒绝
复用该目录，请使用新的 `--cache-dir`。

可用筛选参数：

- `--frame-stride N`：每个 episode 每隔 N 帧取一个样本；
- `--max-episodes N`：只考虑前 N 个 episode，最小为 2；
- `--max-samples N`：在 train/val 内按比例、固定 seed 抽样；
- `--split-seed`、`--val-fraction`：控制 episode 级划分；
- `--inference-seed`：控制原模型每个 dataset index 的初始噪声。

## 2. 训练解码器

```bash
uv run python -m flow_decoder.train \
  --cache-dir flow_decoder_outputs/cache \
  --output-dir flow_decoder_outputs/run_01
```

默认网络为 `model_dim=128, depth=4, num_heads=4, mlp_ratio=4`，优化器为 AdamW，
学习率 `3e-4`、weight decay `1e-4`、batch size 256、训练 100 epochs。训练路径为：

```text
x_t = (1 - t) * x_base + t * predicted_action
target_velocity = predicted_action - x_base
```

每个 epoch 在完整验证 episode 上用 50 步 Euler 解码，按 validation reconstruction MSE
保存 `best/`，同时保存 `last/`。`history.csv` 记录训练 flow loss 和验证指标。

## 3. 验证

```bash
uv run python -m flow_decoder.evaluate \
  --cache-dir flow_decoder_outputs/cache \
  --checkpoint-dir flow_decoder_outputs/run_01/best \
  --output-dir flow_decoder_outputs/run_01/evaluation \
  --save-predictions
```

默认会额外生成 PNG 可视化；可用 `--no-plots` 关闭，或用 `--num-trajectory-samples 0`
跳过动作轨迹图。

### 输出文件说明

| 文件 | 含义 |
|------|------|
| `metrics.json` | 验证集汇总指标：样本数、解码 Euler 步数、平均 flow loss / MSE / RMSE / MAE，以及所用 checkpoint 路径与 epoch |
| `per_sample.csv` | 每个验证样本一行：`cache_index`（缓存行号）、`dataset_index` / `episode_index`（原数据集索引）、该样本的 flow loss / MSE / RMSE / MAE |
| `predictions.npz` | 仅在使用 `--save-predictions` 时生成；含 `cache_indices` 与 `predicted_actions`（解码器从 `x_base` 重建的动作，shape `[N, action_horizon, action_dim]`） |
| `metrics_histogram.png` | 四个指标的 per-sample 直方图，红虚线为均值 |
| `metrics_scatter.png` | flow loss（横轴）vs 重建 MSE（纵轴）散点图，颜色表示 MAE |
| `per_episode_mse.png` | 按 episode 分组的重建 MSE 箱线图，用于看误差是否集中在某些 episode |
| `action_trajectories.png` | 从验证集选取最好 / 中间 / 最差若干样本，对比 **target**（OpenPI 预测动作，实线）与 **decoded**（解码器重建，虚线），默认画前 3 个动作维度 |

指标含义：

- **flow loss**：在 `t=0.5` 处，模型预测速度场与真值速度 `target - x_base` 的均方误差（训练目标）；
- **MSE / RMSE / MAE**：用 `num_steps` 步 Euler 从 `x_base` 解码后，与缓存中 `predicted_actions`（原模型动作）之间的重建误差。

## 测试

```bash
uv run python -m unittest discover -s flow_decoder/tests -v
```

测试覆盖 episode 隔离、样本限制、Euler 积分方向和时间点、双向 self-attention、梯度、
checkpoint 往返，以及合成配对 flow 的收敛。
