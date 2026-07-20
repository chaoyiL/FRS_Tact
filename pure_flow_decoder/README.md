# Pure Flow Decoder

这个目录实现一个无条件、双向自注意力的 Flow Matching 解码器。它先使用指定的
SmolVLA-JAX checkpoint 对数据集 observation 做动作采样，再将预测动作沿原模型速度场
从 `t=0` 反向积分到 base 端 `t=1`。解码器学习从该 `x_base` 重建同一个预测动作。

所有动作均位于模型归一化动作空间。解码器不会读取图像、触觉、state 或语言；
它只读取 `[action_horizon, action_dim]` 的序列和流时间。

## 环境

从仓库根目录运行命令：

```bash
uv sync
```

## 1. 准备配对缓存

用仓库根目录的 `prepare.py`（不要再用本包内的 prepare）：

```bash
uv run python prepare.py \
  --checkpoint-dir checkpoints/black-smash-smolvla-40k \
  --dataset-repo-id chaoyi/black_smash_01 \
  --cache-dir pure_flow_decoder/outputs/cache
```

缓存会写入：

- `predicted_actions.npy`：VLA 采样得到的动作
- `x_base.npy`：沿速度场反向积分得到的噪声端
- `gt_actions.npy`：数据集 Ground Truth 动作（与 VLA 相同的归一化空间）
- 以及 dataset/episode index、split、inversion MSE

冒烟示例：

```bash
uv run python prepare.py \
  --checkpoint-dir checkpoints/black-smash-smolvla-40k \
  --dataset-repo-id chaoyi/black_smash_01 \
  --cache-dir pure_flow_decoder/outputs/cache_smoke \
  --max-episodes 5 \
  --frame-stride 100 \
  --max-samples 32 \
  --batch-size 2
```

## 2. 训练解码器

```bash
uv run python -m pure_flow_decoder.train \
  --cache-dir pure_flow_decoder/outputs/cache \
  --output-dir pure_flow_decoder/outputs/run_01
```

训练目标仍是 `x_base → predicted_actions`（VLA 预测）；`gt_actions` 一并保存在缓存中供后续分析/对比。

## 3. 验证

```bash
uv run python -m pure_flow_decoder.evaluate \
  --cache-dir pure_flow_decoder/outputs/cache \
  --checkpoint-dir pure_flow_decoder/outputs/run_01/best \
  --output-dir pure_flow_decoder/outputs/run_01/evaluation \
  --save-predictions
```

## 测试

```bash
uv run python -m unittest discover -s pure_flow_decoder/tests -v
```
