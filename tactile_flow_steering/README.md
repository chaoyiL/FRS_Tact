# Tactile Flow Steering

触觉条件流匹配解码器：消费根目录 `prepare.py` 生成的 action cache；训练/评估时在线加载时间窗内的触觉图，经**冻结 ResNet** 编码后，由**可训练共享 GRU**（四路触觉各自过同一 GRU）得到 4 个 hidden token，经 **cross-attention** 注入。

## 数据流

1. 根目录 `prepare.py` → `x_base.npy` / `predicted_actions.npy` / `gt_actions.npy`
2. `train`：cache + 时间窗触觉 → frozen ResNet `[B,T,4,D]` → shared GRU → `[B,4,H]` → CrossAttn FM
3. `evaluate`：同样窗口 → `decode(x_base, tactile_seq)` vs **gt_action** MSE

不写单独的 tactile prepare；不微调 ResNet；端到端训练 **GRU + 去噪网络**。

## 环境

在仓库根目录：

```bash
uv sync
```

## 1. 准备 action cache

```bash
uv run python prepare.py \
  --checkpoint-dir checkpoints/black-smash-smolvla-40k \
  --dataset-repo-id chaoyi/black_smash_01 \
  --cache-dir tactile_flow_steering/outputs/cache
```

## 2. 训练

```bash
uv run python -m tactile_flow_steering.train \
  --cache-dir tactile_flow_steering/outputs/cache \
  --tactile-encoder-dir path/to/tactile_encoder/checkpoint \
  --output-dir tactile_flow_steering/outputs/run_01 \
  --tactile-window-divisor 1 \
  --loss-mode gt
```

- `--tactile-window-divisor`：`tactile_window = action_horizon // divisor`（须整除；默认 1）
- GRU hidden 维固定为 **256**（不可配置）
- `--loss-mode`：
  - `gt`（默认）：仅 `L*`，target = GT
  - `gated`：`L = w L* + λ (1-w) L_stop`
    - `L*`：target = GT；`L_stop`：target = VLA `predicted_actions`
    - `s = mean_i(1 - cos(v_i[t], v_i[ep0]))`（当前帧 vs episode 首帧 ResNet token）
    - `w = sigmoid((s - τ) / T)`
    - CLI：`--gate-tau`（默认 0.5）、`--gate-temperature`（默认 0.1）、`--gate-lambda`（默认 1.0）

门控示例：

```bash
uv run python -m tactile_flow_steering.train \
  --cache-dir tactile_flow_steering/outputs/cache \
  --tactile-encoder-dir path/to/tactile_encoder/checkpoint \
  --output-dir tactile_flow_steering/outputs/run_gated \
  --loss-mode gated \
  --gate-tau 0.5 \
  --gate-temperature 0.1 \
  --gate-lambda 1.0
```

## 3. 评估

```bash
uv run python -m tactile_flow_steering.evaluate \
  --cache-dir tactile_flow_steering/outputs/cache \
  --tactile-encoder-dir path/to/tactile_encoder/checkpoint \
  --checkpoint-dir tactile_flow_steering/outputs/run_01/best \
  --output-dir tactile_flow_steering/outputs/run_01/evaluation \
  --save-predictions
```

评估始终相对 **GT**。窗口参数默认从 checkpoint metadata 读取。

## 测试

```bash
uv run python -m unittest discover -s tactile_flow_steering/tests -v
```
