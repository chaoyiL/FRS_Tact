# Tactile Flow Steering

触觉条件流匹配解码器：消费根目录 `prepare.py` 生成的 action cache，训练/评估时在线加载触觉图，经冻结 `tactile_encoder` 编码为 2 个条件 token，用 **cross-attention** 注入，流匹配目标为数据集 **GT 动作**。

## 数据流

1. 根目录 `prepare.py` → `x_base.npy` / `predicted_actions.npy` / `gt_actions.npy`
2. `train`：cache + 在线触觉 encode → `v(x_t, t, tactile_tokens) → gt_action`
3. `evaluate`：`decode(x_base, tactile_tokens)` vs **gt_action** MSE

不写单独的 tactile prepare，也不微调 tactile encoder。

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
  --output-dir tactile_flow_steering/outputs/run_01
```

数据集 repo 默认取 cache manifest 的 `configuration.dataset_repo_id`，可用 `--dataset-repo-id` 覆盖。

每 step：左右腕各 encode 一次 → `tactile_tokens [B, 2, D]`（`stop_gradient`）→ FM loss（target=`gt_action`）。

## 3. 评估

```bash
uv run python -m tactile_flow_steering.evaluate \
  --cache-dir tactile_flow_steering/outputs/cache \
  --tactile-encoder-dir path/to/tactile_encoder/checkpoint \
  --checkpoint-dir tactile_flow_steering/outputs/run_01/best \
  --output-dir tactile_flow_steering/outputs/run_01/evaluation \
  --save-predictions
```

输出 `metrics.json` / `per_sample.csv`（相对 GT 的 MSE/RMSE/MAE + t=0.5 flow loss）。

## 测试

```bash
uv run python -m unittest discover -s tactile_flow_steering/tests -v
```
