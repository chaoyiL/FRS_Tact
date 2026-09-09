# Flow Decoder / Action Decoder 实验交接说明

仓库中没有单独的 `action_decoder/`，对应实现位于 `flow_decoder/`。

## 1. 实验目标

该实验训练一个无条件、双向 self-attention 的 Flow Matching 解码器：

```text
数据集 observation
    ↓
OpenPI / Pi0.5 源模型从固定高斯噪声采样动作 y
    ↓
沿源模型速度场从动作端 t=0 积分回 base 端 t=1，得到 x_base
    ↓
训练独立 decoder：x_base → y
```

需要特别注意：

- 训练目标 `y` 是 OpenPI 预测动作，不是数据集 ground-truth action。
- 解码器不读取 observation、图像、触觉、state 或语言。
- 解码器输入只有 `[B, action_horizon, action_dim]` 的 `x_t` 和标量时间 `t`。
- 所有动作都处于 OpenPI 数据管线产生的归一化动作空间。
- 当前 `pi05_bi` 的动作形状为 `action_horizon=50, action_dim=20`。

核心入口：

- 缓存准备：[`prepare.py`](prepare.py)
- 训练：[`train.py`](train.py)
- 评估：[`evaluate.py`](evaluate.py)
- 不完整缓存收尾：[`finalize_cache.py`](finalize_cache.py)

## 2. 配对缓存是怎样生成的

### 2.1 数据集与源模型

当前 OpenPI 配置为：

- config：`pi05_bi`
- source checkpoint：`checkpoints/50000`
- 当前数据集常量：`chaoyi/white_smash_07`
- normalization asset：`white_smash_07`
- state/action dim：20
- action horizon：50
- backbone：`gemma_2b`
- action expert：`gemma_300m`

配置定义在 [`openpi/src/openpi/training/config.py`](../openpi/src/openpi/training/config.py)。

数据集不是由 `flow_decoder.prepare` 的 CLI 参数直接指定的。`eval_scripts.utils.create_transformed_dataset()` 会强制使用：

```python
DATASET_REPO_NAMESPACE + "/" + DATASET_TRAIN_NAME
```

对应实现位于 [`eval_scripts/utils.py`](../eval_scripts/utils.py)。迁移时最好把 `dataset_repo_id` 和 `asset_id` 改成显式参数，避免依赖全局常量。

### 2.2 每个样本的构造

对 dataset index 为 `i` 的 observation：

1. 生成固定噪声：

   ```python
   z_i = Normal(0, I)
   key_i = fold_in(PRNGKey(inference_seed), dataset_index)
   shape = [50, 20]
   ```

   因此噪声与 batch 划分、处理顺序无关，具体见 [`utils/source_model.py`](utils/source_model.py)。

2. 用 OpenPI 源模型从 `t=1` 到 `t=0` 采样：

   ```text
   y_i = OpenPI.sample_actions(observation_i, noise=z_i, num_steps=10)
   ```

   OpenPI sampling 使用负时间步 Euler，默认 10 步。

3. 从预测动作 `y_i` 出发，沿同一个 observation-conditioned 源速度场从 `t=0` 积分到 `t=1`：

   ```text
   x_base_i = Integrate[source_velocity](initial=y_i, t=0→1)
   ```

   当前支持 `euler` 和 `fireflow` modified midpoint。源模型前缀会构建 KV cache，后续只重复计算 action suffix。

4. 记录反演误差：

   ```text
   inversion_mse_i = mean((x_base_i - z_i)^2)
   ```

   该指标只检查反向积分能否找回初始噪声，不参与 decoder 训练。

### 2.3 当前 prepare 的真实默认参数

以 [`prepare.py`](prepare.py) 的代码解析器为准：

| 参数 | 当前默认值 |
|---|---:|
| `--model-sample-steps` | 10 |
| `--reverse-steps` | 50 |
| `--reverse-solver` | `fireflow` |
| `--batch-size` | 16 |
| `--inference-seed` | 0 |
| `--split-seed` | 0 |
| `--val-fraction` | 0.2 |
| `--frame-stride` | 1 |
| `--max-episodes` | 不限制 |
| `--max-samples` | 不限制 |

FireFlow 的 `N` 步需要 `N+1` 次速度场求值；Euler 为 `N` 次，具体实现见 [`utils/integration.py`](utils/integration.py)。

### 2.4 train/val 划分

划分是 episode-disjoint，不是 frame 随机划分：

1. 对 episode id 排序；
2. 用 `np.random.default_rng(split_seed)` 打乱；
3. `round(num_episodes * val_fraction)` 个 episode 进入 validation；
4. 至少保留一个 train 和一个 val episode；
5. 每个 episode 内用 `[::frame_stride]` 选帧；
6. 如果指定 `max_samples`，再按 train/val 原比例固定 seed 抽样。

相关实现位于 [`utils/cache.py`](utils/cache.py) 和 [`prepare.py`](prepare.py)。

### 2.5 缓存格式

缓存为 `.npy` memmap：

| 文件 | shape / dtype | 含义 |
|---|---|---|
| `x_base.npy` | `[N, 50, 20]`, float32 | 源模型反演得到的 base |
| `predicted_actions.npy` | `[N, 50, 20]`, float32 | OpenPI 预测动作，训练 target |
| `dataset_indices.npy` | `[N]`, int64 | 原数据集行号 |
| `episode_indices.npy` | `[N]`, int64 | episode id |
| `split.npy` | `[N]`, uint8 | train=0，val=1 |
| `inversion_mse.npy` | `[N]`, float32 | `x_base` 与原噪声的 MSE |
| `manifest.json` | JSON | 完整参数、split、hash、进度 |

每个 batch flush 数组，然后原子更新 `completed_samples`，因此相同配置可以断点续跑。缓存定义见 [`utils/cache.py`](utils/cache.py)。

## 3. Decoder 网络结构

模型实现位于 [`utils/model.py`](utils/model.py)。当前训练 CLI 默认模型为：

| 参数 | 默认值 |
|---|---:|
| `action_horizon` | 从 cache 读取，当前 50 |
| `action_dim` | 从 cache 读取，当前 20 |
| `model_dim` | 256 |
| `depth` | 6 |
| `num_heads` | 4 |
| `mlp_ratio` | 4 |
| 参数量 | 5,275,156 |
| dropout | 0 |
| attention mask | 无，完整双向 attention |

网络路径：

```text
x_t [B, 50, 20]
  → Linear(20, model_dim)
  + 固定 sinusoidal sequence position embedding
  + TimeMLP(sinusoidal(t))
  → 6 × PreNorm Transformer Block
  → LayerNorm
  → Linear(model_dim, 20)
  → predicted velocity [B, 50, 20]
```

每个 Transformer block：

```text
x = x + MultiHeadAttention(LayerNorm(x))
x = x + Linear(4d→d)(GELU(Linear(d→4d)(LayerNorm(x))))
```

没有 causal mask，因此任意动作时间点可以读取整个 50-step action chunk。

注意：`DecoderConfig` 类本身仍保留旧默认值 `128/4`，但训练 CLI 会显式传入当前默认 `256/6`；迁移时应以 [`train.py`](train.py) 为准。

## 4. Flow Matching 训练目标

令 `x_base` 为缓存的 base，`y` 为缓存的 OpenPI predicted action，并对每个样本独立采样 `t ~ Uniform(0,1)`。

训练插值为：

$$
x_t=(1-t)x_{\text{base}}+t y
$$

真值速度是常数：

$$
u_t=y-x_{\text{base}}
$$

损失为：

$$
L=\mathbb{E}_{t,x}\left[
\operatorname{mean}_{T,A}
\left(v_\theta(x_t,t)-u_t\right)^2
\right]
$$

对应代码见 [`utils/model.py`](utils/model.py)。解码时求解：

$$
\frac{dx}{dt}=v_\theta(x,t),\qquad x(0)=x_{\text{base}},\qquad \hat y=x(1)
$$

## 5. 当前训练参数

当前真实 CLI 默认值如下，见 [`train.py`](train.py)：

| 类别 | 参数 | 默认值 |
|---|---|---:|
| 模型 | `model-dim` | 256 |
| 模型 | `depth` | 6 |
| 模型 | `num-heads` | 4 |
| 模型 | `mlp-ratio` | 4 |
| 优化 | optimizer | AdamW |
| 优化 | base learning rate | `3e-4` |
| 优化 | weight decay | `1e-4` |
| 优化 | global grad clip | 1.0 |
| 优化 | warmup | 10 epochs |
| 优化 | schedule | cosine |
| 优化 | final LR ratio | 0.1 |
| 优化 | LR reference dim | 256 |
| 训练 | batch size | 256 |
| 训练 | epochs | 1000 |
| 训练 | seed | 0 |
| 验证 | validation Euler steps | 10 |

Optax 0.2.8 的 AdamW 默认参数没有被覆盖：

```text
beta1=0.9
beta2=0.999
eps=1e-8
eps_root=0
nesterov=False
```

有效峰值学习率按模型宽度缩放：

$$
lr_{\text{peak}}=3\times10^{-4}\sqrt{\frac{256}{model\_dim}}
$$

| model_dim | peak LR |
|---:|---:|
| 128 | `4.24264e-4` |
| 256 | `3e-4` |
| 512 | `2.12132e-4` |
| 1024 | `1.5e-4` |

学习率先从 0 线性 warmup 10 epochs，再 cosine decay 到峰值的 10%。实现见 [`utils/model.py`](utils/model.py)。

随机性：

- 参数初始化：`seed`
- epoch shuffle：`seed + epoch`
- 每个 batch 的 `t`：`fold_in(seed, epoch * 1_000_000 + batch_number)`

每个 epoch 都遍历完整 train split，最后一个不足 256 的 batch 也会训练。

### Checkpoint 选择

每个 epoch：

- 用完整 val split 验证；
- flow loss 固定在 `t=0.5` 测量；
- 用 10-step Euler 重建；
- 按最低 validation reconstruction MSE 保存 `best/`；
- 当前状态总是保存到 `last/`。

Checkpoint 只保存模型参数、结构、epoch、指标和 cache hash，不保存 optimizer state、LR schedule 或完整训练 CLI，因此当前实现不能从 `last/` 真正恢复训练。重新使用同一个 output dir 还会覆盖 `history.csv`。具体见 [`utils/checkpoint.py`](utils/checkpoint.py)。

## 6. 评估方法与参数

当前 standalone evaluate 默认值，见 [`evaluate.py`](evaluate.py)：

| 参数 | 默认值 |
|---|---:|
| batch size | 256 |
| decoder steps | 5 |
| solver | `fireflow` |
| save predictions | false |
| PNG plots | true |
| trajectory samples | 6 |
| episode strips | 6 |

指标定义：

- `flow_loss`：固定 `t=0.5` 的速度场 MSE；
- `mse`：所有样本的 action reconstruction MSE；
- `rmse = sqrt(global mean MSE)`；
- `mae`：所有 action 元素的平均绝对误差；
- per-sample 指标均对 `[50,20]` 求平均。

指标实现见 [`utils/metrics.py`](utils/metrics.py)。

一个容易遗漏的不一致是：

- 训练期间的 model selection：10-step Euler；
- standalone evaluate 默认：5-step FireFlow。

严格比较模型时，应显式指定相同 solver 和步数。

## 7. 仓库中的真实实验产物

### 7.1 Exp1-1：`byw_smash_13`

缓存位于 [`outputs/Exp1-1/cache/manifest.json`](outputs/Exp1-1/cache/manifest.json)：

- dataset：`chaoyi/byw_smash_13`
- asset：`byw_smash_13`
- checkpoint：`checkpoints/50000`
- source sampling：10-step Euler
- frame stride：10
- episode：100，train 80 / val 20
- samples：12,013
- train samples：9,624
- val samples：2,389
- reverse steps：120
- mean source inversion MSE：4.0331192

该旧 manifest 没有记录 `reverse_solver`。结合旧 README 和旧格式可判断它来自加入 FireFlow 参数之前，极大概率使用 120-step Euler；这是历史推断，不是 manifest 中显式记录的字段。

结构 sweep 如下；所有实验均为 `heads=4, mlp_ratio=4, action=50×20`：

| Run | dim × depth | 参数量 | history epochs | best epoch | best val MSE |
|---|---:|---:|---:|---:|---:|
| 01 | 128 × 4 | 0.930M | 100 | 82 | 0.01176584 |
| 02 | 256 × 4 | 3.696M | 200 | 188 | 0.00601952 |
| 03 | 256 × 4 | 3.696M | 500 | 419 | 0.00476051 |
| 04 | 512 × 4 | 14.731M | 500 | 494 | 0.00498403 |
| 05 | 256 × 5 | 4.485M | 500 | 452 | 0.00491807 |
| 06 | 256 × 5 | 4.485M | 500 | 383 | 0.00443028 |
| 07 | 512 × 4 | 14.731M | 500 | 419 | 0.00458505 |
| 08 | 256 × 6 | 5.275M | 500 | 388 | **0.00414291** |
| 09 | 256 × 7 | 6.065M | 500 | 326 | 0.00438466 |
| 10 | 256 × 7 | 6.065M | 800 | 400 | 0.00436319 |
| 11 | 1024 × 4 | 58.823M | 500 | 303 | 0.00539852 |
| 12 | 1024 × 4 | 58.823M | 500 | 455 | 0.00516371 |
| 13 | 512 × 6 | 21.036M | 500 | 224 | 0.00443729 |
| 14 | 512 × 6 | 21.036M | 500 | 233 | 0.00439006 |
| 15 | 256 × 6 | 5.275M | 500 | 424 | 0.00430543 |
| 16 | 256 × 6 | 5.275M | 1000 | 419 | 0.00430738 |

Exp1-1 中最低 reconstruction MSE 是 `run_08`。

同一结构存在多个结果明显不同的 run，但旧 checkpoint 没有保存 seed、学习率、schedule 等训练参数，所以不能从仓库产物准确恢复这些重复 run 之间究竟覆盖了哪些 CLI 参数。不要把当前默认 optimizer 设置当作所有历史 run 的确切设置。

### 7.2 Exp1-2：`white_smash_07`

实际缓存位于 [`../data_cache/manifest.json`](../data_cache/manifest.json)：

- dataset：`chaoyi/white_smash_07`
- samples：47,888
- episodes：40，train 32 / val 8
- train samples：38,251
- val samples：9,637
- frame stride：1
- source sampling：10-step Euler
- source reverse：50-step FireFlow
- mean source inversion MSE：4.3048129
- action shape：`[50,20]`

`data_cache` 中额外存在 `ground_truth_actions.npy`，但 `flow_decoder` 的 `CachedPairs` 不会读取它；训练仍然只使用 `predicted_actions.npy`。

模型：

- `model_dim=256`
- `depth=6`
- heads=4
- MLP ratio=4
- 参数量=5,275,156
- history 到 epoch 377
- best epoch 143
- best validation flow loss：0.00291748

评估结果：

| Solver | Steps | NFE | MSE | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| Euler | 10 | 10 | 0.00344144 | 0.05866382 | 0.03771492 |
| FireFlow | 5 | 6 | **0.00307198** | **0.05542545** | **0.03382555** |

5-step FireFlow 相对 10-step Euler 的 MSE 降低约 10.74%，同时速度场调用次数更少。
