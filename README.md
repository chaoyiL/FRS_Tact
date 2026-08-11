# FRS_Tact: pi0.5 JAX 版触觉 Flow Steering

这个仓库是 FRS 触觉 Flow Steering 训练管线的 **pi0.5 JAX 专用版本**。主 base model 从 SmolVLA
切换为 Physical Intelligence openpi 的 pi0.5，模型代码已直接 vendor 到本仓库：

- `src/lerobot/policies/pi05_jax/`：pi0.5 JAX 模型、checkpoint 加载、归一化和 FRS 反向积分需要的单步 denoise 接口。
- `prepare_pi05.py` / `tools/prepare_frs_pi05_cache.py`：用 pi0.5 生成 FRS action cache。
- `configs/train_frs_pick_tube_pi05.yaml`：四个 pick_tube 数据集的 pi0.5 FRS 配置。
- `scripts/start_frs_pi05_train.sh`：pi0.5 一键训练入口。

本分支不再兼顾 SmolVLA JAX 是否可跑。`src/lerobot/policies/smolvla_jax/`、`configs/train_smolvla_jax.yaml`
等文件保留为历史实现和共享工具来源，但当前主线是 pi0.5。

## 为什么不直接安装 openpi

openpi 会依赖官方 `lerobot` 包，而本仓库本身也提供 `src/lerobot/`。两个包在同一个 Python 环境里会发生
import 路径冲突，所以这里不 `pip install openpi`，而是只把 pi0.5 模型相关代码搬进本仓库。

因此，`pyproject.toml` 里的核心 JAX 依赖按 openpi pi0.5 版本固定：

- `jax[cuda12-local]==0.5.3`
- `flax==0.10.2`
- `transformers==4.53.2`
- `orbax-checkpoint==0.11.13`
- `ml-dtypes==0.4.1`

正式训练需要 Linux + NVIDIA GPU。本地 macOS 只适合改代码和做轻量静态检查。

## 快速开始

在训练服务器上执行：

```bash
git clone <this-repo>
cd FRS_Tact
bash scripts/setup_env.sh
```

如果数据还没有准备好，下载并转换 pick_tube 数据集：

```bash
bash scripts/download_data.sh
```

确认 `configs/train_frs_pick_tube_pi05.yaml` 里的路径存在，尤其是：

- `datasets[*].root`：默认指向 `/workspace/lerobot_v30/KaiyueChen/pick_tube_0X`
- `model.tactile_encoder_path`：默认指向 `/workspace/checkpoints/encoder_ckpt_05`
- `action_cache.root`：pi0.5 action cache 输出目录
- `tactile_embedding_cache.root`：触觉 embedding cache 输出目录
- `frs_training.output`：FRS 训练输出目录

启动完整 pi0.5 管线：

```bash
bash scripts/start_frs_pi05_train.sh configs/train_frs_pick_tube_pi05.yaml
```

默认会在 tmux 后台运行，session 名是 `frs_pick_tube_pi05`：

```bash
tmux attach -t frs_pick_tube_pi05
```

如果希望前台运行：

```bash
FRS_FOREGROUND=1 bash scripts/start_frs_pi05_train.sh configs/train_frs_pick_tube_pi05.yaml
```

## 管线步骤

`scripts/start_frs_pi05_train.sh` 会顺序执行：

1. 检查 JAX GPU 是否可用。
2. 加载 `gs://openpi-assets/checkpoints/pi05_base`，确认 checkpoint 参数能匹配 `Pi0Config(pi05=True)`。
3. 预计算四路触觉 ResNet embedding：

   ```bash
   uv run --no-sync python tools/precompute_tactile_embeddings.py \
     --config configs/train_frs_pick_tube_pi05.yaml
   ```

4. 生成 pi0.5 action cache：

   ```bash
   uv run --no-sync python tools/prepare_frs_pi05_cache.py \
     --config configs/train_frs_pick_tube_pi05.yaml
   ```

5. 训练 tactile FRS：

   ```bash
   uv run --no-sync python tools/train_frs.py \
     --config configs/train_frs_pick_tube_pi05.yaml
   ```

pi0.5 不需要 SmolVLA 的 PEFT adapter merge 步骤，官方 checkpoint 直接通过 Orbax/JAX 加载。

## pick_tube 配置要点

当前配置使用四个数据集：

- `KaiyueChen/pick_tube_01`
- `KaiyueChen/pick_tube_02`
- `KaiyueChen/pick_tube_03`
- `KaiyueChen/pick_tube_04`

pi0.5 固定使用三个图像槽位：`base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`。pick_tube 只有两路腕部相机，
所以配置里把：

- 原始 `camera0` 作为左腕相机
- 原始 `camera1` 作为右腕相机
- `base_0_rgb` 留空，由代码补黑图并设置 mask 为 false

数据集经过 `rename_map` 后，实际配置写的是：

```yaml
camera_map:
  left_wrist_0_rgb: observation.images.camera1
  right_wrist_0_rgb: observation.images.camera2
```

动作维度使用 pi0.5 常见的 `action_dim: 32`，pick_tube 原始 20 维 action 会在归一化和 padding 逻辑中处理。
`action_horizon` 当前设为 `50`。

## norm stats

当前配置直接复用官方 pi05_base 的 norm stats：

```yaml
norm_stats:
  dir: gs://openpi-assets/checkpoints/pi05_base/assets
  asset_id: trossen
  use_quantile_norm: true
```

这是为了先跑通整条链路。需要注意：`trossen` 的 state/action 统计量是 14 维，而 pick_tube 是 20 维。
代码会把多出的 6 维补成 mean=0、std=1，也就是不做真实归一化。正式实验如果要更严谨，应当为 pick_tube
单独计算 norm stats。

## 重要文件

- `src/lerobot/policies/pi05_jax/README.md`：pi0.5 vendor 代码来源、裁剪内容和模型级验证清单。
- `pi05_frs_plan.md`：从 SmolVLA 切到 pi0.5 的完整设计记录和审查记录。
- `modalities_eval/pi05_utils.py`：LeRobot sample 到 pi0.5 `Observation` 的适配层。
- `utils/pi05_source_model.py`：pi0.5 sampling + reverse integration。
- `utils/cache.py`：action cache 的 manifest、memmap 和 sample record 逻辑。
- `tactile_flow_steering/`：FRS 模型、数据读取、训练与评估。
- `tactile_encoder/`：触觉 encoder 训练与推理工具。

## 当前状态和必须验证的事

这条 pi0.5 链路的代码已经写完并做过离线审查，但还需要在 Linux + NVIDIA 训练服务器上完整实跑验证。
正式使用前至少确认：

1. `bash scripts/setup_env.sh` 能完成 `uv sync --frozen`。
2. `scripts/start_frs_pi05_train.sh` 的 checkpoint 加载检查能通过。
3. `Pi0.build_prefix_cache` + `Pi0.denoise_step` 手动逐步采样结果能和原始 `sample_actions` 对齐。
4. `tools/prepare_frs_pi05_cache.py` 能为四个 pick_tube 数据集生成完整 action cache。
5. `tools/train_frs.py` 能读取 pi0.5 action cache 和 tactile embedding cache 完成训练。

如果只想先做小规模 smoke test，可以在配置里临时设置：

```yaml
action_cache:
  max_episodes: 1
  max_samples: 32
```

确认通过后再换回完整配置。

## 已知限制

- 没有 pi0.5 真机部署代码；`deploy_smolvla/` 仍是 SmolVLA 时代的部署客户端。
- `modalities_eval/` 下多数分析脚本仍以 SmolVLA 为默认 base model，需要按需改成 `Pi05EvalModel`。
- 当前 pi0.5 只把 RGB 相机喂给 base model，触觉信息只进入下游 FRS 网络；没有采用“触觉图像直接进 pi0.5 SigLIP”的 VB-VLA fork 路线。
- 官方 pi05_base 没有 pick_tube 专属 norm stats，当前 `trossen` 配置只是链路跑通方案。

## 常用命令

```bash
# 环境
bash scripts/setup_env.sh

# 数据
bash scripts/download_data.sh

# 完整 pi0.5 FRS 管线
bash scripts/start_frs_pi05_train.sh configs/train_frs_pick_tube_pi05.yaml

# 分步运行
uv run --no-sync python tools/precompute_tactile_embeddings.py --config configs/train_frs_pick_tube_pi05.yaml
uv run --no-sync python tools/prepare_frs_pi05_cache.py --config configs/train_frs_pick_tube_pi05.yaml
uv run --no-sync python tools/train_frs.py --config configs/train_frs_pick_tube_pi05.yaml

# 评估训练出的 FRS checkpoint（示例：pick_tube_01，对其他数据集换 --cache-dir/--dataset-root）
uv run --no-sync python tactile_flow_steering/evaluate.py \
  --cache-dir /workspace/frs_pick_tube_pi05/action_cache/KaiyueChen/pick_tube_01 \
  --tactile-encoder-dir /workspace/checkpoints/encoder_ckpt_05 \
  --checkpoint-dir /workspace/frs_pick_tube_pi05/run_gated_01/best \
  --output-dir /workspace/frs_pick_tube_pi05/run_gated_01/eval_pick_tube_01 \
  --dataset-repo-id KaiyueChen/pick_tube_01 \
  --dataset-root /workspace/lerobot_v30/KaiyueChen/pick_tube_01
```
