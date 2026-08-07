# FRS base model 切换：SmolVLA → pi0.5（JAX / openpi）

分支：`pi05-frs-jax`（从 `eric` 切出）。目标：FRS（`tactile_flow_steering`）不再使用 SmolVLA，
改用 Physical Intelligence 的 **pi0.5**（openpi 的原生 JAX 实现）作为 base model。这个文件记录
已确认的架构决策、关键发现和还没做完的事，避免跨 session 丢上下文。

## 架构决策：pi0.5 跑在独立环境里，不进主依赖

`openpi`（pi0.5 的官方仓库）自己的 `pyproject.toml` 会拉：

- `lerobot`（pin 死 HuggingFace 官方仓库某个 commit）—— **和本仓库自己的包名撞了**：
  本仓库的 `pyproject.toml` 里 `name = "lerobot"`（精简过的 SmolVLA-only 版本）。
- `jax[cuda12]==0.5.3` / `flax==0.10.2` / `transformers==4.53.2` / `orbax-checkpoint==0.11.13`，
  都是精确 pin，和本仓库现在的 `jax>=0.6.2,<0.9.0`、`flax>=0.11.0,<0.13.0` 等范围对不上。

结论：**不把 openpi 加进主 `pyproject.toml`**。改为在 [openpi_bridge/](openpi_bridge/) 建一个完全独立的
uv 子项目/venv，只用来跑"pi0.5 推理 → 产出 action_cache"这一步；主环境（这个仓库的 jax/flax 版本锁定、
`train_smolvla_jax.yaml` / `train_vtsmolvla_jax.yaml` 等）完全不受影响。

这个决策能成立，是因为现有 FRS 链路本来就在这一步天然解耦：

```
[openpi_bridge，独立 venv]                    [本仓库主环境]
pi0.5 checkpoint + pick_tube 数据集                tactile_encoder（冻结）
        │ 推理 + 反向 ODE 积分                            │ 逐帧编码
        ▼                                                ▼
   action_cache/<repo_id>/                    tactile_embeddings/<repo_id>/
   (target/x_base/gt_action/inversion_mse     (每帧 4 路 frozen ResNet embedding)
    + manifest.json，格式见 utils/cache.py)
        └──────────────────┬─────────────────────────────┘
                            ▼
              tools/train_frs.py → tactile_flow_steering/train.py
              （只读两份 cache，完全不 import 任何 base 模型代码）
```

`tactile_flow_steering/train.py` 不 import `lerobot.policies.smolvla_jax` 或任何 base 模型代码，
只消费 `action_cache` 目录（`utils/cache.py` 定义的 memmap + manifest 格式）。所以只要 pi0.5 一侧
产出同样格式的 cache，`tools/train_frs.py` 和 `configs/train_frs_pick_tube_pi05.yaml`
（见下文）**不需要改一行代码**。

## 现状：只搭了骨架，核心推理逻辑还没写

已经做的：

- [utils/cache.py](utils/cache.py) 里的 `build_records` 现在只依赖鸭子类型的 `metadata`
  （`total_episodes` + `episodes[i]["dataset_from_index"/"dataset_to_index"]`），不 import 任何
  jax/lerobot 代码 —— 本仓库和 openpi 环境里各自的 `LeRobotDatasetMetadata` 都满足这个接口，
  所以两边可以复用同一份记录选择/train-val 切分逻辑，不会出现两套实现漂移的问题。
  （原来这段代码在 `prepare.py` 里，是 SmolVLA 专用的，现在挪出来给两边共用。）
- [openpi_bridge/](openpi_bridge/) 子项目骨架（`pyproject.toml` + `prepare_pi05_cache.py` 骨架）。
- [configs/train_frs_pick_tube_pi05.yaml](configs/train_frs_pick_tube_pi05.yaml)：
  在 `train_frs_pick_tube.yaml` 基础上把 checkpoint 相关字段换成 pi0.5，其余（数据集、
  tactile encoder、frs_training 超参）先原样保留。

**没做的（核心工作，都在 `openpi_bridge/prepare_pi05_cache.py` 里标了 TODO）：**

1. **DataConfig / observation 映射** —— 把 pick_tube 数据集的相机 key（`observation.images.camera1/2`）、
   state、language 映射成 pi0.5 期望的 observation 字典（`base_0_rgb` / `left_wrist_0_rgb` /
   `right_wrist_0_rgb` + `state` + `tokenized_prompt`，参考 openpi `src/openpi/policies/*_policy.py`
   的写法）。这是接自定义机器人数据集时必须要写的部分，官方仓库里没有现成的。
2. **暴露 velocity_fn 用于反向积分** —— `openpi.models.pi0.Pi0.sample_actions` 只在内部一个
   `jax.lax` 循环里做 t:1→0 的采样，没有像本仓库 SmolVLA 那样把单步 `denoise_step(params, prefix_ctx, x_t, t)`
   独立暴露出来。FRS 需要的"反向积分"（t:0→1，见 [utils/source_model.py](utils/source_model.py) 的
   `sample_and_reverse`/`reverse_integrate_actions`，用 `utils/integration.py` 的
   euler/fireflow）要求能在任意 `(x, t)` 上调用 velocity field。需要参考 `Pi0.sample_actions`
   内部的 `embed_prefix` / `embed_suffix` / `action_out_proj` 自己拼一个可复用的单步函数。
   好消息：pi0.5 用的也是同一套 flow-matching 约定（t=1 是噪声，t=0 是数据，`dt = -1/num_steps`），
   和 SmolVLA 完全一致，`utils/integration.py` 里的 euler/fireflow 求解器可以直接复用，
   只需要换 `velocity_fn`。
3. **checkpoint 来源** —— 官方 pi0.5 base checkpoint 在
   `gs://openpi-assets/checkpoints/pi05_base/{params,assets}`（通过
   `openpi.shared.download.maybe_download` 下载，需要能访问 GCS）。openpi 仓库里没有现成的
   "zero-shot 跑 pi05_base" `TrainConfig`，只有一堆以它作初始权重的 finetune 配置
   （`pi05_aloha` / `pi05_droid` / `pi05_libero` 等，见 `src/openpi/training/config.py`）。
   需要照着这些例子给 pick_tube 数据集写一个新的 `TrainConfig`（模型用
   `openpi.models.pi0_config.Pi0Config(pi05=True)`，`weight_loader` 指向
   `gs://openpi-assets/checkpoints/pi05_base/params`）。
4. 跑通后依次执行：`openpi_bridge` 产出 4 个 pick_tube 数据集的 action_cache
   → 现有 `tools/precompute_tactile_embeddings.py`（不用改，和 base 模型无关）
   → 现有 `tools/train_frs.py --config configs/train_frs_pick_tube_pi05.yaml`。

## 关键 openpi API 参考（写下来免得以后重新翻源码）

（版本：openpi `main` 分支，2026-08 抓取）

- `openpi.policies.policy_config.create_trained_policy(train_config, checkpoint_dir) -> Policy`
  （`src/openpi/policies/policy_config.py`）：给定 `TrainConfig` + checkpoint 目录，
  自动识别 JAX/PyTorch 权重格式并构建好 `Policy`（含 normalize/unnormalize transform）。
- `Policy.infer(obs: dict, *, noise: np.ndarray | None = None) -> dict`（`.../policy.py`）：
  单帧/单 batch 推理，返回 `{"actions": ...}`；支持传入外部 `noise` 做确定性推理（对齐
  `utils/source_model.py` 里 `deterministic_noise` 的用法）。
- `openpi.models.pi0.Pi0.sample_actions(rng, observation, *, num_steps=10, noise=None)`：
  和 `Pi0.compute_loss` 用的是同一套 flow-matching 约定；内部用 `embed_prefix`（一次性构建
  prefix KV cache）+ 循环里的 `embed_suffix`/`action_out_proj`。
- `openpi.shared.download.maybe_download(url)`：支持 `gs://` / 本地路径，下载到
  `~/.cache/openpi`（可用 `OPENPI_DATA_HOME` 环境变量改路径）。
- pi0.5 base checkpoint：`gs://openpi-assets/checkpoints/pi05_base/params`
  （+ `.../assets` 存 norm stats）。
- `openpi.models.pi0_config.Pi0Config(pi05=True, action_horizon=..., ...)`。

## 明确不做的事

- 不改 `pyproject.toml`（主环境）、不装 openpi 到主 venv。
- 不改 `lerobot/policies/smolvla_jax/*`、`configs/train_smolvla_jax.yaml`、
  `configs/train_vtsmolvla_jax.yaml`——这些保持给 SmolVLA 用，这个分支的目标是"FRS 换 base"，
  不是同时维护两条线。
- 不在这台本地 macOS 机器上跑 `uv sync` / 下载 checkpoint —— `jax[cuda12]` 只在 Linux+NVIDIA
  上能装，这些都要在训练服务器上跑（参考 [train_for_agent.md](train_for_agent.md) 的环境搭建流程）。
