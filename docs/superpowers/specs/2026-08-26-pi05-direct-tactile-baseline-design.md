# Pi0.5 Direct Tactile Decoder Baseline 设计

## 目标

新增两个互相独立、可搬运的项目：

- `train_baseline_pi05/`：冻结已有 Pi0.5 source policy 与 tactile encoder，只训练一个两层 PyTorch Transformer decoder。
- `deploy_baseline_pi05/`：部署纯视觉 Pi0.5 source policy，并在逐 action 执行前使用最新触觉通过 decoder 直接重新生成动作。

该 baseline 不导入 FRS decoder，不进行 source action reverse integration，也不进行 tactile-conditioned forward flow integration。Pi0.5 自身生成 coarse action 所需的标准 flow sampling 不属于 FRS 的两次积分，继续按原 checkpoint 推理契约保留。

## 已确认的资产与默认数据

- Pi0.5 checkpoint：`checkpoints/model/pi05_bi_two_tubes_0102_step16000/checkpoint`。
- Pi0.5 norm stats：上述 checkpoint 的 `assets/two_tubes_0102/norm_stats.json`。
- Tactile encoder：`checkpoints/encoder/encoder_ckpt_0824`。
- 默认 decoder 训练数据：`KaiyueChen/two_tubes_03`。
- 服务器上的 dataset、cache、checkpoint 和 output 路径全部由 YAML 配置，示例使用 `/workspace/...`，用户部署前自行修改。

这里的“纯视觉 Pi0.5”描述 source policy 的在线输入边界：Pi0.5 只接收两路 RGB、20D state 和 task；四路 tactile 图像只进入独立 tactile encoder 与 direct decoder。

## 选定方案

采用两个自包含项目，而不是给 `train_pi05_frs/`、`deploy_pi05/` 增加模式，也不让新目录在运行时 import 现有 FRS 包。

两个项目各自保留同契约的 `DirectTactileActionDecoder` 定义。训练 checkpoint 使用普通 tensor/primitive 字典保存，部署端以 `torch.load(..., weights_only=True)` 和 `load_state_dict(..., strict=True)` 加载。自动化测试用同一个 state dict 和同一组输入验证训练端与部署端逐元素一致。

Pi0.5 与 tactile encoder 都使用现有 JAX/Flax 实现。Tactile encoder 不转换为 PyTorch；cache 生成和在线部署直接加载同一个 `encoder_ckpt_0824`，避免跨框架权重转换误差。PyTorch 只运行小型 Transformer decoder。启动器在导入 JAX 前设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。

## 模型契约

### 输入输出

```text
coarse_action      float32 [B,50,20]
tactile_embedding  float32 [B,4,512]
decoder_output     float32 [B,50,20]
```

Decoder 输出完整的新动作块，不是 `coarse_action + residual`。全部 20 个维度均由 decoder 输出，包括左右夹爪维 9 和 19。

四路 tactile token 顺序固定为：

1. `observation.images.tactile_left_0`
2. `observation.images.tactile_right_0`
3. `observation.images.tactile_left_1`
4. `observation.images.tactile_right_1`

四张图均为当前帧，不使用 tactile history、背景差分或完整 contrastive encoder 的 `future_projection`。每张图独立经过共享 frozen ResNet18，得到一个 512D token。

### Transformer

```text
chunk_size       50
action_dim       20
tactile_dim      512
d_model          128
nhead            4
num_layers       2
dim_feedforward  256
dropout          0.1
activation       ReLU
norm_first       true
```

动作路径：

```text
[B,50,20]
  -> Linear(20,128)
  -> + learned action_position [50,128]
  -> [B,50,128]
```

触觉路径：

```text
[B,4,512]
  -> per-token RMS normalization
  -> Linear(512,128)
  -> + learned sensor_identity [4,128]
  -> [B,4,128]
```

两层 Transformer Decoder 对动作 token 做非因果 self-attention，并让每个动作 token cross-attend 四个 tactile memory token。末端使用 `LayerNorm(128) + Linear(128,20)`，不加输出激活。

## 训练项目

### 目录

```text
train_baseline_pi05/
  README.md
  pyproject.toml
  uv.lock
  configs/train_baseline_pi05.yaml
  scripts/setup_env.sh
  scripts/start_train.sh
  src/lerobot/...
  __init__.py
  config.py
  model.py
  action_cache.py
  prepare_action_cache.py
  tactile_cache.py
  data.py
  checkpoint.py
  train.py
  evaluate.py
  tests/...
```

`src/lerobot` 只包含 Pi0.5 checkpoint、dataset 和 preprocessing 所需的 vendored runtime。训练项目使用自己的 `.venv`，不复用仓库根环境、`train_pi05_frs/.venv` 或部署环境。

### 三阶段流水线

1. 预计算 frozen tactile embedding cache。
2. 运行 frozen Pi0.5 正向采样并生成精简 action cache。
3. 退出 JAX cache 进程，再启动 PyTorch decoder 训练与评估进程。

流水线不生成 `x_base`，不调用 reverse solver，不导入 FRS model/loss/integration 模块。

### Action cache

每个数据集保存：

```text
manifest.json
coarse_actions.npy   float32 [N,50,20]
expert_actions.npy   float32 [N,50,20]
valid_masks.npy      bool    [N,50]
dataset_indices.npy  int64   [N]
episode_indices.npy  int64   [N]
split_ids.npy        uint8   [N]
```

Pi0.5 checkpoint 内部可能存在大于 20 的 padded model width；cache 只保存和监督机器人有效的前 20D，并在 manifest 中分别记录 source model action width 与 decoder action width，禁止含糊推断。

Coarse action 采用与部署一致的固定 `seed=0` noise。批量 cache 生成时，单样本 noise 必须与部署 batch-size 1 的 `seed=0` 数值一致，而不是按 dataset index 改变随机种子。

Manifest 记录足以避免误用 cache 的核心契约：dataset identity、record digest、split、shape、Pi0.5 checkpoint、norm stats、sample steps、noise seed 和 action-space 名称。实现不扩展成通用资产审计系统。

### Tactile cache

Tactile cache 按 dataset absolute frame index 保存：

```text
embeddings  [total_frames,4,512]
```

Manifest 记录 encoder checkpoint、预处理版本、图像尺寸与 tactile key 顺序。Decoder dataset 通过 action cache 的 `dataset_indices` 读取同一当前帧的四个 token。

### Split 与目标

- Episode-level split，seed 42。
- train 80%、validation 10%、test 10%。
- 同一 episode 不跨 split。
- Episode 尾部 action window 允许 padding，但 padding step 由 `valid_masks` 排除。
- Terminal 全零 action 不作为有效 target。
- Expert action 使用与 Pi0.5 checkpoint 相同的 quantile norm stats 归一化。

### 优化

```text
loss             masked Smooth L1, beta=1
batch size       256
epochs           50
optimizer        AdamW
learning rate    3e-4
weight decay     1e-4
betas            (0.9,0.999)
epsilon          1e-8
scheduler        none
gradient clip    none
explicit AMP     none
seed             0
```

仅 decoder 参数进入 optimizer。Pi0.5 和 tactile encoder 不在训练进程中加载，从结构上保证冻结。

每个 epoch 记录 train/validation masked Smooth L1、coarse baseline loss、normalized MSE、physical MAE/RMSE、relative error reduction、delta RMS 和左右夹爪误差。按 validation masked Smooth L1 选择 `best.pt`。训练完成后只对 test split生成一次正式报告，并额外报告 synchronized tactile 与 shuffled tactile 的差异。

### Checkpoint

`best.pt` 至少包含：

```text
checkpoint_schema_version = 1
run_kind = formal
mode = action_tactile
epoch / global_step
decoder_config
decoder_state_dict
metrics
source_contract
```

`decoder_config` 固定记录上述两层、50-step、20D 契约与 tactile key 顺序。`source_contract` 记录 Pi0.5 checkpoint、norm stats、encoder 和 action-cache identity。`last.pt` 额外保存 optimizer state、RNG state 和当前 best 指标，以支持恢复。

保存使用同目录临时文件加 `os.replace`，避免中断时留下半写 checkpoint。

## 部署项目

### 目录

```text
deploy_baseline_pi05/
  README.md
  pyproject.toml
  uv.lock
  configs/deploy_baseline_pi05.yaml
  scripts/start_baseline_pi05.sh
  src/lerobot/...
  __init__.py
  deployment.py
  policy.py
  tactile_encoder.py
  direct_decoder.py
  checkpoint.py
  runtime.py
  bridge_client.py
  protocol.py
  remote_client.py
  tests/...
```

部署 YAML 分别引用 Pi0.5 checkpoint、norm stats、decoder `best.pt` 和 tactile encoder checkpoint，不复制 9GB Pi0.5 权重到 decoder bundle。

### 在线数据流

每个 chunk：

1. 使用当前 RGB/state/task 运行一次 frozen Pi0.5，得到 normalized coarse action `[1,50,20]`。
2. 缓存 coarse action 和 chunk identity。
3. 把 chunk-ready 消息发送给机器人服务器。

每个唯一 action request：

1. 严格检查 chunk、request 和 action index。
2. 使用最新四路 tactile 图像计算 `[1,4,512]` embedding。
3. 使用固定 coarse action 和最新 tactile 运行一次 decoder，得到 `[1,50,20]`。
4. 选择 `decoder_output[0, action_index]`。
5. 使用同一 Pi0.5 norm stats 反归一化并发送 20D robot action。
6. 等待 action acknowledgement 后进入下一 index。

相同 request ID 与相同 payload 重试时返回缓存结果；冲突 payload 或非递增 action index 报错。一个 chunk 只运行一次 Pi0.5；每个新 action request 运行一次 tactile encoder 与 decoder。

### Wire protocol

复用现有 `frs_steering_v1` 的 server-directed chunk/action 消息顺序，以避免要求机器人服务器同时修改。这里仅复用调度协议；新项目的内部类名、日志和 checkpoint mode 均使用 `direct_decoder`，且不导入 FRS 数学实现。

### 必要校验与失败策略

按用户要求保持最小范围，只实现运行必需的检查：

- checkpoint schema、decoder shape/config 和 tactile key 顺序匹配；
- 输入输出 shape 正确且数值 finite；
- chunk/request/action index 顺序有效；
- 可配置的 normalized action absolute limit 与 delta RMS limit。

任何上述错误均停止当前部署，不静默回退 coarse action。不增加额外安全框架、复杂路径审计或与本 baseline 无关的机器人策略检查。

每个 action trace 保存 coarse normalized、corrected normalized、selected normalized/physical、delta RMS、四路 tactile 图像、配置 identity 和推理耗时。

## 配置

训练与部署配置必须显式包含：

- source checkpoint 和 norm stats 路径；
- `action_horizon=50`、`action_dim=20`、`state_dim=20`；
- 两路 Pi0.5 visual camera map；
- 四路 tactile key 顺序；
- `encoder_ckpt_0824` 路径；
- decoder checkpoint/output/cache 路径；
- Pi0.5 sample steps 与固定 noise seed；
- dataset identity、root 与 action key；
- bridge connection、task prompt、control frequency 和日志路径。

`--check` 只做依赖轻量的 YAML、路径与 schema 预检，不初始化 JAX/GPU，不创建 cache/output，也不连接机器人。

## 测试与验收

测试聚焦功能契约，不扩展成泛化安全审计：

1. Decoder 输入输出为 `[B,50,20]`，配置严格为两层 Transformer。
2. 一次 optimizer step 只改变 decoder 参数。
3. Masked Smooth L1 不计 padding step。
4. Action cache 不含 `x_base`，producer 不调用 reverse integration。
5. Cache 的 fixed-noise coarse action 与部署 batch-size 1 推理一致。
6. Episode split 无交叉，action/tactile cache 按 absolute frame index 对齐。
7. `best.pt` 能以 `weights_only=True`、`strict=True` round-trip。
8. 训练端和部署端 decoder 对同一 state dict/input 的输出一致。
9. Cache 与部署对 `encoder_ckpt_0824` 的 embedding 一致。
10. Fake bridge 证明每个 chunk 只调用一次 Pi0.5，每个唯一 action request 使用最新 tactile 并调用一次 decoder。
11. Shape、finite、checkpoint contract 或协议顺序错误会停止且不发送动作。
12. CPU synthetic cache → 单步训练 → save/load → evaluate 端到端 smoke。

服务器交付命令包括：环境安装、`--check`、小样本 cache、单步/单 epoch GPU smoke、正式训练、checkpoint evaluate、部署 `--check` 和有限 iteration 真机前检查。真实 GPU 数据训练与机器人执行不在本地 CPU 验收范围内。

## 非目标

- 不修改或重训 Pi0.5 source policy。
- 不训练 tactile encoder。
- 不实现 FRS reverse/forward integration。
- 不兼容旧 SmolVLA `[20,20]` decoder checkpoint。
- 不加载现有 `checkpoints/ablation/decoder/best.pt` 作为初始化。
- 不增加 residual action 模式。
- 不修改机器人服务器协议实现。
- 不建立通用 checkpoint registry、资产签名服务或额外安全审查框架。
