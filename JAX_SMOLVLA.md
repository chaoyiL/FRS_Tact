# SmolVLA JAX 使用说明

本目录提供 SmolVLA 的纯 JAX 前向、flow-matching 训练、checkpoint 转换、推理和多设备数据并行路径。模型直接读取原 PyTorch safetensors 的 tensor 名称与 `[out, in]` 权重布局，因此转换过程不做有损转置；portable 输出仍是 safetensors，Orbax 是可选格式。

## 安装

```bash
uv sync --extra smolvla-jax
```

运行测试或 PyTorch/JAX 数值对比时安装测试依赖：

```bash
uv sync --extra smolvla-jax-test
.venv/bin/pytest -q tests/jax
```

默认 JAX wheel 可在 CPU 上运行。CUDA/TPU 环境应按目标平台安装对应的 `jax`/`jaxlib` wheel，业务代码不需要变化。

## 1. 转换 checkpoint

本地 checkpoint：

```bash
.venv/bin/python tools/convert_smolvla_pt_to_jax.py \
  --source checkpoints/black-smash-smolvla-40k \
  --output checkpoints/black-smash-smolvla-40k-jax
```

`--source` 也可使用 Hugging Face repo id。默认输出包含：

- `model.safetensors`：JAX 可直接读取的参数；
- `conversion_manifest.json`：tensor 数、参数量、dtype 和输入/输出 SHA-256；
- 原 checkpoint 的模型配置、预处理器、后处理器及归一化统计。

转换默认是逐 tensor 无损复制，可用以下命令只检查 checkpoint：

```bash
.venv/bin/python tools/convert_smolvla_pt_to_jax.py \
  --source checkpoints/black-smash-smolvla-40k \
  --output /tmp/not-used \
  --inspect-only
```

如需要 Orbax 参数目录，增加 `--format orbax`。portable safetensors 更适合跨框架共享和发布。

## 2. 推理

输入 NPZ 至少包含：

- `observation.state`：`[state_dim]` 或 `[B, state_dim]`；
- checkpoint `input_features` 中至少一个相机键；图像可为 HWC/CHW、单张或 batch、uint8 或 float。

缺失相机会按 checkpoint 的 `empty_cameras` 设置补空图。图像 resize/pad、tokenizer、字段重命名、state 归一化、action 反归一化和 Aloha 适配均由 JAX 预处理路径完成。

```bash
.venv/bin/python tools/infer_smolvla_jax.py \
  --checkpoint checkpoints/black-smash-smolvla-40k-jax \
  --observation observation.npz \
  --task "smash the black object" \
  --output actions.npy
```

第一次调用包含 JIT 编译时间。调试时可加 `--no-jit`，或用 `--num-steps 1` 缩短 denoise。远端 checkpoint 需要显式加 `--allow-download`。

Python API：

```python
from lerobot.policies.smolvla_jax import JaxSmolVLAPolicy

policy = JaxSmolVLAPolicy.from_pretrained("checkpoints/black-smash-smolvla-40k-jax")
chunk = policy.predict_action_chunk(observation, "smash the black object", seed=0)
action = policy.select_action(observation, "smash the black object", seed=0)
policy.reset()
```

RTC checkpoint 可通过 `previous_chunk`、`inference_delay` 和 `execution_horizon` 传参；命令行对应 `--previous-chunk`、`--inference-delay` 和 `--execution-horizon`。

## 3. 准备训练数据

当前仓库是 LeRobot 的 inference-only 子集，不包含上游 `lerobot.datasets`。训练入口因此使用一个明确、可审计的 NPZ 桥接格式。

原始 NPZ 需要：

- `task`：`[N]` 字符串数组；
- `observation.state`：`[N, state_dim]`；
- checkpoint 使用的相机键：`[N, H, W, C]` 或 `[N, C, H, W]`；
- `actions`：`[N, chunk_size, action_dim]`；
- 可选 `action_is_pad`：`[N, chunk_size]` bool。

```bash
.venv/bin/python tools/prepare_smolvla_jax_npz.py \
  --checkpoint checkpoints/black-smash-smolvla-40k-jax \
  --input raw_train.npz \
  --output prepared_train.npz
```

输出已经完成图像预处理、tokenize、state/action 归一化，可直接随机 batch。大型数据集应按 shard 生成多个 NPZ；若接回完整版 LeRobot 数据集，可让 dataloader 直接产生同样六个必需字段：`images`、`image_masks`、`language_tokens`、`language_masks`、`state`、`actions`。

## 4. 训练和断点续训

```bash
.venv/bin/python tools/train_smolvla_jax.py \
  --checkpoint checkpoints/black-smash-smolvla-40k-jax \
  --dataset prepared_train.npz \
  --output outputs/smolvla-jax \
  --steps 30000 \
  --batch-size 8 \
  --save-freq 1000
```

训练实现包括与原配置一致的：

- expert/action/state 参数冻结规则；
- global-norm gradient clipping；
- AdamW 与 cosine warmup/decay；
- flow-matching Beta 时间采样和 padding-aware loss；
- JIT train step、确定性 PRNG、模型/optimizer/RNG/step 保存。

恢复时，`--steps` 表示目标总 step：

```bash
.venv/bin/python tools/train_smolvla_jax.py \
  --checkpoint checkpoints/black-smash-smolvla-40k-jax \
  --dataset prepared_train.npz \
  --output outputs/smolvla-jax-resumed \
  --resume outputs/smolvla-jax/checkpoint-00001000 \
  --steps 30000 \
  --batch-size 8
```

多张可见设备上增加 `--data-parallel`。模型和 optimizer state 会复制到设备，batch 的第 0 维沿 `data` mesh 分片；batch size 应能被设备数整除。

## 5. 数值对比

```bash
.venv/bin/python tools/compare_smolvla_backends.py \
  --checkpoint checkpoints/black-smash-smolvla-40k \
  --stage all \
  --num-steps 10
```

`--float32` 用于验证模型语义，BF16 用于观察真实后端舍入差异。当前真实 450,046,176 参数 checkpoint 的本机 CPU 结果：

| 路径 | 模式 | PyTorch/JAX 结果 |
| --- | --- | --- |
| 16 层 joint transformer | FP32 | prefix max abs `1.06e-4`，suffix max abs `1.34e-5`，cosine `1.0` |
| vision tower | FP32 | max abs `6.58e-5`，cosine `0.99999988` |
| 单次完整 denoise velocity | FP32 | max abs `6.56e-6`，cosine `1.0000001` |
| 单次完整 denoise velocity | BF16 | cosine `0.999986` |
| 10-step action sample | BF16 | mean abs `0.00280`，cosine `0.9999588` |

不同 accelerator 的 BF16 舍入可能产生略有差异；FP32 对比用于定位语义错误，端到端 BF16 cosine 用于验证部署一致性。

## 代码入口

- `src/lerobot/policies/smolvla_jax/modeling.py`：vision、connector、VLM/expert、KV cache、flow loss 和采样；
- `src/lerobot/policies/smolvla_jax/preprocessing.py`：图像、token、归一化和 Aloha；
- `src/lerobot/policies/smolvla_jax/checkpoint.py`：safetensors/Orbax 转换与加载；
- `src/lerobot/policies/smolvla_jax/training.py`：optimizer、冻结、JIT train state、保存/恢复；
- `src/lerobot/policies/smolvla_jax/policy.py`：面向部署的 stateful policy。
