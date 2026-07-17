# SmolVLA JAX 使用说明

本目录提供 SmolVLA 的纯 JAX 前向、flow-matching 训练、checkpoint 转换、推理和多设备数据并行路径。模型直接读取原 PyTorch safetensors 的 tensor 名称与 `[out, in]` 权重布局，因此转换过程不做有损转置；portable 输出仍是 safetensors，Orbax 是可选格式。

## 安装

```bash
uv sync
```

默认依赖已包含 SmolVLA 的 PyTorch / JAX 栈，以及复用本机 CUDA 12 的
`jax[cuda12-local]` 插件（不额外安装 CUDA 运行库）。

运行测试：

```bash
.venv/bin/pytest -q tests/jax
```

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

命令行推理直接读取 LeRobotDataset 的 episode/frame。task 默认取自 dataset，也可以用 `--task` 覆盖。缺失相机会按 checkpoint 的 `empty_cameras` 设置补空图；图像 resize/pad、tokenizer、字段重命名、state 归一化、action 反归一化和 Aloha 适配均由 JAX 预处理路径完成。

```bash
.venv/bin/python tools/infer_smolvla_jax.py \
  --checkpoint checkpoints/black-smash-smolvla-40k-jax \
  --dataset-repo-id chaoyi/black_smash_02 \
  --dataset-root ~/.cache/huggingface/lerobot/chaoyi/black_smash_02 \
  --episode 0 \
  --frame 0 \
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

## 3. LeRobotDataset 数据流

JAX trainer 直接构造 `LeRobotDataset` 和 PyTorch `DataLoader`。worker 负责 Parquet、图像和视频读取，主进程负责 tokenize、resize、使用 dataset metadata 中的 mean/std 归一化并转换为 JAX array。

未来 action chunk 由 dataset FPS 和 checkpoint `chunk_size` 自动生成。episode 尾部由 LeRobotDataset 重复边界 action，并产生 `action_is_pad`，loss 会忽略这些 padding。当前标准的 `action` 和部分旧数据集使用的 `actions` 都会自动识别，也可以用 `--action-key` 显式指定。

字段重命名默认读取 checkpoint 的 preprocessor，例如把 dataset 的 `camera0/camera1` 映射为模型的 `camera1/camera2`。需要覆盖时使用 `--rename-map`。

## 4. 训练和断点续训

```bash
.venv/bin/python tools/train_smolvla_jax.py \
  --checkpoint checkpoints/black-smash-smolvla-40k-jax \
  --dataset-repo-id chaoyi/black_smash_02 \
  --dataset-root ~/.cache/huggingface/lerobot/chaoyi/black_smash_02 \
  --output outputs/smolvla-jax \
  --steps 30000 \
  --batch-size 8 \
  --num-workers 4 \
  --save-freq 1000
```

不传 `--dataset-root` 时会按 repo id 从 Hugging Face Hub 获取数据。可用 `--episodes 0 1 2` 限定训练 episode，或用 `--dataset-revision` 固定数据版本。

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
  --dataset-repo-id chaoyi/black_smash_02 \
  --dataset-root ~/.cache/huggingface/lerobot/chaoyi/black_smash_02 \
  --output outputs/smolvla-jax-resumed \
  --resume outputs/smolvla-jax/checkpoint-00001000 \
  --steps 30000 \
  --batch-size 8
```

多张可见设备上增加 `--data-parallel`。模型和 optimizer state 会复制到设备，batch 的第 0 维沿 `data` mesh 分片；batch size 应能被设备数整除。

## 5. Eval 链路

likelihood / 绘图 / t-SNE 依赖已包含在默认 `uv sync` 中。所有 eval 脚本直接读取 JAX
checkpoint 和 `LeRobotDataset`，不再依赖 OpenPI/Pi0 的 config、transform 或
`policy/src`。例如单帧模态 action-error：

```bash
.venv/bin/python eval_scripts-jax/action_error_evaluate.py \
  --checkpoint-dir checkpoints/black-smash-smolvla-40k-jax \
  --dataset-repo-id chaoyi/black_smash_02 \
  --dataset-root ~/.cache/huggingface/lerobot/chaoyi/black_smash_02 \
  --episode-index 0 --frame 0 --remove-modality state
```

概率流 likelihood 使用 Hutchinson-JVP 估计 divergence：

```bash
.venv/bin/python eval_scripts-jax/loglike_evaluate.py \
  --checkpoint-dir checkpoints/black-smash-smolvla-40k-jax \
  --dataset-repo-id chaoyi/black_smash_02 \
  --dataset-root ~/.cache/huggingface/lerobot/chaoyi/black_smash_02 \
  --episode-index 0 --frame 0 --num-steps 120 --remove-modality vision
```

`plot_loglike_modalities.py` 可批量运行多个模态并合图，`action_reverse_tsne.py` 可将真实 action 正向积分到 base noise 后做配对 t-SNE。`eval_scripts-jax/test/k_trace_sweep.py` 和 `eps_trace_sweep.py` 分别用于积分步数与 trace 精度 sweep。

数据集结构检查脚本已放在 `tools/` 下：

```bash
.venv/bin/python tools/inspect_dataset.py \
  --dataset-path ~/.cache/huggingface/lerobot/chaoyi/black_smash_02 \
  --repo-id chaoyi/black_smash_02 --print-text
```

## 6. 数值对比

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
