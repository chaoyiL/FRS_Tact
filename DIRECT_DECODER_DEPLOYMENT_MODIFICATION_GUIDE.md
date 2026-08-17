# SmolVLA Direct Tactile Decoder 真机部署修改指南

本文档针对以下两个实际 checkout：

- Policy client：`/home/yunjing/vb3/FRS_Tact`，当前分支 `eric`。
- Robot server：`/home/yunjing/vb3/vb3_robot_server`，当前分支 `main`。

目标是部署 VB3 中已经训练完成的 direct tactile action decoder：冻结 SmolVLA 先产生归一化粗动作，冻结触觉 ResNet18 产生四个触觉 embedding，小型 Transformer decoder 直接重新生成完整动作块。

本文档描述需要实施的代码修改，不表示这些修改已经完成，也不表示模型已经通过真机验证。

## 1. 结论

推荐在 `FRS_Tact` 的同一个远程 policy client 进程中运行：

- JAX SmolVLA：产生归一化 coarse action。
- PyTorch ResNet18：编码四路当前触觉图像。
- PyTorch `DirectTactileActionDecoder`：产生归一化 fine action。
- JAX checkpoint 保存的 action statistics：只对 fine action 反归一化一次。

```text
Robot server
  │
  │  two RGB + four tactile RGB + state[20] + task
  ▼
FRS_Tact policy client
  ├─ JAX SmolVLA (pick_tube_01 exact merged checkpoint)
  │    └─ normalized coarse action [1,20,20]
  ├─ PyTorch frozen tactile ResNet18
  │    └─ tactile embedding [1,4,512]
  ├─ PyTorch DirectTactileActionDecoder
  │    └─ normalized fine action [1,20,20]
  └─ action unnormalizer, exactly once
       └─ physical action [20,20]
              │
              ▼
       existing robot-bridge-v1
              │
              ▼
Robot server validates full chunk and schedules fresh actions
```

不要把本模型接入 `FRSRuntime`。`FRSRuntime` 是逐动作 Flow Re-Steering 系统，需要 reverse integration 和 `frs_steering_v1` 协议；本模型是整块 `[20,20]` direct decoder，应继续使用现有普通 `send_action`/`action_ack` 协议。

## 2. 不能直接使用当前部署配置的原因

当前 `deploy_smolvla/configs/deploy_smolvla_jax.yaml` 至少有四项不符合 decoder 训练契约：

1. 当前 checkpoint 是 `pick_tube_02_3w_jax`，而 decoder 使用 `pick_tube_01` frozen producer 训练。
2. 当前 `observation.data_type` 是 `vision`，robot server 不会发布四路 tactile。
3. 当前 `action_horizon` 是 10，而 decoder 固定输出 20 步。
4. 当前普通推理循环使用 `seed + iteration`；decoder 的训练 cache 使用固定 PyTorch seed-0 noise，每一帧都使用相同 noise template。

另外，`FRS_Tact` 的 checkpoint loader 只接受 merged JAX SmolVLA checkpoint，不能把 PyTorch `best.pt` 当作 JAX checkpoint 加载。`best.pt` 只包含 decoder 的 PyTorch state dict 和实验 metadata。

## 3. 必须保持的训练契约

| 项目 | 固定值 |
|---|---|
| SmolVLA base | `lerobot/smolvla_base` |
| SmolVLA adapter | `KaiyueChen/pick_tube_01` |
| adapter revision | `c2bb4296cf7405ac3c0ad89e6f577fa620a660a6` |
| tactile encoder | `KaiyueChen/encoder_ckpt_0809` |
| encoder revision | `450aa60963cde9540bd6c8047bf2529eff1def37` |
| decoder checkpoint mode | `action_tactile` |
| decoder checkpoint run kind | `formal` |
| action horizon | 20 |
| action dimension | 20 |
| tactile count | 4 |
| tactile embedding dimension | 512 |
| decoder hidden dimension | 128 |
| Transformer decoder layers | 2 |
| fixed noise seed | PyTorch CPU generator seed 0 |
| fixed noise shape | `[1,20,32]`，前20维随机、后12维补零 |

四路触觉顺序必须严格固定：

```text
observation.images.tactile_left_0
observation.images.tactile_right_0
observation.images.tactile_left_1
observation.images.tactile_right_1
```

正式 decoder checkpoint 当前为：

```text
/home/yunjing/outputs/tactile_action_decoder/formal-20260815-154648/best.pt
```

触觉 encoder 当前为：

```text
/home/yunjing/assets/tactile_action_decoder/encoder.safetensors
/home/yunjing/assets/tactile_action_decoder/encoder.json
```

`best.pt` 加载时至少检查：

```text
checkpoint_schema_version == 1
run_kind == "formal"
mode == "action_tactile"
decoder_config.chunk_size == 20
decoder_config.action_dim == 20
decoder_config.tactile_dim == 512
decoder_config.tactile_keys == fixed four-key order
decoder_state_dict exists
```

不能使用 `smoke/best.pt` 进行真机部署。

## 4. 推荐的 FRS_Tact 文件结构

在 `FRS_Tact` 增加独立模块，不修改 `FRSRuntime`：

```text
deploy_smolvla/
├── direct_decoder/
│   ├── __init__.py
│   ├── configuration.py
│   ├── model.py
│   ├── tactile_resnet.py
│   ├── runtime.py
│   └── bundle.py
├── configs/
│   └── deploy_direct_decoder.yaml
├── scripts/
│   └── start_direct_decoder.sh
└── remote_client.py
```

职责划分：

- `configuration.py`：固定 20/20/4/512/128/2 等结构契约。
- `model.py`：纯 PyTorch `DirectTactileActionDecoder`。
- `tactile_resnet.py`：训练时使用的 frozen ResNet18 和完全相同的预处理。
- `bundle.py`：加载并校验 decoder、encoder、noise 和 manifest。
- `runtime.py`：将 raw tactile 转成 embedding，并用 decoder refine normalized coarse action。
- `remote_client.py`：保留 JAX SmolVLA、WebSocket、warmup、ACK 和日志，只增加 direct-decoder backend 分支。

不要让 FRS_Tact 在部署时通过 `PYTHONPATH` 导入整个 `/home/yunjing/vb3/src`。FRS_Tact 自带的是精简 JAX `lerobot`，与 VB3 使用的 upstream PyTorch LeRobot policy package 存在同名包边界。应将两个小型纯 PyTorch 模块移植到 `deploy_smolvla/direct_decoder/`，并用 parity test 固定其行为。

## 5. 新增 `DirectDecoderRuntime`

建议接口：

```python
class DirectDecoderRuntime:
    tactile_keys: tuple[str, ...]

    @classmethod
    def from_bundle(
        cls,
        bundle_root: Path,
        *,
        device: str | torch.device,
    ) -> "DirectDecoderRuntime": ...

    def reset(self) -> None: ...

    @torch.inference_mode()
    def refine(
        self,
        coarse_normalized: np.ndarray,
        observation: Mapping[str, Any],
    ) -> np.ndarray:
        """Return float32 normalized fine action shaped [1,20,20]."""
```

### 5.1 触觉预处理

`refine()` 应从 observation 按固定顺序读取四路当前帧：

```python
images = [observation[key] for key in self.tactile_keys]
```

输入允许 HWC RGB `uint8`，或者数值范围 `[0,1]` 的 float RGB。训练时的准确预处理为：

1. 转为 RGB `uint8`。
2. 保持长宽比缩放，使图像不超过 `224×224`。
3. 中心黑边补齐至 `224×224`。
4. 除以 255，得到 float32 NCHW。
5. 不使用 ImageNet mean/std。
6. 不使用背景帧差分。
7. 只使用当前触觉帧。

批量 shape：

```text
four raw images              [4,H,W,3]
preprocessed                 [4,3,224,224]
shared frozen ResNet18       [4,512]
per-sensor RMS normalize     [4,512]
add batch dimension          [1,4,512]
```

RMS normalization：

```python
x = x / torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True) + 1e-6)
```

注意：训练 cache 生成阶段对 encoder embedding 做过 RMS normalize，decoder forward 还会再次 RMS normalize。部署应保持当前模型实现，不要擅自删除其中任何一次。

### 5.2 Decoder forward

```python
coarse = torch.from_numpy(coarse_normalized).to(
    device=self.device,
    dtype=torch.float32,
)
tactile = self.encode_tactile(observation)
fine = self.decoder(coarse, tactile, mode="action_tactile")
```

必须验证：

```python
assert fine.shape == (1, 20, 20)
assert torch.isfinite(fine).all()
```

返回 CPU float32 NumPy：

```python
return fine.detach().cpu().numpy().astype(np.float32, copy=False)
```

Decoder 是 direct output：

```text
fine_action = decoder(coarse_action, tactile)
```

禁止写成：

```text
fine_action = coarse_action + decoder(...)
```

## 6. 修改 `deploy_smolvla/remote_client.py`

### 6.1 配置解析

增加 root-level backend：

```yaml
backend: direct_tactile_decoder
```

增加 section：

```yaml
direct_decoder:
  bundle: /home/typhon/FRS_Tact/checkpoints/direct_decoder_bundle
  device: cuda:0
```

`load_config()` 在 backend 为 `direct_tactile_decoder` 时必须：

- 要求 `direct_decoder.bundle` 和 `direct_decoder.device`。
- 要求 `observation.data_type == "vitac"`。
- 要求 `control.action_horizon == 20`。
- 拒绝启用 `frs`。
- 拒绝 RTC action stitching；本实验训练时没有 previous-chunk/RTC 条件。

普通 JAX SmolVLA 和现有 FRS backend 行为保持不变。

### 6.2 Policy 选择

Direct decoder 使用 visual-only `JaxSmolVLAPolicy` 作为 coarse producer。不要因为 observation 包含 tactile 就改成 `VTJaxSmolVLAPolicy`；这次实验的触觉不进入 SmolVLA transformer。

JAX checkpoint 必须是由同一个 `pick_tube_01` adapter 合并得到的 visual checkpoint。其 contract 应为：

```text
state_dim=20
action_dim=20
chunk_size=20
image_keys=camera1,camera2 after rename
tactile_num_tokens=0
```

### 6.3 Observation keys

普通 visual policy 的 `policy.config.image_keys` 仍只有两路 RGB。Direct decoder 分支需要额外添加：

```python
DIRECT_TACTILE_KEYS = (
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)

robot_tactile_keys = DIRECT_TACTILE_KEYS
robot_image_keys = tuple(
    dict.fromkeys((*robot_image_keys, *robot_tactile_keys))
)
```

这样现有 `_validate_observation()`、`_prepare_observation()` 和 `ObservationSaver` 可以继续复用，并将四路 tactile 视为 required image keys。

### 6.4 修改 `_predict_chunk()` 的 normalized boundary

推荐扩展签名：

```python
def _predict_chunk(
    policy: JaxSmolVLAPolicy,
    observation: Mapping[str, Any],
    task: str,
    *,
    seed: int,
    jit: bool,
    num_steps: int | None,
    previous_chunk: np.ndarray | None,
    inference_delay: int | None,
    execution_horizon: int | None,
    direct_decoder: DirectDecoderRuntime | None = None,
) -> tuple[np.ndarray, np.ndarray]:
```

核心流程改为：

```python
noise = None
if direct_decoder is not None:
    noise = direct_decoder.fixed_noise_jax

coarse_normalized = policy.predict_action_chunk(
    observation,
    task,
    seed=seed,
    noise=noise,
    jit=jit,
    normalized=True,
    num_steps=num_steps,
    previous_chunk=None,
    inference_delay=None,
    execution_horizon=None,
)
jax.block_until_ready(coarse_normalized)

if direct_decoder is None:
    final_normalized = np.asarray(coarse_normalized, dtype=np.float32)
else:
    final_normalized = direct_decoder.refine(
        np.asarray(coarse_normalized, dtype=np.float32),
        observation,
    )

physical = policy.preprocessor.unnormalize_actions(final_normalized)
```

最后验证：

```python
physical = np.asarray(physical, dtype=np.float32)
final_normalized = np.asarray(final_normalized, dtype=np.float32)
assert physical.shape == (1, 20, 20)
assert final_normalized.shape == (1, 20, 20)
assert np.isfinite(physical).all()
assert np.isfinite(final_normalized).all()
return physical[0], final_normalized[0]
```

必须只有这里的一次 `unnormalize_actions(final_normalized)`。

禁止：

- 先对 coarse action 反归一化再输入 decoder。
- 对 decoder 输出调用两次 unnormalize。
- 将 physical action 输入 decoder。
- 在 direct decoder 模式中继续使用 `seed + iteration` 产生变化 noise。

### 6.5 `run()` 修改

加载 JAX visual policy 后：

```python
backend = str(config.get("backend", "jax_smolvla"))
if backend == "direct_tactile_decoder":
    direct_decoder = DirectDecoderRuntime.from_bundle(...)
else:
    direct_decoder = None
```

warmup 和正式循环都必须传入同一个 runtime：

```python
_predict_chunk(..., direct_decoder=direct_decoder)
```

正式循环中 direct decoder 使用固定 noise，因此不要把 `seed + iteration` 当作 coarse noise：

```python
predict_seed = seed if direct_decoder is not None else seed + iteration
```

`direct_decoder.reset()` 应在 episode 开始前调用。当前 runtime 无时序状态，但保留接口便于与 client 生命周期一致。

发送动作部分不变：

```python
bridge.send_action(action, obs_seq, trace=trace)
bridge.receive_action_ack(obs_seq, timeout=action_ack_timeout_s)
```

## 7. 固定 noise 文件

PyTorch seed 0 和 JAX seed 0 不会生成相同数值。必须在部署 bundle 中保存训练时的 PyTorch noise，而不是仅在 YAML 写 `seed: 0`。

生成一次：

```bash
cd /home/yunjing/vb3
source .venv/bin/activate

python - <<'PY'
from pathlib import Path
import numpy as np
import torch

output = Path("/home/yunjing/assets/tactile_action_decoder/fixed_noise.npy")
generator = torch.Generator(device="cpu")
generator.manual_seed(0)
noise = torch.randn((1, 20, 20), generator=generator, dtype=torch.float32)
noise = torch.nn.functional.pad(noise, (0, 12))
assert noise.shape == (1, 20, 32)
output.parent.mkdir(parents=True, exist_ok=True)
np.save(output, noise.numpy())
print(output)
PY
```

Bundle loader 必须验证：

```text
dtype=float32
shape=[1,20,32]
all finite
noise[:,:,20:32] exactly zero
SHA256 matches bundle_manifest.json
```

加载后转换一次：

```python
self.fixed_noise_jax = jax.device_put(jnp.asarray(noise, dtype=jnp.float32))
```

## 8. 生成匹配的 pick_tube_01 JAX checkpoint

不能沿用当前 `pick_tube_02_3w_jax`。

在 FRS_Tact 环境中执行：

```bash
cd /home/yunjing/vb3/FRS_Tact
source .venv/bin/activate

python tools/merge_smolvla_peft_to_jax.py \
  --base /home/yunjing/assets/smolvla_base/c83c3163b8ca9b7e67c509fffd9121e66cb96205 \
  --adapter /home/yunjing/assets/pick_tube_01 \
  --adapter-revision c2bb4296cf7405ac3c0ad89e6f577fa620a660a6 \
  --output /home/yunjing/assets/pick_tube_01_jax \
  --no-allow-download
```

输出目录至少需要：

```text
model.safetensors
config.json
conversion_manifest.json
policy_preprocessor.json
policy_postprocessor.json
policy_preprocessor_step_5_normalizer_processor.safetensors
policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

必须先做同帧 coarse parity：

1. 使用完全相同的两路 RGB、state、task。
2. PyTorch 和 JAX 都使用 bundle 中的固定 `[1,20,32]` noise。
3. 两边都输出 normalized coarse action。
4. 对比 shape、finite、max absolute error、mean absolute error。
5. 若 JAX coarse 与构建训练 cache 时的 PyTorch coarse 不满足事先约定的数值等价阈值，则不要继续真机部署；改用完整 PyTorch client。

第一版可使用实验已接受的数值等价门限作为起点：

```python
np.testing.assert_allclose(jax_coarse, torch_coarse, rtol=2e-3, atol=2e-3)
```

必须在实际 checkpoint 和真实 observation 上记录结果，不能仅靠静态代码认为二者等价。

## 9. 部署 bundle

推荐目录：

```text
direct_decoder_bundle/
├── bundle_manifest.json
├── smolvla_jax/
│   ├── model.safetensors
│   ├── config.json
│   ├── conversion_manifest.json
│   ├── policy_preprocessor.json
│   ├── policy_postprocessor.json
│   ├── policy_preprocessor_step_5_normalizer_processor.safetensors
│   └── policy_postprocessor_step_0_unnormalizer_processor.safetensors
├── tactile/
│   ├── encoder.safetensors
│   └── encoder.json
├── decoder/
│   └── best.pt
└── noise/
    └── fixed_noise.npy
```

`bundle_manifest.json` 至少记录：

```json
{
  "schema_version": 1,
  "backend": "jax_smolvla_plus_torch_direct_decoder",
  "smolvla_base_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
  "smolvla_adapter_revision": "c2bb4296cf7405ac3c0ad89e6f577fa620a660a6",
  "tactile_encoder_revision": "450aa60963cde9540bd6c8047bf2529eff1def37",
  "decoder_mode": "action_tactile",
  "decoder_run_kind": "formal",
  "chunk_size": 20,
  "action_dim": 20,
  "tactile_dim": 512,
  "tactile_keys": [
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1"
  ],
  "noise_shape": [1, 20, 32],
  "noise_dtype": "float32",
  "noise_seed_source": "torch_cpu_generator_manual_seed_0",
  "action_boundary": {
    "smolvla_to_decoder": "adapter_normalized_mean_std",
    "decoder_to_robot": "physical_after_single_unnormalize"
  },
  "files": {}
}
```

`files` 中为 bundle 内每个文件记录相对路径、SHA256 和 size。还应复制 `best.pt` 内的 cache/split/producer hashes，作为实验 provenance；部署时不需要携带或扫描完整训练 cache。

不要直接复用 VB3 当前 `FrozenSmolVLATactileDecoderPolicy.from_pretrained_assets()`，因为它加载和首次推理时都会验证正式训练 cache。训练 cache 适合实验审计，不适合真机启动依赖。

## 10. 新部署 YAML

建议新增：

```yaml
# deploy_smolvla/configs/deploy_direct_decoder.yaml

backend: direct_tactile_decoder

checkpoint: /home/typhon/FRS_Tact/checkpoints/direct_decoder_bundle/smolvla_jax
revision: null
allow_download: false
seed: 0
jit: true
num_steps: null
checkpoint_contract: {}

rename_map:
  observation.images.camera0: observation.images.camera1
  observation.images.camera1: observation.images.camera2

direct_decoder:
  bundle: /home/typhon/FRS_Tact/checkpoints/direct_decoder_bundle
  device: cuda:0

connection:
  address: 127.0.0.1
  port: 26421
  add_port: true
  retry_interval_s: 1.0
  ping_interval_s: 20.0
  ping_timeout_s: 20.0
  observation_timeout_s: 30.0
  action_ack_timeout_s: 30.0
  token: null
  token_env: VB_ROBOT_TOKEN
  require_token: true

observation:
  data_type: vitac
  language_prompt: Use the left hand to pick up the green tube, and then use the right hand to pick up the blue tube.
  single_arm_mode: false
  no_state_obs_mode: false

control:
  control_frequency: 30.0
  controller_frequency: 80.0
  action_horizon: 20
  # 保持当前 robot server 安全默认时写 5；首次真机动作测试写 1。
  steps_per_inference: 5
  inference_delay: null
  execution_horizon: null

runtime:
  auto_start: false
  warmup_runs: 2
  status_interval_s: 2.0
  max_iterations: 0

logging:
  save_observations: true
  output_dir: outputs/direct_decoder_observations
  save_every: 1
  queue_size: 32
```

不要配置 `frs` section，或者明确写：

```yaml
frs:
  enabled: false
```

## 11. 启动脚本和 JAX/PyTorch 显存共存

新增 `deploy_smolvla/scripts/start_direct_decoder.sh`，复用现有 `start_remote_client.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${DIRECT_DECODER_CONFIG:-${ROOT}/deploy_smolvla/configs/deploy_direct_decoder.yaml}"

# 防止 JAX 预占绝大部分 GPU 显存，导致 PyTorch encoder/decoder OOM。
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

exec bash "${ROOT}/deploy_smolvla/scripts/start_remote_client.sh" \
  --config "${CONFIG}" "$@"
```

第一版使用 NumPy 作为 JAX→PyTorch 的边界即可：coarse action 只有 400 个 float。不要一开始加入 DLPack，以免把 CUDA stream/lifetime 问题引入首版真机部署。

运行时应打印并记录：

```text
JAX backend
Torch device
SmolVLA checkpoint path/hash
decoder checkpoint path/hash
encoder path/hash
fixed noise hash
observation keys
chunk/action dimensions
JAX inference time
tactile encoder time
decoder time
total inference time
```

## 12. vb3_robot_server 需要和不需要修改的地方

### 12.1 不需要修改

Robot server 已经具备：

- 两台复合 UVC 相机采集。
- 每帧三等分裁剪。
- 四路 tactile 的字段命名。
- HWC RGB uint8 输出。
- `vitac` config 分支。
- msgpack ndarray 透明传输。
- 普通 `[20,20]` action chunk 协议。
- 完整 chunk 的 finite/位移/旋转/gripper 安全校验。
- obs_seq 匹配和 action ACK。

因此首版无需修改：

```text
real_world/bimanual_umi_env.py
real_world/real_inference_util.py
client/robot_client.py
client/msgpack_numpy.py
deploy_scripts/vbvla_safety.py
```

服务端 `data_type` 由 policy client 握手发送，不是单独的 server YAML。只要新的 client config 使用 `observation.data_type: vitac`，server 就会进入四路触觉路径。

### 12.2 触觉相机现有契约

默认硬件：

```text
/dev/video0 = left_hand
/dev/video2 = right_hand
MJPG, 3840x800, 30 fps
```

每台相机的复合画面：

```text
[left tactile 1280x800 | visual 1280x800 | right tactile 1280x800]
```

现有代码只对 left tactile 旋转 180 度，然后三个 panel 都做 BGR→RGB、按比例 resize/center crop 到 `256×256`。

上线前必须人工确认：

- 真机确实输出 3840×800 三联画面。
- panel 顺序与训练 `pick_tube_05` 一致。
- 仅左触觉旋转 180 度与训练一致。
- `/dev/video0`、`/dev/video2` 对应左右手没有交换。
- 曝光、白平衡、gain、gamma 与采集训练数据时一致。
- 四路 tactile 没有黑屏、重复画面或明显错帧。

### 12.3 真实执行步数

Decoder config 中记录 `execute_steps=10`，但当前 robot server 默认：

```text
steps_per_inference = 5
max_executed_actions = 5
```

服务端会执行：

```python
effective_max = min(
    server_steps_per_inference,
    client_steps_per_inference,
    max_executed_actions,
)
```

当前默认即使 client 请求 10，也最多选择 5 个动作。

而且 RTC 模式不是简单执行 `chunk[:5]`：服务端先为 20 步生成时间戳，丢弃到达时已经过期的动作，再从剩余 future actions 选最多 5 个。因此每轮实际 schedule 数量为 0–5。

推荐首轮真机测试：

```yaml
steps_per_inference: 1
```

并用 server CLI：

```bash
--max-executed-actions 1
```

确认动作方向后，再恢复当前安全默认 5。

如果实验明确要求每轮最多 10 个未来有效动作，则必须同时修改：

```python
# vb3_robot_server/configs/server_config.py
max_executed_actions = 10
steps_per_inference = 10
```

或者 `max_executed_actions` 使用 CLI 覆盖为10，但 `SERVER_CONFIG.steps_per_inference` 仍必须改为10。即便如此，RTC 仍会丢弃 stale actions，不能声称严格执行原 chunk 的前10步。

## 13. 最小验证顺序

不要直接从离线 checkpoint 跳到自动真机运行。

### 阶段 A：资产和 shape smoke

不连接 robot server：

1. 加载 bundle。
2. 检查 decoder 为 formal/action_tactile。
3. 检查 encoder/decoder/noise hash。
4. 用六张 synthetic uint8 RGB 图和 20D state 运行一次。
5. 输出必须是 finite float32 `[20,20]`。

### 阶段 B：同帧数值一致性

取一条 `pick_tube_05` 或保存的真实 observation：

1. 原 VB3 PyTorch pipeline 运行 frozen SmolVLA + encoder + decoder。
2. FRS_Tact hybrid pipeline 运行 JAX SmolVLA + 相同 PyTorch encoder/decoder。
3. 对比 normalized coarse、tactile embedding、normalized fine、physical action。
4. 如果 coarse parity 失败，停止 hybrid 部署，使用全 PyTorch client。

### 阶段 C：真实相机、禁止机械臂动作

连接 robot server，但不发送 START：

1. 收一帧 observation。
2. 保存六路图像。
3. 核对字段名、shape、dtype、RGB颜色和传感器顺序。
4. 完成 warmup 并记录总延迟。
5. 推理总延迟应明显小于下一轮重规划时间预算。

### 阶段 D：单动作真机

```text
client steps_per_inference=1
server --max-executed-actions 1
runtime.auto_start=false
```

每轮人工确认后才发送 START。检查：

- 左右臂没有交换。
- 相对位姿方向正确。
- rotation-6D 有效。
- gripper 方向和范围正确。
- ACK 对应相同 obs_seq。
- 推理延迟不会导致全部 action stale。

### 阶段 E：恢复最多5步

单步通过后再设置 `steps_per_inference=5`。只有论文实验确实需要 10 步时，才修改 server 两个上限并重新做安全验证。

## 14. 启动示例

Robot server：

```bash
cd /home/yunjing/vb3/vb3_robot_server
source .venv/bin/activate
export VB_ROBOT_TOKEN='真实Token'

# 首次动作验证只允许调度1步
bash scripts/bimanual_smolvla.sh --max-executed-actions 1
```

Policy client：

```bash
cd /home/yunjing/vb3/FRS_Tact
source .venv/bin/activate
export VB_ROBOT_TOKEN='与服务端一致的Token'
export DIRECT_DECODER_CONFIG=/home/yunjing/vb3/FRS_Tact/deploy_smolvla/configs/deploy_direct_decoder.yaml

bash deploy_smolvla/scripts/start_direct_decoder.sh --check
bash deploy_smolvla/scripts/start_direct_decoder.sh
```

首版必须保持：

```yaml
runtime:
  auto_start: false
```

## 15. 不推荐的方案

### 15.1 把 decoder 接入 `FRSRuntime`

不推荐。FRS 使用逐动作 steer request、reverse integration 和独立 checkpoint，算法与本 direct decoder 不同。

### 15.2 直接使用 `VTJaxSmolVLAPolicy`

不推荐。这会把 tactile token 注入 SmolVLA transformer，改变消融实验边界。当前 decoder 实验的 SmolVLA 必须保持 visual-only frozen producer。

### 15.3 继续使用 `pick_tube_02_3w_jax`

禁止。Decoder 训练时看到的是 `pick_tube_01` producer 分布。

### 15.4 只写 `seed: 0`，但让 JAX 自己采样 noise

禁止。JAX PRNG 与 PyTorch CPU generator 的 seed 0 数值不同。

### 15.5 直接部署完整训练 cache loader

不推荐。真机不应依赖或扫描约1.5GB训练 cache；部署 bundle 应使用文件 hash 和 checkpoint metadata。

### 15.6 第一版就把全部模块转换成 JAX

不推荐。47万参数 decoder 虽然可以转换，但需要正确拆分 PyTorch MultiheadAttention 的 Q/K/V、处理权重转置、pre-norm、LayerNorm epsilon、FFN 和 parity。先完成 hybrid 数值对齐，再决定是否优化为全 JAX。

## 16. 文件级实施清单

### FRS_Tact：必须新增

```text
deploy_smolvla/direct_decoder/__init__.py
deploy_smolvla/direct_decoder/configuration.py
deploy_smolvla/direct_decoder/model.py
deploy_smolvla/direct_decoder/tactile_resnet.py
deploy_smolvla/direct_decoder/bundle.py
deploy_smolvla/direct_decoder/runtime.py
deploy_smolvla/configs/deploy_direct_decoder.yaml
deploy_smolvla/scripts/start_direct_decoder.sh
```

### FRS_Tact：必须修改

```text
deploy_smolvla/remote_client.py
```

修改点：backend config、direct runtime load、vitac contract、四路 required tactile keys、fixed noise、normalized boundary、单次 unnormalize、warmup和正式循环。

### FRS_Tact：无需修改

```text
deploy_smolvla/frs_runtime.py
deploy_smolvla/frs_protocol.py
deploy_smolvla/bridge_client.py
train_vtsmolvla/*
```

### vb3_robot_server：首版无需修改

```text
real_world/bimanual_umi_env.py
real_world/real_inference_util.py
client/robot_client.py
client/msgpack_numpy.py
deploy_scripts/bimanual_smolvla_online.py
```

### vb3_robot_server：只有10步需求才修改

```text
configs/server_config.py
```

## 17. 完成标准

只有满足以下条件，才能进入真机动作测试：

- 使用 exact `pick_tube_01` merged JAX checkpoint。
- JAX/PyTorch normalized coarse parity 已有真实样本记录。
- 使用正式 `action_tactile` decoder checkpoint，不是 smoke/action-only。
- fixed noise 与训练 cache 完全一致。
- 四路 tactile 名称、顺序、RGB方向和预处理完全一致。
- decoder 输入输出均为 normalized action。
- physical action 只反归一化一次。
- 输出为 finite float32 `[20,20]`。
- FRS 和 RTC 在 direct decoder backend 中禁用。
- 六路真实图像已经人工核对。
- warmup/推理延迟已经记录。
- 首次机械臂测试限制为每轮最多1个动作。

离线 test MAE 改善、协议 dry-run、相机预览和单步动作测试都不能单独证明真机任务成功率。

## 18. 代码依据

FRS_Tact：

- `deploy_smolvla/remote_client.py`：JAX policy load、observation contract、normalized inference、ordinary action loop。
- `deploy_smolvla/frs_runtime.py`：现有 FRS 是另一套 reverse/steering 算法。
- `deploy_smolvla/bridge_client.py`：普通 chunk send/ack 协议。
- `train_smolvla/policy.py`：外部 noise、normalized output 和 JAX inference。
- `train_smolvla/preprocessing.py`：state/action normalization 与 unnormalization。
- `tools/merge_smolvla_peft_to_jax.py`：base+PEFT 合并及 processor/manifest 复制。

VB3 decoder：

- `../src/vb3/policies/tactile_action_decoder/configuration.py`
- `../src/vb3/policies/tactile_action_decoder/model.py`
- `../src/vb3/policies/tactile_action_decoder/tactile_resnet.py`
- `../src/vb3/policies/tactile_action_decoder/frozen_smolvla.py`
- `../src/vb3/policies/tactile_action_decoder/policy.py`
- `../src/vb3/train/tactile_action_decoder.py`

Robot server：

- `../vb3_robot_server/configs/server_config.py`
- `../vb3_robot_server/real_world/bimanual_umi_env.py`
- `../vb3_robot_server/real_world/real_inference_util.py`
- `../vb3_robot_server/deploy_scripts/bimanual_smolvla_online.py`
- `../vb3_robot_server/deploy_scripts/vbvla_safety.py`
