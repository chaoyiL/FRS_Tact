# Direct Tactile Decoder Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the released direct tactile decoder through the existing JAX SmolVLA remote client while preserving the ordinary robot bridge protocol.

**Architecture:** A new self-contained PyTorch runtime loads `checkpoints/ablation`, encodes four current tactile frames, and refines the normalized action chunk produced by the existing visual-only JAX policy. `remote_client.py` selects this path through a new backend value and performs the existing action unnormalization exactly once after refinement.

**Tech Stack:** Python 3.11, JAX, PyTorch, safetensors, NumPy, OpenCV, PyYAML, pytest, Bash.

## Global Constraints

- Keep `deploy_smolvla/scripts/start_vtsmolvla.sh`, `start_frs.sh`, `FRSRuntime`, and the robot server unchanged.
- Use `checkpoints/model/pick_tube_01_jax` as the visual-only coarse-action checkpoint.
- Use `checkpoints/ablation` directly as the decoder asset root.
- Decoder contract is fixed at chunk/action/tactile dimensions `20/20/512`, four tactile sensors, model width 128, four heads, two layers, and FFN width 256.
- Tactile key order is left-0, right-0, left-1, right-1.
- Decoder output is a complete normalized fine action, not a residual.
- Unnormalize the fine action exactly once.
- Use one saved PyTorch CPU seed-0 noise array shaped `[1,20,32]`; never use `seed + iteration` for this backend.
- Use `steps_per_inference: 10`, matching the released checkpoint metadata.
- Do not add broad test-suite, code-review, robot-server, or real-robot validation work.

---

## File map

- Create `deploy_smolvla/direct_decoder.py`: PyTorch model definitions, tactile preprocessing, asset loading, and runtime refinement API.
- Modify `deploy_smolvla/remote_client.py`: backend validation and normalized-boundary integration.
- Create `deploy_smolvla/configs/deploy_direct_decoder.yaml`: dedicated deployment configuration.
- Create `deploy_smolvla/scripts/start_direct_decoder.sh`: launcher that reuses `start_remote_client.sh`.
- Create `tests/jax/test_direct_decoder_deployment.py`: focused runtime/backend/launcher tests only.
- Generate ignored runtime asset `checkpoints/ablation/fixed_noise.npy`.

### Task 1: Direct decoder runtime and fixed noise

**Files:**
- Create: `deploy_smolvla/direct_decoder.py`
- Create: `tests/jax/test_direct_decoder_deployment.py`
- Generate: `checkpoints/ablation/fixed_noise.npy`

**Interfaces:**
- Produces: `DIRECT_TACTILE_KEYS: tuple[str, ...]`
- Produces: `DirectDecoderRuntime.from_bundle(bundle_root: Path, *, device: str | torch.device) -> DirectDecoderRuntime`
- Produces: `DirectDecoderRuntime.refine(coarse_normalized: np.ndarray, observation: Mapping[str, Any]) -> np.ndarray`
- Produces: `DirectDecoderRuntime.fixed_noise_jax`
- Produces: `DirectDecoderRuntime.reset() -> None`

- [ ] **Step 1: Add focused failing tests for the released asset contract**

```python
from pathlib import Path

import numpy as np
import pytest
import torch

from deploy_smolvla.direct_decoder import (
    DIRECT_TACTILE_KEYS,
    DirectDecoderRuntime,
    DirectTactileActionDecoder,
)

ROOT = Path(__file__).resolve().parents[2]
ABLATION = ROOT / "checkpoints" / "ablation"


def test_released_decoder_state_loads_strictly() -> None:
    checkpoint = torch.load(
        ABLATION / "decoder" / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = DirectTactileActionDecoder.from_config(checkpoint["decoder_config"])
    model.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    assert sum(parameter.numel() for parameter in model.parameters()) == 471_828


def test_fixed_noise_matches_training_contract() -> None:
    noise = np.load(ABLATION / "fixed_noise.npy", allow_pickle=False)
    assert noise.dtype == np.float32
    assert noise.shape == (1, 20, 32)
    assert np.isfinite(noise).all()
    np.testing.assert_array_equal(noise[:, :, 20:], 0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="deployment uses cuda:0")
def test_runtime_refine_returns_finite_normalized_chunk() -> None:
    runtime = DirectDecoderRuntime.from_bundle(ABLATION, device="cuda:0")
    observation = {
        key: np.zeros((240, 320, 3), dtype=np.uint8)
        for key in DIRECT_TACTILE_KEYS
    }
    result = runtime.refine(np.zeros((1, 20, 20), dtype=np.float32), observation)
    assert result.shape == (1, 20, 20)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
```

- [ ] **Step 2: Run the tests and confirm the missing runtime/noise failure**

Run:

```bash
.venv/bin/python -m pytest tests/jax/test_direct_decoder_deployment.py -v
```

Expected: collection fails because `deploy_smolvla.direct_decoder` does not yet exist, or the noise test fails because `fixed_noise.npy` is absent.

- [ ] **Step 3: Generate and pin the training noise once**

Run:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
import torch

output = Path("checkpoints/ablation/fixed_noise.npy")
generator = torch.Generator(device="cpu")
generator.manual_seed(0)
noise = torch.randn((1, 20, 20), generator=generator, dtype=torch.float32)
noise = torch.nn.functional.pad(noise, (0, 12))
array = noise.numpy().astype(np.float32, copy=False)
assert array.shape == (1, 20, 32)
assert np.isfinite(array).all()
assert np.count_nonzero(array[:, :, 20:]) == 0
np.save(output, array, allow_pickle=False)
print(output)
PY
```

Expected: prints `checkpoints/ablation/fixed_noise.npy`. This checkpoint asset is ignored runtime data and is not staged in Git.

- [ ] **Step 4: Implement the exact PyTorch runtime**

Create `deploy_smolvla/direct_decoder.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

DIRECT_TACTILE_KEYS = (
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)


class SamePadConv2d(nn.Conv2d):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        out_h = (height + self.stride[0] - 1) // self.stride[0]
        out_w = (width + self.stride[1] - 1) // self.stride[1]
        pad_h = max((out_h - 1) * self.stride[0] + self.kernel_size[0] - height, 0)
        pad_w = max((out_w - 1) * self.stride[1] + self.kernel_size[1] - width, 0)
        value = F.pad(value, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        return F.conv2d(value, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class SamePadMaxPool2d(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        out_h, out_w = (height + 1) // 2, (width + 1) // 2
        pad_h = max((out_h - 1) * 2 + 3 - height, 0)
        pad_w = max((out_w - 1) * 2 + 3 - width, 0)
        value = F.pad(value, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=float("-inf"))
        return F.max_pool2d(value, kernel_size=3, stride=2)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = SamePadConv2d(in_channels, channels, 3, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = SamePadConv2d(channels, channels, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        if stride != 1 or in_channels != channels:
            self.proj_conv = SamePadConv2d(in_channels, channels, 1, stride=stride, bias=False)
            self.proj_bn = nn.BatchNorm2d(channels)
        else:
            self.proj_conv = None
            self.proj_bn = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = F.relu(self.bn1(self.conv1(value)))
        value = self.bn2(self.conv2(value))
        if self.proj_conv is not None and self.proj_bn is not None:
            residual = self.proj_bn(self.proj_conv(residual))
        return F.relu(value + residual)


class TactileResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = SamePadConv2d(3, 64, 7, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = SamePadMaxPool2d()
        self.layer1 = self._stage(64, 64, 1)
        self.layer2 = self._stage(64, 128, 2)
        self.layer3 = self._stage(128, 256, 2)
        self.layer4 = self._stage(256, 512, 2)

    @staticmethod
    def _stage(in_channels: int, channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(BasicBlock(in_channels, channels, stride), BasicBlock(channels, channels, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.maxpool(F.relu(self.bn1(self.conv1(value))))
        value = self.layer4(self.layer3(self.layer2(self.layer1(value))))
        return value.mean(dim=(-2, -1))


class DirectTactileActionDecoder(nn.Module):
    def __init__(self, *, chunk_size: int, action_dim: int, tactile_dim: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.action_position = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        self.sensor_identity = nn.Parameter(torch.randn(4, d_model) * 0.02)
        self.action_in = nn.Linear(action_dim, d_model)
        self.tactile_in = nn.Linear(tactile_dim, d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.action_out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, action_dim))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DirectTactileActionDecoder":
        return cls(**{key: config[key] for key in ("chunk_size", "action_dim", "tactile_dim", "d_model", "nhead", "num_layers", "dim_feedforward", "dropout")})

    def forward(self, coarse: torch.Tensor, tactile: torch.Tensor) -> torch.Tensor:
        action_tokens = self.action_in(coarse) + self.action_position
        tactile = tactile.float()
        tactile = tactile / tactile.square().mean(-1, keepdim=True).sqrt().clamp_min(torch.finfo(torch.float32).eps)
        memory = self.tactile_in(tactile) + self.sensor_identity
        return self.action_out(self.decoder(tgt=action_tokens, memory=memory))
```

Complete the same module with `_preprocess_image()` and `DirectDecoderRuntime` using these exact rules:

```python
def _preprocess_image(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"tactile image must be HWC RGB, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min(initial=0.0) < 0 or image.max(initial=0.0) > 1:
            raise ValueError("float tactile image must be finite and in [0, 1]")
        image = np.rint(image * 255.0).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    scale = min(224 / height, 224 / width)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    top = (224 - resized.shape[0]) // 2
    left = (224 - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))


class DirectDecoderRuntime:
    def __init__(
        self,
        *,
        encoder: TactileResNet18,
        decoder: DirectTactileActionDecoder,
        fixed_noise: np.ndarray,
        device: torch.device,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.tactile_keys = DIRECT_TACTILE_KEYS
        self.fixed_noise_jax = jax.device_put(jnp.asarray(fixed_noise, dtype=jnp.float32))

    @classmethod
    def from_bundle(cls, bundle_root: Path, *, device: str | torch.device) -> "DirectDecoderRuntime":
        root = Path(bundle_root).expanduser().resolve()
        torch_device = torch.device(device)
        checkpoint = torch.load(
            root / "decoder" / "best.pt",
            map_location="cpu",
            weights_only=True,
        )
        if checkpoint.get("checkpoint_schema_version") != 1:
            raise ValueError("decoder checkpoint_schema_version must be 1")
        if checkpoint.get("run_kind") != "formal":
            raise ValueError("decoder run_kind must be 'formal'")
        if checkpoint.get("mode") != "action_tactile":
            raise ValueError("decoder mode must be 'action_tactile'")
        config = checkpoint.get("decoder_config")
        state = checkpoint.get("decoder_state_dict")
        if not isinstance(config, Mapping) or not isinstance(state, Mapping):
            raise ValueError("decoder checkpoint is missing config or state dict")
        expected = {
            "chunk_size": 20,
            "execute_steps": 10,
            "action_dim": 20,
            "tactile_dim": 512,
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "smolvla_noise_seed": 0,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"decoder_config.{key} must be {value!r}")
        if tuple(config.get("tactile_keys", ())) != DIRECT_TACTILE_KEYS:
            raise ValueError("decoder_config.tactile_keys has the wrong order")

        encoder = TactileResNet18()
        encoder.load_state_dict(
            load_file(str(root / "tactile_encoder" / "encoder.safetensors")),
            strict=True,
        )
        decoder = DirectTactileActionDecoder.from_config(config)
        decoder.load_state_dict(state, strict=True)
        encoder.to(torch_device).eval().requires_grad_(False)
        decoder.to(torch_device).eval().requires_grad_(False)

        noise = np.load(root / "fixed_noise.npy", allow_pickle=False)
        if noise.dtype != np.float32 or noise.shape != (1, 20, 32):
            raise ValueError("fixed noise must be float32 shaped [1,20,32]")
        if not np.isfinite(noise).all():
            raise ValueError("fixed noise contains NaN or Inf")
        if np.count_nonzero(noise[:, :, 20:]) != 0:
            raise ValueError("fixed noise padding channels must be zero")
        return cls(
            encoder=encoder,
            decoder=decoder,
            fixed_noise=noise,
            device=torch_device,
        )

    def reset(self) -> None:
        return None

    @torch.inference_mode()
    def refine(self, coarse_normalized: np.ndarray, observation: Mapping[str, Any]) -> np.ndarray:
        coarse_array = np.asarray(coarse_normalized, dtype=np.float32)
        if coarse_array.shape != (1, 20, 20) or not np.isfinite(coarse_array).all():
            raise ValueError("coarse normalized action must be finite and shaped [1,20,20]")
        missing = [key for key in self.tactile_keys if key not in observation]
        if missing:
            raise ValueError(f"observation is missing tactile keys: {missing}")
        images = np.stack([_preprocess_image(observation[key]) for key in self.tactile_keys])
        image_tensor = torch.from_numpy(images).to(self.device, dtype=torch.float32)
        tactile = self.encoder(image_tensor)
        tactile = tactile / torch.sqrt(tactile.square().mean(-1, keepdim=True) + 1e-6)
        tactile = tactile.reshape(1, 4, 512)
        coarse = torch.from_numpy(coarse_array).to(self.device, dtype=torch.float32)
        fine = self.decoder(coarse, tactile)
        if fine.shape != (1, 20, 20) or not torch.isfinite(fine).all():
            raise ValueError("decoder output must be finite and shaped [1,20,20]")
        return fine.detach().cpu().numpy().astype(np.float32, copy=False)
```

Do not introduce cache validation, download behavior, residual addition,
background subtraction, or ImageNet normalization.

- [ ] **Step 5: Run only the direct runtime tests**

Run:

```bash
.venv/bin/python -m pytest tests/jax/test_direct_decoder_deployment.py -v
```

Expected: decoder state and noise tests pass; CUDA forward passes when CUDA is available and is otherwise skipped.

- [ ] **Step 6: Commit the runtime**

```bash
git add deploy_smolvla/direct_decoder.py tests/jax/test_direct_decoder_deployment.py
git commit -m "feat: add direct tactile decoder runtime"
```

Do not stage `checkpoints/ablation/fixed_noise.npy` if ignored by repository policy.

### Task 2: Remote-client backend integration

**Files:**
- Modify: `deploy_smolvla/remote_client.py:35-45,161-220,491-530,886-1120`
- Modify: `tests/jax/test_direct_decoder_deployment.py`

**Interfaces:**
- Consumes: `DIRECT_TACTILE_KEYS`, `DirectDecoderRuntime`, `fixed_noise_jax`, and `refine()` from Task 1.
- Produces: root backend `direct_tactile_decoder` and `_predict_chunk(..., direct_decoder: DirectDecoderRuntime | None = None)`.

- [ ] **Step 1: Add failing backend validation and normalized-boundary tests**

Add tests which copy the normal YAML into a temporary file and assert:

```python
def test_direct_backend_requires_vitac_horizon_and_bundle(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "deploy_smolvla/configs/deploy_smolvla_jax.yaml").read_text())
    config["backend"] = "direct_tactile_decoder"
    config["direct_decoder"] = {"bundle": str(ABLATION), "device": "cpu"}
    config["observation"]["data_type"] = "vision"
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="requires observation.data_type='vitac'"):
        remote_client.load_config(path)
```

Add a stub policy/runtime test asserting that `_predict_chunk` passes
`normalized=True`, supplies `direct_decoder.fixed_noise_jax`, calls
`runtime.refine()` with normalized coarse actions, calls the preprocessor
unnormalizer once on the refined result, and returns the physical/refined pair.

- [ ] **Step 2: Run the focused backend tests and verify failure**

```bash
.venv/bin/python -m pytest tests/jax/test_direct_decoder_deployment.py -v
```

Expected: new validation/refinement tests fail because the backend is not wired.

- [ ] **Step 3: Add backend parsing and prediction integration**

Implement these exact decisions in `remote_client.py`:

```python
from .direct_decoder import DIRECT_TACTILE_KEYS, DirectDecoderRuntime

DIRECT_DECODER_BACKEND = "direct_tactile_decoder"


def _direct_decoder_config(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    backend = str(config.get("backend", "jax_smolvla"))
    if backend == "jax_smolvla":
        return None
    if backend != DIRECT_DECODER_BACKEND:
        raise ValueError(f"Unsupported backend: {backend}")
    section = config.get("direct_decoder")
    if not isinstance(section, Mapping):
        raise ValueError("Missing YAML section: direct_decoder")
    _required(section, "bundle", "direct_decoder")
    _required(section, "device", "direct_decoder")
    return section
```

In `load_config()`, for a non-`None` direct section require `vitac`, horizon 20,
`inference_delay is None`, `execution_horizon is None`, and no enabled FRS.

Extend `_predict_chunk` with:

```python
direct_decoder: DirectDecoderRuntime | None = None,
```

and implement the action boundary as:

```python
noise = direct_decoder.fixed_noise_jax if direct_decoder is not None else None
actions_norm = policy.predict_action_chunk(
    observation,
    task,
    seed=seed,
    noise=noise,
    jit=jit,
    normalized=True,
    num_steps=num_steps,
    previous_chunk=None if direct_decoder is not None else previous_chunk,
    inference_delay=None if direct_decoder is not None else inference_delay,
    execution_horizon=None if direct_decoder is not None else execution_horizon,
)
jax.block_until_ready(actions_norm)
final_norm = (
    np.asarray(actions_norm, dtype=np.float32)
    if direct_decoder is None
    else direct_decoder.refine(np.asarray(actions_norm, dtype=np.float32), observation)
)
actions = policy.preprocessor.unnormalize_actions(final_norm)
```

In `run()`, load `DirectDecoderRuntime` after the JAX policy; append
`DIRECT_TACTILE_KEYS` to `robot_image_keys`; use them as required tactile keys;
call `reset()` before warmup; pass the runtime to warmup and normal inference;
and use `seed` instead of `seed + iteration` in direct mode. Leave FRS and normal
SmolVLA branches unchanged.

- [ ] **Step 4: Run focused direct-decoder and legacy prediction tests**

```bash
.venv/bin/python -m pytest \
  tests/jax/test_direct_decoder_deployment.py \
  tests/jax/test_frs_deployment.py::test_predict_chunk_unnormalizes_the_legacy_source_chunk_without_frs \
  -v
```

Expected: all selected tests pass, with only the CUDA runtime forward test potentially skipped.

- [ ] **Step 5: Commit the client integration**

```bash
git add deploy_smolvla/remote_client.py tests/jax/test_direct_decoder_deployment.py
git commit -m "feat: route SmolVLA actions through direct decoder"
```

### Task 3: Dedicated YAML and launcher

**Files:**
- Create: `deploy_smolvla/configs/deploy_direct_decoder.yaml`
- Create: `deploy_smolvla/scripts/start_direct_decoder.sh`
- Modify: `tests/jax/test_direct_decoder_deployment.py`

**Interfaces:**
- Consumes: backend/config schema from Task 2.
- Produces: `bash deploy_smolvla/scripts/start_direct_decoder.sh [--check]`.

- [ ] **Step 1: Add a failing launcher/config test**

```python
def test_direct_decoder_config_and_launcher() -> None:
    config_path = ROOT / "deploy_smolvla/configs/deploy_direct_decoder.yaml"
    launcher_path = ROOT / "deploy_smolvla/scripts/start_direct_decoder.sh"
    config = remote_client.load_config(config_path)
    assert config["backend"] == "direct_tactile_decoder"
    assert config["observation"]["data_type"] == "vitac"
    assert config["control"]["action_horizon"] == 20
    assert config["control"]["steps_per_inference"] == 10
    launcher = launcher_path.read_text()
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in launcher
    assert "start_remote_client.sh" in launcher
```

- [ ] **Step 2: Run the test and verify missing-file failure**

```bash
.venv/bin/python -m pytest tests/jax/test_direct_decoder_deployment.py::test_direct_decoder_config_and_launcher -v
```

Expected: FAIL because the YAML and launcher do not exist.

- [ ] **Step 3: Add the deployment files**

Copy the current SmolVLA YAML connection, prompt, runtime, and logging values,
then set:

```yaml
backend: direct_tactile_decoder
checkpoint: /home/typhon/FRS_Tact/checkpoints/model/pick_tube_01_jax
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
  bundle: /home/typhon/FRS_Tact/checkpoints/ablation
  device: cuda:0
observation:
  data_type: vitac
  language_prompt: Use the left hand to pick up the green tube, and then use the right hand to pick up the blue tube.
  single_arm_mode: false
  no_state_obs_mode: false
control:
  control_frequency: 20.0
  controller_frequency: 80.0
  action_horizon: 20
  steps_per_inference: 10
  inference_delay: null
  execution_horizon: null
```

Create the launcher exactly as:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${DIRECT_DECODER_CONFIG:-${ROOT}/deploy_smolvla/configs/deploy_direct_decoder.yaml}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
exec bash "${ROOT}/deploy_smolvla/scripts/start_remote_client.sh" \
    --config "${CONFIG}" "$@"
```

- [ ] **Step 4: Run focused config/launcher checks**

```bash
.venv/bin/python -m pytest tests/jax/test_direct_decoder_deployment.py::test_direct_decoder_config_and_launcher -v
VB_ROBOT_TOKEN=dummy bash deploy_smolvla/scripts/start_direct_decoder.sh --check
```

Expected: pytest passes; shell output names the direct-decoder YAML and `deploy_smolvla.remote_client` entrypoint without loading a model.

- [ ] **Step 5: Commit deployment entrypoints**

```bash
git add \
  deploy_smolvla/configs/deploy_direct_decoder.yaml \
  deploy_smolvla/scripts/start_direct_decoder.sh \
  tests/jax/test_direct_decoder_deployment.py
git commit -m "feat: add direct decoder deployment launcher"
```

### Task 4: Minimal final verification

**Files:**
- Verify only; no new files unless a discovered integration error requires a focused correction.

**Interfaces:**
- Consumes all previous tasks.
- Produces a ready-to-run direct-decoder launcher and a concise handoff.

- [ ] **Step 1: Compile only touched Python files**

```bash
.venv/bin/python -m py_compile \
  deploy_smolvla/direct_decoder.py \
  deploy_smolvla/remote_client.py \
  tests/jax/test_direct_decoder_deployment.py
```

Expected: exit status 0 and no output.

- [ ] **Step 2: Run the focused tests only**

```bash
.venv/bin/python -m pytest \
  tests/jax/test_direct_decoder_deployment.py \
  tests/jax/test_frs_deployment.py::test_predict_chunk_unnormalizes_the_legacy_source_chunk_without_frs \
  -v
```

Expected: selected tests pass; CUDA-only runtime forward may skip if no GPU is available.

- [ ] **Step 3: Check the launcher without model loading**

```bash
VB_ROBOT_TOKEN=dummy bash deploy_smolvla/scripts/start_direct_decoder.sh --check
```

Expected: exit status 0; reported config is `deploy_direct_decoder.yaml`.

- [ ] **Step 4: Inspect only the scoped diff**

```bash
git status --short
git diff --check
git diff --stat HEAD~3..HEAD
```

Expected: no whitespace errors; the user's untracked
`DIRECT_DECODER_DEPLOYMENT_MODIFICATION_GUIDE.md` remains untouched; changes are
limited to the runtime, remote client, direct YAML/launcher, focused test, design,
and plan documents.
