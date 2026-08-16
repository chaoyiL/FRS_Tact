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
        pad_h = max(
            (out_h - 1) * self.stride[0] + self.kernel_size[0] - height, 0
        )
        pad_w = max(
            (out_w - 1) * self.stride[1] + self.kernel_size[1] - width, 0
        )
        value = F.pad(
            value,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        )
        return F.conv2d(
            value,
            self.weight,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )


class SamePadMaxPool2d(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        out_h, out_w = (height + 1) // 2, (width + 1) // 2
        pad_h = max((out_h - 1) * 2 + 3 - height, 0)
        pad_w = max((out_w - 1) * 2 + 3 - width, 0)
        value = F.pad(
            value,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            value=float("-inf"),
        )
        return F.max_pool2d(value, kernel_size=3, stride=2)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = SamePadConv2d(
            in_channels, channels, 3, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = SamePadConv2d(channels, channels, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        if stride != 1 or in_channels != channels:
            self.proj_conv = SamePadConv2d(
                in_channels, channels, 1, stride=stride, bias=False
            )
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
        return nn.Sequential(
            BasicBlock(in_channels, channels, stride), BasicBlock(channels, channels, 1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.maxpool(F.relu(self.bn1(self.conv1(value))))
        value = self.layer4(self.layer3(self.layer2(self.layer1(value))))
        return value.mean(dim=(-2, -1))


class DirectTactileActionDecoder(nn.Module):
    def __init__(
        self,
        *,
        chunk_size: int,
        action_dim: int,
        tactile_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.action_position = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        self.sensor_identity = nn.Parameter(torch.randn(4, d_model) * 0.02)
        self.action_in = nn.Linear(action_dim, d_model)
        self.tactile_in = nn.Linear(tactile_dim, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.action_out = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, action_dim)
        )

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any]
    ) -> "DirectTactileActionDecoder":
        return cls(
            **{
                key: config[key]
                for key in (
                    "chunk_size",
                    "action_dim",
                    "tactile_dim",
                    "d_model",
                    "nhead",
                    "num_layers",
                    "dim_feedforward",
                    "dropout",
                )
            }
        )

    def forward(self, coarse: torch.Tensor, tactile: torch.Tensor) -> torch.Tensor:
        action_tokens = self.action_in(coarse) + self.action_position
        tactile = tactile.float()
        tactile = tactile / tactile.square().mean(-1, keepdim=True).sqrt().clamp_min(
            torch.finfo(torch.float32).eps
        )
        memory = self.tactile_in(tactile) + self.sensor_identity
        return self.action_out(self.decoder(tgt=action_tokens, memory=memory))


def _preprocess_image(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"tactile image must be HWC RGB, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if (
            not np.isfinite(image).all()
            or image.min(initial=0.0) < 0
            or image.max(initial=0.0) > 1
        ):
            raise ValueError("float tactile image must be finite and in [0, 1]")
        image = np.rint(image * 255.0).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    scale = min(224 / height, 224 / width)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
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
        self.fixed_noise_jax = jax.device_put(
            jnp.asarray(fixed_noise, dtype=jnp.float32)
        )
        self.last_vla_normalized: np.ndarray | None = None
        self.last_direct_normalized: np.ndarray | None = None

    @classmethod
    def from_bundle(
        cls, bundle_root: Path, *, device: str | torch.device
    ) -> "DirectDecoderRuntime":
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
        self.last_vla_normalized = None
        self.last_direct_normalized = None

    @torch.inference_mode()
    def refine(
        self, coarse_normalized: np.ndarray, observation: Mapping[str, Any]
    ) -> np.ndarray:
        coarse_array = np.asarray(coarse_normalized, dtype=np.float32)
        if coarse_array.shape != (1, 20, 20) or not np.isfinite(coarse_array).all():
            raise ValueError(
                "coarse normalized action must be finite and shaped [1,20,20]"
            )
        missing = [key for key in self.tactile_keys if key not in observation]
        if missing:
            raise ValueError(f"observation is missing tactile keys: {missing}")
        images = np.stack(
            [_preprocess_image(observation[key]) for key in self.tactile_keys]
        )
        image_tensor = torch.from_numpy(images).to(self.device, dtype=torch.float32)
        tactile = self.encoder(image_tensor)
        tactile = tactile / torch.sqrt(tactile.square().mean(-1, keepdim=True) + 1e-6)
        tactile = tactile.reshape(1, 4, 512)
        coarse = torch.from_numpy(coarse_array).to(self.device, dtype=torch.float32)
        fine = self.decoder(coarse, tactile)
        if fine.shape != (1, 20, 20) or not torch.isfinite(fine).all():
            raise ValueError("decoder output must be finite and shaped [1,20,20]")
        direct = fine.detach().cpu().numpy().astype(np.float32, copy=False)
        self.last_vla_normalized = np.array(coarse_array, copy=True)
        self.last_direct_normalized = np.array(direct, copy=True)
        return direct
