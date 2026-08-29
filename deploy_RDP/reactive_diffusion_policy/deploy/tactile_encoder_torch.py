"""PyTorch inference loader for the pick-tube Flax tactile ResNet18."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _same_padding(size: int, kernel: int, stride: int, dilation: int = 1) -> tuple[int, int]:
    output = math.ceil(size / stride)
    total = max((output - 1) * stride + (kernel - 1) * dilation + 1 - size, 0)
    return total // 2, total - total // 2


class SamePadConv2d(nn.Conv2d):
    """Conv2d with Flax/TensorFlow ``padding='SAME'`` semantics."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        top, bottom = _same_padding(
            inputs.shape[-2], self.kernel_size[0], self.stride[0], self.dilation[0]
        )
        left, right = _same_padding(
            inputs.shape[-1], self.kernel_size[1], self.stride[1], self.dilation[1]
        )
        inputs = F.pad(inputs, (left, right, top, bottom))
        return F.conv2d(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )


def _same_max_pool(inputs: torch.Tensor) -> torch.Tensor:
    top, bottom = _same_padding(inputs.shape[-2], 3, 2)
    left, right = _same_padding(inputs.shape[-1], 3, 2)
    inputs = F.pad(inputs, (left, right, top, bottom), value=float("-inf"))
    return F.max_pool2d(inputs, kernel_size=3, stride=2)


class ResNetBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = SamePadConv2d(
            input_channels, output_channels, kernel_size=3, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = SamePadConv2d(
            output_channels, output_channels, kernel_size=3, stride=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        if input_channels != output_channels or stride != 1:
            self.proj_conv = SamePadConv2d(
                input_channels, output_channels, kernel_size=1, stride=stride, bias=False
            )
            self.proj_bn = nn.BatchNorm2d(output_channels)
        else:
            self.proj_conv = None
            self.proj_bn = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = F.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.proj_conv is not None and self.proj_bn is not None:
            residual = self.proj_bn(self.proj_conv(residual))
        return F.relu(outputs + residual)


class TactileResNet18(nn.Module):
    embedding_dim = 512

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = SamePadConv2d(3, 64, kernel_size=7, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        block_specs = (
            (64, 64, 1),
            (64, 64, 1),
            (64, 128, 2),
            (128, 128, 1),
            (128, 256, 2),
            (256, 256, 1),
            (256, 512, 2),
            (512, 512, 1),
        )
        self.blocks = nn.ModuleList(ResNetBlock(*spec) for spec in block_specs)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected tactile images shaped [N,3,H,W], got {images.shape}")
        outputs = _same_max_pool(F.relu(self.bn1(self.conv1(images))))
        for block in self.blocks:
            outputs = block(outputs)
        return outputs.mean(dim=(-2, -1))


def _copy_tensor(target: torch.Tensor, value: np.ndarray, *, convolution: bool = False) -> None:
    if convolution:
        value = np.transpose(value, (3, 2, 0, 1))
    source = torch.from_numpy(np.ascontiguousarray(value))
    if source.shape != target.shape:
        raise ValueError(f"Encoder parameter shape {source.shape} does not match {target.shape}")
    target.data.copy_(source)


def _load_batch_norm(
    module: nn.BatchNorm2d,
    arrays: dict[str, np.ndarray],
    path: str,
) -> None:
    _copy_tensor(module.weight, arrays[f"tactile_resnet/params/{path}/scale"])
    _copy_tensor(module.bias, arrays[f"tactile_resnet/params/{path}/bias"])
    _copy_tensor(module.running_mean, arrays[f"tactile_resnet/batch_stats/{path}/mean"])
    _copy_tensor(module.running_var, arrays[f"tactile_resnet/batch_stats/{path}/var"])


def load_tactile_resnet18(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> TactileResNet18:
    """Load the original Flax checkpoint without importing JAX or Flax."""
    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / "checkpoint.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    checkpoint_config = metadata["tactile_clip_config"]
    if int(checkpoint_config["embedding_dim"]) != TactileResNet18.embedding_dim:
        raise ValueError("Only the 512D pick-tube tactile encoder is supported")

    params_path = checkpoint_dir / str(metadata["params_file"])
    with np.load(params_path) as archive:
        arrays = {
            path: np.asarray(archive[f"p{index:05d}"])
            for index, path in enumerate(metadata["parameter_paths"])
        }

    model = TactileResNet18()
    _copy_tensor(
        model.conv1.weight,
        arrays["tactile_resnet/params/conv1/kernel"],
        convolution=True,
    )
    _load_batch_norm(model.bn1, arrays, "bn1")
    for block_index, block in enumerate(model.blocks):
        stage = block_index // 2 + 1
        index_in_stage = block_index % 2
        prefix = f"block{stage}_{index_in_stage}"
        _copy_tensor(
            block.conv1.weight,
            arrays[f"tactile_resnet/params/{prefix}/conv1/kernel"],
            convolution=True,
        )
        _load_batch_norm(block.bn1, arrays, f"{prefix}/bn1")
        _copy_tensor(
            block.conv2.weight,
            arrays[f"tactile_resnet/params/{prefix}/conv2/kernel"],
            convolution=True,
        )
        _load_batch_norm(block.bn2, arrays, f"{prefix}/bn2")
        if block.proj_conv is not None and block.proj_bn is not None:
            _copy_tensor(
                block.proj_conv.weight,
                arrays[f"tactile_resnet/params/{prefix}/proj_conv/kernel"],
                convolution=True,
            )
            _load_batch_norm(block.proj_bn, arrays, f"{prefix}/proj_bn")

    model.eval().to(device)
    return model
