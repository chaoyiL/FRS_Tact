"""PyTorch ResNet18 compatible with the frozen Flax tactile backbone."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


RESNET18_EMBEDDING_DIM = 512
TACTILE_IMAGE_SHAPE = (3, 224, 224)


class SamePadConv2d(nn.Conv2d):
    """TensorFlow/Flax ``SAME`` convolution with explicit dynamic padding."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        stride_h, stride_w = self.stride
        kernel_h, kernel_w = self.kernel_size
        dilation_h, dilation_w = self.dilation
        out_h = (height + stride_h - 1) // stride_h
        out_w = (width + stride_w - 1) // stride_w
        pad_h = max((out_h - 1) * stride_h + dilation_h * (kernel_h - 1) + 1 - height, 0)
        pad_w = max((out_w - 1) * stride_w + dilation_w * (kernel_w - 1) + 1 - width, 0)
        if pad_h or pad_w:
            value = F.pad(value, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        return F.conv2d(value, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


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
        self.conv1 = SamePadConv2d(in_channels, channels, 3, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = SamePadConv2d(channels, channels, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        if stride != 1 or in_channels != channels:
            self.proj_conv: SamePadConv2d | None = SamePadConv2d(in_channels, channels, 1, stride=stride, bias=False)
            self.proj_bn: nn.BatchNorm2d | None = nn.BatchNorm2d(channels)
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
    """Frozen tactile ResNet18 producing exactly one 512-D vector per RGB frame."""

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
        if not isinstance(value, torch.Tensor):
            raise ValueError("tactile encoder input must be a torch.Tensor shaped [N, 3, 224, 224]")
        if value.ndim != 4 or tuple(value.shape[1:]) != TACTILE_IMAGE_SHAPE:
            raise ValueError(f"tactile encoder input must be shaped [N, 3, 224, 224], got {tuple(value.shape)}")
        if value.shape[0] < 1:
            raise ValueError("tactile encoder input batch dimension N must be positive")
        if not (value.is_floating_point() or value.is_complex()):
            raise ValueError("tactile encoder input must use a floating-point dtype")
        if not torch.isfinite(value).all():
            raise ValueError("tactile encoder input must contain only finite values")
        value = value.float()
        value = self.maxpool(F.relu(self.bn1(self.conv1(value))))
        value = self.layer4(self.layer3(self.layer2(self.layer1(value))))
        return value.mean(dim=(-2, -1))
