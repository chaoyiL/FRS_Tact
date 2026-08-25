"""Dataset-specific input adaptation outside the unmodified upstream DECO model."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TVF


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class LowLightAugmentationConfig:
    version: str = "low-light-v1"
    enabled: bool = True
    identity_probability: float = 0.25
    low_light_probability: float = 0.55
    mild_probability: float = 0.20
    exposure_probability: float = 0.5
    exposure_range: tuple[float, float] = (0.58, 0.90)
    gamma_range: tuple[float, float] = (1.10, 1.50)
    mild_brightness_range: tuple[float, float] = (0.90, 1.10)
    contrast_range: tuple[float, float] = (0.85, 1.10)
    saturation_range: tuple[float, float] = (0.90, 1.10)
    blur_probability: float = 0.20
    blur_kernel_sizes: tuple[int, ...] = (3, 5)
    blur_sigma_range: tuple[float, float] = (0.1, 1.0)
    shared_across_cameras: bool = True


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_range(name: str, value: tuple[float, float]) -> None:
    if (
        len(value) != 2
        or not all(math.isfinite(endpoint) for endpoint in value)
        or value[0] <= 0
        or value[0] > value[1]
    ):
        raise ValueError(f"{name} must be an ordered positive pair, got {value}")


def validate_augmentation_config(config: LowLightAugmentationConfig) -> None:
    for name in (
        "identity_probability",
        "low_light_probability",
        "mild_probability",
        "exposure_probability",
        "blur_probability",
    ):
        _validate_probability(name, getattr(config, name))
    branch_total = (
        config.identity_probability
        + config.low_light_probability
        + config.mild_probability
    )
    if abs(branch_total - 1.0) > 1e-8:
        raise ValueError(f"augmentation branch probabilities must sum to 1, got {branch_total}")
    for name in (
        "exposure_range",
        "gamma_range",
        "mild_brightness_range",
        "contrast_range",
        "saturation_range",
        "blur_sigma_range",
    ):
        _validate_range(name, getattr(config, name))
    if not config.blur_kernel_sizes or any(
        kernel <= 0 or kernel % 2 == 0 for kernel in config.blur_kernel_sizes
    ):
        raise ValueError(
            "blur_kernel_sizes must contain positive odd integers, "
            f"got {config.blur_kernel_sizes}"
        )
    if not config.shared_across_cameras:
        raise ValueError("low-light-v1 requires shared_across_cameras=True")


def _sample_range(value: tuple[float, float], device: torch.device) -> float:
    low, high = value
    return low + (high - low) * torch.rand((), device=device).item()


def augment_training_images(
    images: torch.Tensor,
    config: LowLightAugmentationConfig | None = None,
) -> torch.Tensor:
    """Apply calibrated low-light augmentation shared by all sample views."""
    config = config or LowLightAugmentationConfig()
    validate_augmentation_config(config)
    if not config.enabled:
        return images
    augmented = images.clone()
    for sample_index in range(augmented.shape[0]):
        pair = augmented[sample_index]
        branch = torch.rand((), device=pair.device).item()
        identity_branch = branch < config.identity_probability
        if identity_branch:
            pass
        elif branch < config.identity_probability + config.low_light_probability:
            if torch.rand((), device=pair.device).item() < config.exposure_probability:
                pair = TVF.adjust_brightness(
                    pair, _sample_range(config.exposure_range, pair.device)
                )
            else:
                pair = TVF.adjust_gamma(
                    pair, _sample_range(config.gamma_range, pair.device)
                )
            contrast = _sample_range(config.contrast_range, pair.device)
            saturation = _sample_range(config.saturation_range, pair.device)
            pair = TVF.adjust_contrast(pair, contrast)
            pair = TVF.adjust_saturation(pair, saturation)
        else:
            brightness = _sample_range(config.mild_brightness_range, pair.device)
            contrast = _sample_range(config.contrast_range, pair.device)
            saturation = _sample_range(config.saturation_range, pair.device)
            pair = TVF.adjust_brightness(pair, brightness)
            pair = TVF.adjust_contrast(pair, contrast)
            pair = TVF.adjust_saturation(pair, saturation)
        if (
            not identity_branch
            and torch.rand((), device=pair.device).item() < config.blur_probability
        ):
            kernel_index = int(
                torch.randint(
                    len(config.blur_kernel_sizes), (), device=pair.device
                ).item()
            )
            kernel = config.blur_kernel_sizes[kernel_index]
            sigma = _sample_range(config.blur_sigma_range, pair.device)
            pair = TVF.gaussian_blur(pair, [kernel, kernel], [sigma, sigma])
        augmented[sample_index] = pair.clamp_(0.0, 1.0)
    return augmented


def letterbox_and_normalize(images: torch.Tensor, image_size: int) -> torch.Tensor:
    """Letterbox and ImageNet-normalize [B,N,3,H,W], where N is 2 or 3."""
    if images.ndim != 5 or images.shape[1] not in (2, 3) or images.shape[2] != 3:
        raise ValueError(
            "DECO expects 2 or 3 RGB cameras [B,N,3,H,W], "
            f"got {tuple(images.shape)}"
        )
    batch, cameras, channels, height, width = images.shape
    scale = image_size / max(int(height), int(width))
    resized_height = max(1, int(int(height) * scale))
    resized_width = max(1, int(int(width) * scale))
    flattened = images.reshape(batch * cameras, channels, height, width)
    flattened = F.interpolate(
        flattened,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    pad_height = image_size - resized_height
    pad_width = image_size - resized_width
    flattened = F.pad(
        flattened,
        (
            pad_width // 2,
            pad_width - pad_width // 2,
            pad_height // 2,
            pad_height - pad_height // 2,
        ),
        value=128.0 / 255.0,
    )
    mean = flattened.new_tensor(IMAGE_MEAN).view(1, 3, 1, 1)
    std = flattened.new_tensor(IMAGE_STD).view(1, 3, 1, 1)
    return ((flattened - mean) / std).reshape(
        batch, cameras, channels, image_size, image_size
    )


def select_deco_observation(
    observation: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    if observation.ndim != 2:
        raise ValueError(
            "The DECO source adapter expects one state vector per sample, "
            f"got {tuple(observation.shape)}"
        )
    return observation.index_select(-1, indices.to(observation.device))
