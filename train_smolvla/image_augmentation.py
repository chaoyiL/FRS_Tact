"""SmolVLA image augmentation presets calibrated from the DECO reference recipe."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torchvision.transforms import functional as TVF


@dataclass(frozen=True)
class ImageAugmentationConfig:
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


AUGMENTATION_PRESET_NAMES = ("low-light-v1", "balanced-light-v2")


def augmentation_preset(
    name: str,
    *,
    enabled: bool = True,
) -> ImageAugmentationConfig:
    """Return the SmolVLA preset matching the calibrated reference values."""
    if name == "low-light-v1":
        return ImageAugmentationConfig(enabled=enabled)
    if name == "balanced-light-v2":
        return ImageAugmentationConfig(
            version="balanced-light-v2",
            enabled=enabled,
            identity_probability=0.25,
            low_light_probability=0.0,
            mild_probability=0.75,
            mild_brightness_range=(0.90, 1.20),
        )
    raise ValueError(
        f"unknown SmolVLA image augmentation preset {name!r}; "
        f"expected one of {AUGMENTATION_PRESET_NAMES}"
    )


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


def validate_augmentation_config(config: ImageAugmentationConfig) -> None:
    if config.version not in AUGMENTATION_PRESET_NAMES:
        raise ValueError(f"unknown SmolVLA augmentation version: {config.version!r}")
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
        raise ValueError("SmolVLA image augmentation requires shared_across_cameras=True")


def _sample_range(value: tuple[float, float], device: torch.device) -> float:
    low, high = value
    return low + (high - low) * torch.rand((), device=device).item()


def augment_training_images(
    images: torch.Tensor,
    config: ImageAugmentationConfig | None = None,
) -> torch.Tensor:
    """Augment [B,N,C,H,W] images with parameters shared across sample views."""
    config = config or ImageAugmentationConfig()
    validate_augmentation_config(config)
    if not config.enabled:
        return images
    augmented = images.clone()
    for sample_index in range(augmented.shape[0]):
        views = augmented[sample_index]
        branch = torch.rand((), device=views.device).item()
        identity_branch = branch < config.identity_probability
        if identity_branch:
            pass
        elif branch < config.identity_probability + config.low_light_probability:
            if torch.rand((), device=views.device).item() < config.exposure_probability:
                views = TVF.adjust_brightness(
                    views, _sample_range(config.exposure_range, views.device)
                )
            else:
                views = TVF.adjust_gamma(
                    views, _sample_range(config.gamma_range, views.device)
                )
            views = TVF.adjust_contrast(
                views, _sample_range(config.contrast_range, views.device)
            )
            views = TVF.adjust_saturation(
                views, _sample_range(config.saturation_range, views.device)
            )
        else:
            views = TVF.adjust_brightness(
                views, _sample_range(config.mild_brightness_range, views.device)
            )
            views = TVF.adjust_contrast(
                views, _sample_range(config.contrast_range, views.device)
            )
            views = TVF.adjust_saturation(
                views, _sample_range(config.saturation_range, views.device)
            )
        if (
            not identity_branch
            and torch.rand((), device=views.device).item() < config.blur_probability
        ):
            kernel_index = int(
                torch.randint(
                    len(config.blur_kernel_sizes), (), device=views.device
                ).item()
            )
            kernel = config.blur_kernel_sizes[kernel_index]
            sigma = _sample_range(config.blur_sigma_range, views.device)
            views = TVF.gaussian_blur(views, [kernel, kernel], [sigma, sigma])
        augmented[sample_index] = views.clamp_(0.0, 1.0)
    return augmented
