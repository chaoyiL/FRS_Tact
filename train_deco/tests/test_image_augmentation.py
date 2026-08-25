from dataclasses import replace
import math

import pytest
import torch

from train_deco.input_adapter import (
    LowLightAugmentationConfig,
    augment_training_images,
    validate_augmentation_config,
)


def _config(**overrides) -> LowLightAugmentationConfig:
    config = LowLightAugmentationConfig(
        identity_probability=0.0,
        low_light_probability=1.0,
        mild_probability=0.0,
        exposure_probability=1.0,
        exposure_range=(0.6, 0.6),
        gamma_range=(1.4, 1.4),
        mild_brightness_range=(1.0, 1.0),
        contrast_range=(1.0, 1.0),
        saturation_range=(1.0, 1.0),
        blur_probability=0.0,
    )
    return replace(config, **overrides)


def test_disabled_augmentation_is_bitwise_identity():
    images = torch.rand(2, 2, 3, 8, 8)

    result = augment_training_images(images, _config(enabled=False))

    assert torch.equal(result, images)


def test_identity_branch_stays_original_even_when_blur_probability_is_one():
    images = torch.rand(1, 2, 3, 8, 8)
    config = _config(
        identity_probability=1.0,
        low_light_probability=0.0,
        blur_probability=1.0,
    )

    result = augment_training_images(images, config)

    assert torch.equal(result, images)


def test_fixed_seed_reproduces_augmentation():
    images = torch.rand(3, 2, 3, 8, 8)
    config = LowLightAugmentationConfig()

    torch.manual_seed(123)
    first = augment_training_images(images, config)
    torch.manual_seed(123)
    second = augment_training_images(images, config)

    assert torch.equal(first, second)


def test_all_camera_views_share_photometric_parameters():
    view = torch.rand(3, 8, 8)
    images = view[None, None].repeat(1, 2, 1, 1, 1)

    result = augment_training_images(images, _config(exposure_range=(0.7, 0.7)))

    assert torch.equal(result[:, 0], result[:, 1])
    assert not torch.equal(result[:, 0], images[:, 0])


def test_exposure_branch_preserves_contract_and_darkens():
    images = torch.full((1, 2, 3, 4, 4), 0.8, dtype=torch.float32)

    result = augment_training_images(images, _config())

    assert result.shape == images.shape
    assert result.dtype == images.dtype
    assert result.device == images.device
    assert result.min().item() >= 0.0
    assert result.max().item() <= 1.0
    torch.testing.assert_close(result, torch.full_like(images, 0.48))


def test_gamma_branch_darkens_midtones():
    images = torch.full((1, 2, 3, 4, 4), 0.5)
    config = _config(exposure_probability=0.0)

    result = augment_training_images(images, config)

    torch.testing.assert_close(result, torch.full_like(images, 0.5**1.4))


@pytest.mark.parametrize(
    "config, message",
    [
        (
            _config(identity_probability=0.2, low_light_probability=0.7),
            "sum to 1",
        ),
        (_config(exposure_range=(0.9, 0.6)), "exposure_range"),
        (_config(gamma_range=(0.0, 1.4)), "gamma_range"),
        (_config(exposure_range=(math.nan, 0.9)), "exposure_range"),
        (_config(gamma_range=(1.1, math.inf)), "gamma_range"),
        (_config(blur_probability=1.1), "blur_probability"),
        (_config(blur_kernel_sizes=(2, 3)), "blur_kernel_sizes"),
    ],
)
def test_invalid_augmentation_config_is_rejected(config, message):
    with pytest.raises(ValueError, match=message):
        validate_augmentation_config(config)
