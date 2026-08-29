from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from train_deco.input_adapter import augmentation_preset
from train_smolvla.torch_train import (
    augment_smolvla_training_batch,
    resolve_deco_augmentation,
)


def _config(*, preset: str = "balanced-light-v2", lerobot_transforms: bool = False) -> dict:
    return {
        "dataset": {
            "image_keys": [
                "observation.images.camera0",
                "observation.images.camera1",
            ],
            "image_transforms": {"enable": lerobot_transforms},
        },
        "augmentation": {"preset": preset, "enabled": True},
    }


def test_resolve_deco_augmentation_uses_exact_balanced_light_v2_preset() -> None:
    assert resolve_deco_augmentation(_config()) == augmentation_preset("balanced-light-v2")


def test_resolve_deco_augmentation_retains_low_light_v1_for_reproduction() -> None:
    assert resolve_deco_augmentation(_config(preset="low-light-v1")) == augmentation_preset(
        "low-light-v1"
    )


def test_deco_augmentation_rejects_double_augmentation() -> None:
    with pytest.raises(ValueError, match="image_transforms must be disabled"):
        resolve_deco_augmentation(_config(lerobot_transforms=True))


def test_batch_augmentation_shares_parameters_across_cameras() -> None:
    image = torch.linspace(0.0, 1.0, 3 * 8 * 8).reshape(1, 3, 8, 8)
    batch = {"camera1": image.clone(), "camera2": image.clone()}
    config = replace(
        augmentation_preset("low-light-v1"),
        identity_probability=0.0,
        low_light_probability=1.0,
        mild_probability=0.0,
        exposure_probability=1.0,
        exposure_range=(0.7, 0.7),
        contrast_range=(1.0, 1.0),
        saturation_range=(1.0, 1.0),
        blur_probability=0.0,
    )

    augment_smolvla_training_batch(batch, ("camera1", "camera2"), config)

    torch.testing.assert_close(batch["camera1"], image * 0.7)
    torch.testing.assert_close(batch["camera1"], batch["camera2"])


def test_disabled_batch_augmentation_is_identity() -> None:
    batch = {"camera1": torch.rand(2, 3, 4, 4), "camera2": torch.rand(2, 3, 4, 4)}
    original = {key: value.clone() for key, value in batch.items()}

    result = augment_smolvla_training_batch(
        batch,
        ("camera1", "camera2"),
        augmentation_preset("low-light-v1", enabled=False),
    )

    assert result is batch
    for key in batch:
        torch.testing.assert_close(batch[key], original[key])
