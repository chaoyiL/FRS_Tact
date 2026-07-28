from __future__ import annotations

import torch

from lerobot.datasets.transforms import (
    ImageTransforms,
    build_image_transforms,
    image_transforms_config_from_dict,
)


def test_build_image_transforms_disabled_returns_none() -> None:
    assert build_image_transforms(None) is None
    assert build_image_transforms({"enable": False}) is None


def test_build_image_transforms_enabled_applies_to_chw_tensor() -> None:
    tf = build_image_transforms({"enable": True, "max_num_transforms": 2})
    assert isinstance(tf, ImageTransforms)
    image = torch.rand(3, 32, 32)
    out = tf(image)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


def test_image_transforms_config_parses_yaml_lists() -> None:
    cfg = image_transforms_config_from_dict(
        {
            "enable": True,
            "max_num_transforms": 1,
            "tfs": {
                "brightness": {
                    "weight": 1.0,
                    "type": "ColorJitter",
                    "kwargs": {"brightness": [0.9, 1.1]},
                }
            },
        }
    )
    assert cfg.enable is True
    assert cfg.tfs["brightness"].kwargs["brightness"] == (0.9, 1.1)
    out = ImageTransforms(cfg)(torch.rand(3, 16, 16))
    assert out.shape == (3, 16, 16)
