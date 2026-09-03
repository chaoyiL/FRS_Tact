import torch

from reactive_diffusion_policy.model.vision.multi_image_obs_encoder import (
    MultiImageObsEncoder,
)


def test_deploy_encoder_accepts_training_only_bread_config():
    shape_meta = {
        "obs": {
            "camera1": {"shape": [3, 8, 8], "type": "rgb"},
            "camera2": {"shape": [3, 8, 8], "type": "rgb"},
        }
    }
    rgb_model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
    )

    encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        photometric_augmentation={"type": "Bread"},
    )
    output = encoder(
        {
            "camera1": torch.rand(2, 3, 8, 8),
            "camera2": torch.rand(2, 3, 8, 8),
        }
    )

    assert output.shape == (2, 6)
