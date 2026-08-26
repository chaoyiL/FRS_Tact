import pytest
import torch

from train_deco.input_adapter import IMAGE_MEAN, letterbox_tactile_images


def test_letterbox_tactile_images_preserves_rgb_unit_space_with_black_padding() -> None:
    images = torch.zeros(2, 4, 3, 2, 4)
    images[:, :, 0].fill_(0.25)
    images[:, :, 1].fill_(0.50)
    images[:, :, 2].fill_(0.75)

    result = letterbox_tactile_images(images, (6, 6))

    assert result.shape == (2, 4, 3, 6, 6)
    assert torch.count_nonzero(result[:, :, :, :1, :]) == 0
    assert torch.count_nonzero(result[:, :, :, -2:, :]) == 0
    assert result.min().item() == 0.0
    assert result.max().item() <= 1.0
    torch.testing.assert_close(result[:, :, 0, 1:4], torch.full((2, 4, 3, 6), 0.25))
    torch.testing.assert_close(result[:, :, 1, 1:4], torch.full((2, 4, 3, 6), 0.50))
    torch.testing.assert_close(result[:, :, 2, 1:4], torch.full((2, 4, 3, 6), 0.75))
    assert not torch.allclose(result[:, :, :, 1:4], torch.as_tensor(IMAGE_MEAN).view(1, 1, 3, 1, 1))


@pytest.mark.parametrize(
    "images, target_size",
    [
        (torch.zeros(1, 3, 3, 4, 4), (4, 4)),
        (torch.zeros(1, 4, 1, 4, 4), (4, 4)),
        (torch.zeros(1, 4, 3, 4), (4, 4)),
        (torch.zeros(1, 4, 3, 4, 4), (0, 4)),
    ],
)
def test_letterbox_tactile_images_rejects_invalid_contract(images, target_size) -> None:
    with pytest.raises(ValueError):
        letterbox_tactile_images(images, target_size)
