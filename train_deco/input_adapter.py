"""Dataset-specific input adaptation outside the unmodified upstream DECO model."""

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TVF


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def augment_training_images(images: torch.Tensor) -> torch.Tensor:
    """Apply one shared color augmentation to all views in each sample."""
    augmented = images.clone()
    for sample_index in range(augmented.shape[0]):
        pair = augmented[sample_index]
        if torch.rand((), device=pair.device) < 0.5:
            brightness = 0.7 + 0.6 * torch.rand((), device=pair.device).item()
            contrast = 0.8 + 0.4 * torch.rand((), device=pair.device).item()
            saturation = 0.8 + 0.4 * torch.rand((), device=pair.device).item()
            pair = TVF.adjust_brightness(pair, brightness)
            pair = TVF.adjust_contrast(pair, contrast)
            pair = TVF.adjust_saturation(pair, saturation)
        if torch.rand((), device=pair.device) < 0.5:
            kernel = (3, 5, 7)[int(torch.randint(3, (), device=pair.device).item())]
            sigma = 0.1 + 1.9 * torch.rand((), device=pair.device).item()
            pair = TVF.gaussian_blur(pair, [kernel, kernel], [sigma, sigma])
        augmented[sample_index] = pair
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
