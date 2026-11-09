"""Tactile image preprocessing used by online FRS deployment."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize keeping aspect ratio and pad with black to ``(height, width)``.

    Accepts uint8 HWC ``[0, 255]`` or float32 HWC ``[0, 1]``.
    """

    if image.ndim != 3 or image.shape[-1] not in (1, 3):
        raise ValueError(f"Expected HWC image, got shape {image.shape}.")
    cur_h, cur_w = image.shape[:2]
    if cur_h == height and cur_w == width:
        return image
    ratio = max(cur_w / width, cur_h / height)
    resized_h = max(1, int(round(cur_h / ratio)))
    resized_w = max(1, int(round(cur_w / ratio)))
    interpolation = cv2.INTER_AREA if ratio > 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    if image.dtype == np.uint8:
        resized = np.clip(np.rint(resized), 0, 255).astype(np.uint8)
        pad_value = 0
    else:
        resized = np.clip(resized, 0.0, 1.0).astype(np.float32)
        pad_value = 0.0
    pad_h0, rem_h = divmod(height - resized_h, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(width - resized_w, 2)
    pad_w1 = pad_w0 + rem_w
    return np.pad(
        resized,
        ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )


def _normalize_image_layout(image: Any) -> np.ndarray:
    """Normalize a raw LeRobot image to HWC with 3 channels (dtype unchanged)."""

    array = np.asarray(image)
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {array.shape}.")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {array.shape}.")
    return array


def parse_image_to_uint8(image: Any, *, image_size: int) -> np.ndarray:
    """Convert a raw LeRobot image to uint8 HWC in ``[0, 255]`` at ``image_size``.

    Fast path: already-uint8 HWC at the target size skips resize and rescale.
    """

    array = _normalize_image_layout(image)
    if array.dtype == np.uint8 and array.shape[0] == image_size and array.shape[1] == image_size:
        return np.ascontiguousarray(array)

    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32, copy=False)
        if array.size and float(np.nanmin(array)) < 0.0:
            array = (array + 1.0) * 0.5
        elif array.size and float(np.nanmax(array)) > 1.5:
            array = array / 255.0
        array = np.clip(array, 0.0, 1.0)
        resized = resize_with_pad(array, image_size, image_size)
        return np.clip(np.rint(resized * 255.0), 0, 255).astype(np.uint8)

    # Integer / other non-float: treat as 0..255 then resize as uint8.
    array = np.clip(array, 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(resize_with_pad(array, image_size, image_size))


def parse_image_to_unit(image: Any, *, image_size: int) -> np.ndarray:
    """Convert a raw LeRobot image to float32 HWC in ``[0, 1]`` at ``image_size``."""

    return parse_image_to_uint8(image, image_size=image_size).astype(np.float32) * (1.0 / 255.0)
