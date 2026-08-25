"""TorchScript inference boundary for the two-camera DECO policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .artifact import load_torchscript


class DECOPolicy:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cuda:0",
        verify_hash: bool = True,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required for DECO deployment") from error
        self.torch = torch
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"DECO device {device!r} requires CUDA, but CUDA is unavailable")
        self.device = torch.device(device)
        self.model, self.metadata = load_torchscript(
            checkpoint, device=str(self.device), verify_hash=verify_hash
        )
        self.image_keys = tuple(self.metadata["camera_names"])
        self.state_dim = int(self.metadata["input"]["observation"][1])
        self.action_horizon = int(self.metadata["output"]["action"][1])
        self.action_dim = int(self.metadata["output"]["action"][2])
        self.expected_sample_hz = float(self.metadata["expected_sample_hz"])

    @staticmethod
    def _image(value: Any, key: str) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{key} must be HWC RGB, got {image.shape}")
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        if not np.issubdtype(image.dtype, np.floating):
            raise ValueError(f"{key} must be uint8 or floating RGB, got {image.dtype}")
        result = image.astype(np.float32, copy=False)
        if not np.isfinite(result).all() or result.min() < 0.0 or result.max() > 1.0:
            raise ValueError(f"{key} floating values must be finite in [0,1]")
        return result

    def prepare_inputs(self, observation: Mapping[str, Any]):
        missing = [
            key for key in (*self.image_keys, "observation.state") if key not in observation
        ]
        if missing:
            raise ValueError(f"robot observation is missing keys: {missing}")
        images = [self._image(observation[key], key) for key in self.image_keys]
        image_shapes = {image.shape for image in images}
        if len(image_shapes) != 1:
            raise ValueError(f"DECO camera shapes must match, got {sorted(image_shapes)}")
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        if state.shape != (self.state_dim,) or not np.isfinite(state).all():
            raise ValueError(
                f"DECO state must be finite with shape ({self.state_dim},), got {state.shape}"
            )
        image_batch = np.stack(images, axis=0).transpose(0, 3, 1, 2)[None]
        state_batch = state[None]
        torch = self.torch
        return (
            torch.from_numpy(np.ascontiguousarray(image_batch)).to(self.device),
            torch.from_numpy(np.ascontiguousarray(state_batch)).to(self.device),
        )

    def predict(self, observation: Mapping[str, Any], *, seed: int) -> np.ndarray:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("DECO inference seed must be a nonnegative integer")
        torch = self.torch
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        images, state = self.prepare_inputs(observation)
        with torch.inference_mode():
            output = self.model(images, state)
        action = output.detach().to(device="cpu", dtype=torch.float32).numpy()
        expected = (1, self.action_horizon, self.action_dim)
        if action.shape != expected or not np.isfinite(action).all():
            raise RuntimeError(f"DECO output must be finite with shape {expected}, got {action.shape}")
        return np.array(action[0], dtype=np.float32, copy=True)
