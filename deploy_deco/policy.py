"""TorchScript inference boundary for the two-camera DECO policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .artifact import TACTILE_FIELD_ORDER, artifact_uses_tactile, load_torchscript


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
        phase_count = self.metadata.get("phase_count")
        if phase_count is None:
            self.phase_count: int | None = None
        elif isinstance(phase_count, bool) or not isinstance(phase_count, int) or phase_count <= 0:
            raise ValueError("DECO phase_count must be a positive integer")
        else:
            self.phase_count = phase_count
        self.uses_tactile = artifact_uses_tactile(self.metadata)
        self.tactile_keys = TACTILE_FIELD_ORDER if self.uses_tactile else ()
        self.visual_hw = tuple(self.metadata["input"]["images"][3:5])
        self.tactile_hw = (
            tuple(self.metadata["input"]["tactile_images"][3:5])
            if self.uses_tactile
            else None
        )
        if self.uses_tactile and self.phase_count is not None:
            raise ValueError("DECO Stage 2 tactile artifacts cannot also be phase-conditioned")

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

    @staticmethod
    def _stack_images(
        images: list[np.ndarray], keys: tuple[str, ...], expected_hw: tuple[int, int] | None = None
    ) -> np.ndarray:
        image_shapes = {image.shape for image in images}
        if len(image_shapes) != 1:
            raise ValueError(f"DECO image shapes must match, got {sorted(image_shapes)}")
        if expected_hw is not None:
            expected_shape = (*expected_hw, 3)
            if any(image.shape != expected_shape for image in images):
                raise ValueError(
                    f"DECO Stage 2 {keys[0]} images must have shape {expected_shape}, "
                    f"got {sorted(image_shapes)}"
                )
        return np.ascontiguousarray(np.stack(images, axis=0).transpose(0, 3, 1, 2)[None])

    def prepare_inputs(self, observation: Mapping[str, Any]):
        missing = [
            key
            for key in (*self.image_keys, *self.tactile_keys, "observation.state")
            if key not in observation
        ]
        if missing:
            raise ValueError(f"robot observation is missing keys: {missing}")
        images = [self._image(observation[key], key) for key in self.image_keys]
        image_batch = self._stack_images(
            images, self.image_keys, self.visual_hw if self.uses_tactile else None
        )
        if self.uses_tactile:
            tactile_images = [self._image(observation[key], key) for key in self.tactile_keys]
            tactile_batch = self._stack_images(tactile_images, self.tactile_keys, self.tactile_hw)
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        if state.shape != (self.state_dim,) or not np.isfinite(state).all():
            raise ValueError(
                f"DECO state must be finite with shape ({self.state_dim},), got {state.shape}"
            )
        state_batch = state[None]
        torch = self.torch
        visual_tensor = torch.from_numpy(image_batch).to(self.device)
        state_tensor = torch.from_numpy(np.ascontiguousarray(state_batch)).to(self.device)
        if self.uses_tactile:
            tactile_tensor = torch.from_numpy(tactile_batch).to(self.device)
            return visual_tensor, tactile_tensor, state_tensor
        return visual_tensor, state_tensor

    def predict(
        self,
        observation: Mapping[str, Any],
        *,
        seed: int,
        phase_id: int | None = None,
    ) -> np.ndarray:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("DECO inference seed must be a nonnegative integer")
        if self.phase_count is not None:
            if (
                isinstance(phase_id, bool)
                or not isinstance(phase_id, int)
                or phase_id < 0
                or phase_id >= self.phase_count
            ):
                raise ValueError(
                    f"DECO phase_id must be an integer in [0,{self.phase_count - 1}]"
                )
        torch = self.torch
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        inputs = self.prepare_inputs(observation)
        with torch.inference_mode():
            if self.uses_tactile:
                images, tactile_images, state = inputs
                output = self.model(images, tactile_images, state)
            elif self.phase_count is not None:
                images, state = inputs
                phase = torch.tensor(
                    [phase_id], dtype=torch.long, device=self.device
                )
                output = self.model(images, state, phase)
            else:
                images, state = inputs
                output = self.model(images, state)
        action = output.detach().to(device="cpu", dtype=torch.float32).numpy()
        expected = (1, self.action_horizon, self.action_dim)
        if action.shape != expected or not np.isfinite(action).all():
            raise RuntimeError(f"DECO output must be finite with shape {expected}, got {action.shape}")
        return np.array(action[0], dtype=np.float32, copy=True)
