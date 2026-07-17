from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.flax import load_file as load_safetensors_file
from transformers import AutoTokenizer

from .configuration import JaxSmolVLAConfig

Array = jax.Array


def _normalize(value: Array, minimum: float, maximum: float) -> Array:
    return (value - minimum) / (maximum - minimum)


def _unnormalize(value: Array, minimum: float, maximum: float) -> Array:
    return value * (maximum - minimum) + minimum


def aloha_decode_state(state: Array) -> Array:
    if state.shape[-1] < 14:
        raise ValueError("Aloha state adaptation requires at least 14 state dimensions")
    state = state.at[..., jnp.asarray([1, 2, 8, 9])].multiply(-1)
    for index in (6, 13):
        linear_position = _unnormalize(state[..., index], 0.01844, 0.05800)
        ratio = (0.022**2 + linear_position**2 - 0.036**2) / (2 * 0.022 * linear_position)
        radians = jnp.arcsin(jnp.clip(ratio, -1.0, 1.0))
        state = state.at[..., index].set(_normalize(radians, 0.4, 1.5))
    return state


def aloha_encode_actions(actions: Array) -> Array:
    if actions.shape[-1] < 14:
        raise ValueError("Aloha action adaptation requires at least 14 action dimensions")
    actions = actions.at[..., jnp.asarray([1, 2, 8, 9])].multiply(-1)
    for index in (6, 13):
        radians = _unnormalize(actions[..., index], 0.4, 1.5)
        actions = actions.at[..., index].set(_normalize(radians, -0.6213, 1.4910))
    return actions


def aloha_encode_actions_inverse(actions: Array) -> Array:
    if actions.shape[-1] < 14:
        raise ValueError("Aloha action adaptation requires at least 14 action dimensions")
    actions = actions.at[..., jnp.asarray([1, 2, 8, 9])].multiply(-1)
    for index in (6, 13):
        radians = _unnormalize(actions[..., index], -0.6213, 1.4910)
        actions = actions.at[..., index].set(_normalize(radians, 0.4, 1.5))
    return actions


def resize_with_pad(image: Array, width: int, height: int, pad_value: float = 0.0) -> Array:
    """Match the PyTorch BCHW bilinear resize followed by left/top padding."""

    if image.ndim != 4:
        raise ValueError(f"expected BCHW images, got {image.shape}")
    current_height, current_width = image.shape[-2:]
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = jax.image.resize(
        image,
        (image.shape[0], image.shape[1], resized_height, resized_width),
        method="linear",
        antialias=False,
    )
    pad_height = max(0, height - resized_height)
    pad_width = max(0, width - resized_width)
    return jnp.pad(
        resized,
        ((0, 0), (0, 0), (pad_height, 0), (pad_width, 0)),
        constant_values=pad_value,
    )


def _as_bchw(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[None, ...]
    if image.ndim != 4:
        raise ValueError(f"expected an HWC/CHW image or batch, got {image.shape}")
    if image.shape[-1] in (1, 3):
        image = np.transpose(image, (0, 3, 1, 2))
    elif image.shape[1] not in (1, 3):
        raise ValueError(f"cannot identify image channels in shape {image.shape}")
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
        if image.size and float(np.max(image)) > 1.0:
            image = image / 255.0
    return image


class JaxSmolVLAPreprocessor:
    """Backend-independent tokenizer, normalization, and image preparation."""

    def __init__(
        self,
        checkpoint: str | Path,
        config: JaxSmolVLAConfig | None = None,
        *,
        rename_map: Mapping[str, str] | None = None,
        local_files_only: bool = True,
    ):
        self.checkpoint = Path(checkpoint).expanduser()
        self.config = config or JaxSmolVLAConfig.from_pretrained(self.checkpoint)
        processor_config = self._load_json("policy_preprocessor.json", default={})
        self.rename_map = dict(
            rename_map
            or self._find_step_config(processor_config, "rename_observations_processor").get("rename_map", {})
        )
        tokenizer_config = self._find_step_config(processor_config, "tokenizer_processor")
        tokenizer_name = tokenizer_config.get("tokenizer_name", self.config.tokenizer_name)
        os.environ.setdefault("HF_HUB_OFFLINE", "1" if local_files_only else "0")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=local_files_only)
        self.max_length = int(tokenizer_config.get("max_length", self.config.tokenizer_max_length))
        self.padding = tokenizer_config.get("padding", self.config.pad_language_to)
        self.padding_side = tokenizer_config.get("padding_side", "right")
        self.stats = self._load_stats("policy_preprocessor_step_5_normalizer_processor.safetensors")
        self.post_stats = self._load_stats("policy_postprocessor_step_0_unnormalizer_processor.safetensors")

    def _load_json(self, filename: str, default: Any) -> Any:
        path = self.checkpoint / filename
        if not path.is_file():
            return default
        with path.open() as file:
            return json.load(file)

    @staticmethod
    def _find_step_config(config: Mapping[str, Any], registry_name: str) -> dict[str, Any]:
        for step in config.get("steps", []):
            if step.get("registry_name") == registry_name:
                return dict(step.get("config", {}))
        return {}

    def _load_stats(self, filename: str) -> dict[str, Array]:
        path = self.checkpoint / filename
        return dict(load_safetensors_file(path)) if path.is_file() else {}

    def _stat(self, key: str, name: str, length: int, *, postprocess: bool = False) -> Array | None:
        values = self.post_stats if postprocess and self.post_stats else self.stats
        value = values.get(f"{key}.{name}")
        if value is None:
            return None
        value = jnp.asarray(value)
        if value.ndim == 1 and value.shape[0] > length:
            value = value[:length]
        return value

    def normalize_state(self, state: Array) -> Array:
        mean = self._stat("observation.state", "mean", state.shape[-1])
        std = self._stat("observation.state", "std", state.shape[-1])
        if mean is None or std is None:
            return state
        return (state - mean) / (std + 1e-8)

    def unnormalize_actions(self, actions: Array) -> Array:
        mean = self._stat("action", "mean", actions.shape[-1], postprocess=True)
        std = self._stat("action", "std", actions.shape[-1], postprocess=True)
        if mean is None or std is None:
            return actions
        return actions * std + mean

    def normalize_actions(self, actions: Array) -> Array:
        mean = self._stat("action", "mean", actions.shape[-1])
        std = self._stat("action", "std", actions.shape[-1])
        if mean is not None and std is not None:
            actions = (actions - mean) / (std + 1e-8)
        if self.config.adapt_to_pi_aloha:
            actions = aloha_encode_actions_inverse(actions)
        return actions

    def tokenize(self, task: str) -> tuple[Array, Array]:
        if not task.endswith("\n"):
            task = f"{task}\n"
        tokenized = self.tokenizer(
            [task],
            max_length=self.max_length,
            truncation=True,
            padding=self.padding,
            padding_side=self.padding_side,
            return_tensors="np",
        )
        return jnp.asarray(tokenized["input_ids"], dtype=jnp.int32), jnp.asarray(
            tokenized["attention_mask"], dtype=jnp.bool_
        )

    def prepare(self, observation: Mapping[str, Any], task: str) -> dict[str, Array]:
        renamed = {self.rename_map.get(key, key): value for key, value in observation.items()}
        if "observation.state" not in renamed:
            raise KeyError("observation.state is required")
        state = jnp.asarray(renamed["observation.state"], dtype=jnp.float32)
        if state.ndim == 1:
            state = state[None, :]
        state = self.normalize_state(state)
        if self.config.adapt_to_pi_aloha:
            state = aloha_decode_state(state)

        present_keys = [key for key in self.config.image_keys if key in renamed]
        missing_keys = [key for key in self.config.image_keys if key not in renamed]
        if not present_keys:
            raise ValueError(f"none of the expected image keys are present: {self.config.image_keys}")
        images: list[Array] = []
        masks: list[Array] = []
        for key in present_keys:
            image = jnp.asarray(_as_bchw(renamed[key]), dtype=jnp.float32)
            image = resize_with_pad(
                image,
                self.config.resize_width,
                self.config.resize_height,
                pad_value=0.0,
            )
            images.append(image * 2.0 - 1.0)
            masks.append(jnp.ones((image.shape[0],), dtype=jnp.bool_))
        for _ in missing_keys[: self.config.empty_cameras]:
            images.append(-jnp.ones_like(images[-1]))
            masks.append(jnp.zeros_like(masks[-1]))
        tokens, language_masks = self.tokenize(task)
        if tokens.shape[0] != state.shape[0]:
            tokens = jnp.broadcast_to(tokens, (state.shape[0], tokens.shape[1]))
            language_masks = jnp.broadcast_to(language_masks, tokens.shape)
        return {
            "images": jnp.stack(images, axis=1),
            "image_masks": jnp.stack(masks, axis=1),
            "language_tokens": tokens,
            "language_masks": language_masks,
            "state": state,
        }
