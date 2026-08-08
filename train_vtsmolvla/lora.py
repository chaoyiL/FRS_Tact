from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from train_smolvla import lora as visual_lora

from .configuration import VTSmolVLAConfig

Array = jax.Array
TACTILE_MODULE_NAME = "tactile_proj"
MODULE_NAMES = (*visual_lora.MODULE_NAMES, TACTILE_MODULE_NAME)


def _visual_config(config: VTSmolVLAConfig) -> VTSmolVLAConfig:
    if config.module_modes is None:
        return config
    modes = {
        name: mode
        for name, mode in config.module_modes.items()
        if name in visual_lora.MODULE_NAMES
    }
    return replace(config, module_modes=modes)


def module_for_parameter(name: str) -> str | None:
    if name.startswith("model.tactile_proj."):
        return TACTILE_MODULE_NAME
    return visual_lora.module_for_parameter(name)


def resolve_module_modes(config: VTSmolVLAConfig) -> dict[str, visual_lora.TrainMode]:
    if config.module_modes is not None:
        if not isinstance(config.module_modes, Mapping):
            raise ValueError("module_modes must be a mapping")
        unknown = sorted(set(config.module_modes) - set(MODULE_NAMES))
        if unknown:
            raise ValueError(f"unknown module_modes keys: {unknown}")
    modes = visual_lora.resolve_module_modes(_visual_config(config))
    raw_modes = config.module_modes or {}
    mode = str(
        raw_modes.get(
            TACTILE_MODULE_NAME,
            "full" if config.use_tactile_encoder else "frozen",
        )
    ).lower()
    if mode not in visual_lora.VALID_TRAIN_MODES:
        raise ValueError(
            f"invalid mode for module {TACTILE_MODULE_NAME!r}: {mode!r}; "
            f"expected one of {sorted(visual_lora.VALID_TRAIN_MODES)}"
        )
    modes[TACTILE_MODULE_NAME] = mode  # type: ignore[assignment]
    return modes


def is_lora_eligible_weight(name: str, value: Array | np.ndarray | None = None) -> bool:
    if name == "model.tactile_proj.weight":
        return value is None or getattr(value, "ndim", None) == 2
    return visual_lora.is_lora_eligible_weight(name, value)


def initialize_lora_params(
    params: Mapping[str, Array],
    config: VTSmolVLAConfig,
    *,
    seed: int = 0,
) -> dict[str, Array]:
    output = visual_lora.initialize_lora_params(params, _visual_config(config), seed=seed)
    if resolve_module_modes(config)[TACTILE_MODULE_NAME] != "lora":
        return output
    if config.lora_rank <= 0:
        raise ValueError(f"lora_rank must be positive, got {config.lora_rank}")
    if config.lora_alpha <= 0:
        raise ValueError(f"lora_alpha must be positive, got {config.lora_alpha}")
    weight = output.get("model.tactile_proj.weight")
    if weight is None:
        return output
    out_features, in_features = weight.shape
    rng = np.random.default_rng(seed + 211)
    output.setdefault(
        "model.tactile_proj.lora_a",
        jnp.asarray(
            rng.normal(0.0, 1.0 / np.sqrt(float(in_features)), (config.lora_rank, in_features)),
            dtype=jnp.float32,
        ),
    )
    output.setdefault(
        "model.tactile_proj.lora_b",
        jnp.zeros((out_features, config.lora_rank), dtype=jnp.float32),
    )
    output.setdefault(
        "model.tactile_proj.lora_scale",
        jnp.asarray(config.lora_alpha / config.lora_rank, dtype=jnp.float32),
    )
    return output


def is_trainable_parameter(name: str, config: VTSmolVLAConfig) -> bool:
    if name.startswith("model.tactile_encoder."):
        return False
    if name.startswith("model.tactile_proj."):
        if not config.use_tactile_encoder:
            return False
        if config.module_modes is None:
            return True
        mode = resolve_module_modes(config)[TACTILE_MODULE_NAME]
        adapter_prefix = visual_lora.lora_prefix_from_key(name)
        if adapter_prefix is not None:
            return mode == "lora" and not name.endswith(".lora_scale")
        return mode == "full"
    return visual_lora.is_trainable_parameter(name, _visual_config(config))


__all__ = [
    "MODULE_NAMES",
    "initialize_lora_params",
    "is_lora_eligible_weight",
    "is_trainable_parameter",
    "module_for_parameter",
    "resolve_module_modes",
]
