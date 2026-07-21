from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .configuration import JaxSmolVLAConfig

Array = jax.Array
TrainMode = Literal["frozen", "full", "lora"]

MODULE_NAMES = ("vision", "connector", "vlm_text", "expert", "action", "state_proj")
VALID_TRAIN_MODES = frozenset(("frozen", "full", "lora"))

_ACTION_PREFIXES = (
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.action_time_mlp_in.",
    "model.action_time_mlp_out.",
)
_LINEAR_WEIGHT_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".self_attn.out_proj.weight",
    ".mlp.fc1.weight",
    ".mlp.fc2.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)
_EXACT_LINEAR_WEIGHTS = frozenset(
    {
        "model.state_proj.weight",
        "model.action_in_proj.weight",
        "model.action_out_proj.weight",
        "model.action_time_mlp_in.weight",
        "model.action_time_mlp_out.weight",
        "model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight",
    }
)


def module_for_parameter(name: str) -> str | None:
    """Map a flat SmolVLA parameter name to its configurable module."""

    if ".lm_expert." in name:
        return "expert"
    if ".vision_model." in name:
        return "vision"
    if ".connector." in name:
        return "connector"
    if ".text_model." in name:
        return "vlm_text"
    if name.startswith(_ACTION_PREFIXES):
        return "action"
    if name.startswith("model.state_proj."):
        return "state_proj"
    return None


def legacy_module_modes(config: JaxSmolVLAConfig) -> dict[str, TrainMode]:
    """Translate the original boolean switches into the six module modes."""

    modes: dict[str, TrainMode] = {
        "vision": "frozen",
        "connector": "frozen",
        "vlm_text": "frozen",
        "expert": "full",
        "action": "full",
        "state_proj": "full" if config.train_state_proj else "frozen",
    }
    if not config.train_expert_only:
        modes["connector"] = "full"
        modes["vlm_text"] = "full"
        modes["vision"] = "frozen" if config.freeze_vision_encoder else "full"
    return modes


def resolve_module_modes(config: JaxSmolVLAConfig) -> dict[str, TrainMode]:
    raw_modes = config.module_modes
    if raw_modes is None:
        return legacy_module_modes(config)
    if not isinstance(raw_modes, Mapping):
        raise ValueError("module_modes must be a mapping")

    unknown_modules = sorted(set(raw_modes) - set(MODULE_NAMES))
    if unknown_modules:
        raise ValueError(f"unknown module_modes keys: {unknown_modules}")
    missing_modules = sorted(set(MODULE_NAMES) - set(raw_modes))
    if missing_modules:
        raise ValueError(f"module_modes must configure every module; missing: {missing_modules}")

    modes: dict[str, TrainMode] = {}
    for module in MODULE_NAMES:
        mode = str(raw_modes[module]).lower()
        if mode not in VALID_TRAIN_MODES:
            raise ValueError(
                f"invalid mode for module {module!r}: {mode!r}; "
                f"expected one of {sorted(VALID_TRAIN_MODES)}"
            )
        modes[module] = mode  # type: ignore[assignment]
    return modes


def is_lora_eligible_weight(name: str, value: Array | np.ndarray | None = None) -> bool:
    """Return whether ``name`` is a linear weight reached by model._linear."""

    if name in _EXACT_LINEAR_WEIGHTS:
        eligible = True
    else:
        eligible = name.endswith(_LINEAR_WEIGHT_SUFFIXES)
    if not eligible:
        return False
    return value is None or getattr(value, "ndim", None) == 2


def lora_prefix_from_key(name: str) -> str | None:
    for suffix in (".lora_a", ".lora_b", ".lora_scale"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def initialize_lora_params(
    params: Mapping[str, Array],
    config: JaxSmolVLAConfig,
    *,
    seed: int = 0,
) -> dict[str, Array]:
    """Add zero-impact LoRA adapters for every linear layer in LoRA modules."""

    modes = resolve_module_modes(config)
    if not any(mode == "lora" for mode in modes.values()):
        return dict(params)
    if config.lora_rank <= 0:
        raise ValueError(f"lora_rank must be positive, got {config.lora_rank}")
    if config.lora_alpha <= 0:
        raise ValueError(f"lora_alpha must be positive, got {config.lora_alpha}")

    output = dict(params)
    rng = np.random.default_rng(seed)
    scale = np.float32(config.lora_alpha / config.lora_rank)
    for name, value in tuple(params.items()):
        module = module_for_parameter(name)
        if module is None or modes[module] != "lora" or not is_lora_eligible_weight(name, value):
            continue
        prefix = name.removesuffix(".weight")
        a_key = f"{prefix}.lora_a"
        b_key = f"{prefix}.lora_b"
        scale_key = f"{prefix}.lora_scale"
        out_features, in_features = value.shape
        if a_key not in output:
            std = 1.0 / np.sqrt(float(in_features))
            output[a_key] = jnp.asarray(
                rng.normal(0.0, std, (config.lora_rank, in_features)),
                dtype=jnp.float32,
            )
        if b_key not in output:
            output[b_key] = jnp.zeros((out_features, config.lora_rank), dtype=jnp.float32)
        if scale_key not in output:
            output[scale_key] = jnp.asarray(scale, dtype=jnp.float32)
    return output


def is_trainable_parameter(name: str, config: JaxSmolVLAConfig) -> bool:
    if config.module_modes is None:
        if ".lm_expert." in name:
            return ".lm_head." not in name
        if name.startswith("model.state_proj."):
            return config.train_state_proj
        if name.startswith(_ACTION_PREFIXES):
            return True
        if config.train_expert_only:
            return False
        if config.freeze_vision_encoder and ".vision_model." in name:
            return False
        frozen_fragments = [
            ".vlm.lm_head.",
            ".text_model.norm.weight",
            f".text_model.layers.{config.num_vlm_layers - 1}.",
        ]
        if (
            config.num_vlm_layers != config.num_expert_layers
            and config.num_vlm_layers % config.num_expert_layers == 0
        ):
            frozen_fragments.append(f".text_model.layers.{config.num_vlm_layers - 2}.")
        return not any(fragment in name for fragment in frozen_fragments)

    module = module_for_parameter(name)
    if module is None:
        return False
    mode = resolve_module_modes(config)[module]
    adapter_prefix = lora_prefix_from_key(name)
    if adapter_prefix is not None:
        return mode == "lora" and not name.endswith(".lora_scale")
    return mode == "full"
