from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import traverse_util

from train_smolvla.checkpoint import (
    count_expert_layers,
    count_vlm_layers,
    extend_vlm_layers,
    load_params,
    load_safetensors_params,
    parameter_summary,
    resolve_checkpoint,
    restore_orbax_params,
    save_orbax_params,
    save_portable_params,
    write_effective_config as write_visual_effective_config,
)

from .configuration import VTSmolVLAConfig

Array = jax.Array
_TACTILE_ENCODER_PREFIX = "model.tactile_encoder."


def _flatten_tactile_resnet_params(tactile_resnet: Mapping[str, Any]) -> dict[str, Array]:
    flat = traverse_util.flatten_dict(dict(tactile_resnet))
    return {
        _TACTILE_ENCODER_PREFIX + "/".join(str(part) for part in path): jax.device_put(value)
        for path, value in flat.items()
    }


def initialize_tactile_fusion_params(
    params: Mapping[str, Array],
    config: VTSmolVLAConfig,
    *,
    seed: int = 0,
) -> dict[str, Array]:
    """Attach the frozen ResNet and stable ``model.tactile_*`` fusion keys."""

    if not config.use_tactile_encoder:
        return dict(params)
    if not config.freeze_tactile_encoder:
        raise NotImplementedError("第一版 VT-SmolVLA 只支持 freeze_tactile_encoder=True")
    if not config.tactile_encoder_path:
        raise ValueError("tactile_encoder_path is required when use_tactile_encoder=True")

    output = dict(params)
    if not any(name.startswith(_TACTILE_ENCODER_PREFIX) for name in output):
        from train_encoder.utils.checkpoint import load_tactile_encoder
        from train_encoder.utils.model import tactile_clip_config_from_dict

        bundle = load_tactile_encoder(config.tactile_encoder_path)
        tactile_cfg = tactile_clip_config_from_dict(bundle.metadata["tactile_clip_config"])
        if int(tactile_cfg.embedding_dim) != int(config.tactile_embedding_dim):
            raise ValueError(
                "tactile_embedding_dim does not match tactile encoder checkpoint "
                f"({config.tactile_embedding_dim} != {tactile_cfg.embedding_dim})"
            )
        if int(tactile_cfg.tactile_image_size) != int(config.tactile_image_size):
            raise ValueError(
                "tactile_image_size does not match tactile encoder checkpoint "
                f"({config.tactile_image_size} != {tactile_cfg.tactile_image_size})"
            )
        if "tactile_resnet" not in bundle.params:
            raise KeyError("tactile encoder checkpoint is missing tactile_resnet params")
        output.update(_flatten_tactile_resnet_params(bundle.params["tactile_resnet"]))

    weight_key = "model.tactile_proj.weight"
    bias_key = "model.tactile_proj.bias"
    if weight_key not in output:
        rng = np.random.default_rng(seed + 101)
        in_dim = int(config.tactile_embedding_dim)
        out_dim = int(config.text_hidden_size)
        limit = np.sqrt(6.0 / float(in_dim + out_dim))
        output[weight_key] = jax.device_put(
            rng.uniform(-limit, limit, (out_dim, in_dim)).astype(np.float32)
        )
    if bias_key not in output:
        output[bias_key] = jax.device_put(
            np.zeros((int(config.text_hidden_size),), dtype=np.float32)
        )
    return output


def write_effective_config(destination: str | Path, config: VTSmolVLAConfig) -> Path:
    config_path = write_visual_effective_config(destination, config)
    with config_path.open(encoding="utf-8") as file:
        raw = json.load(file)
    raw.update(
        {
            "use_tactile_encoder": config.use_tactile_encoder,
            "tactile_encoder_path": config.tactile_encoder_path,
            "freeze_tactile_encoder": config.freeze_tactile_encoder,
            "tactile_keys": list(config.tactile_keys),
            "tactile_embedding_dim": config.tactile_embedding_dim,
            "tactile_num_tokens": config.tactile_num_tokens,
            "tactile_image_size": config.tactile_image_size,
        }
    )
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(raw, file, indent=4)
        file.write("\n")
    return config_path


def load_config(path: str | Path) -> VTSmolVLAConfig:
    path = Path(path).expanduser()
    if path.name == "params":
        path = path.parent
    return VTSmolVLAConfig.from_pretrained(path)


__all__ = [
    "count_expert_layers",
    "count_vlm_layers",
    "extend_vlm_layers",
    "initialize_tactile_fusion_params",
    "load_config",
    "load_params",
    "load_safetensors_params",
    "parameter_summary",
    "resolve_checkpoint",
    "restore_orbax_params",
    "save_orbax_params",
    "save_portable_params",
    "write_effective_config",
]
