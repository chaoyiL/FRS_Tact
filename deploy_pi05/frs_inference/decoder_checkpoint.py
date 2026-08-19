"""Load the Pi0.5 tactile FRS decoder for online inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx, traverse_util

from .decoder import DecoderConfig, TactileConditionedFlowDecoder


PARAMS_NAME = "params.npz"
CHECKPOINT_NAME = "checkpoint.json"


def _full_parameter_state(model: TactileConditionedFlowDecoder):
    state = nnx.state(model, nnx.Param)
    return state, traverse_util.flatten_dict(state.to_pure_dict())


def _flat_parameter_state(model: TactileConditionedFlowDecoder):
    state, flat = _full_parameter_state(model)
    return state, {path: value for path, value in flat.items() if value is not None}


def _path_name(path: tuple[Any, ...]) -> str:
    return "/".join(f"{type(part).__name__}:{part}" for part in path)


def load_checkpoint(directory: Path) -> tuple[TactileConditionedFlowDecoder, dict[str, Any]]:
    with (directory / CHECKPOINT_NAME).open(encoding="utf-8") as file:
        metadata = json.load(file)
    config = DecoderConfig(**metadata["decoder_config"])
    model = TactileConditionedFlowDecoder(config, rngs=nnx.Rngs(0))
    state, full_template = _full_parameter_state(model)
    ordered_full = sorted(full_template.items(), key=lambda item: _path_name(item[0]))
    ordered_filtered = [(path, value) for path, value in ordered_full if value is not None]
    metadata_names = metadata["parameter_paths"]
    if metadata_names == [_path_name(path) for path, _ in ordered_filtered]:
        indexed_paths = [(index, path) for index, (path, _) in enumerate(ordered_filtered)]
    elif metadata_names == [_path_name(path) for path, _ in ordered_full]:
        indexed_paths = [
            (index, path)
            for index, (path, value) in enumerate(ordered_full)
            if value is not None
        ]
    else:
        raise ValueError("Checkpoint parameter structure does not match the decoder implementation.")

    with np.load(directory / PARAMS_NAME) as archive:
        restored_flat = {
            path: jnp.asarray(archive[f"p{index:05d}"])
            for index, path in indexed_paths
        }
    state.replace_by_pure_dict(traverse_util.unflatten_dict(restored_flat))
    nnx.update(model, state)
    return model, metadata
