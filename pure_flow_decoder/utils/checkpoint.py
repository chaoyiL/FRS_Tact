from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax import traverse_util

from pure_flow_decoder.utils.cache import atomic_write_json
from pure_flow_decoder.utils.model import DecoderConfig
from pure_flow_decoder.utils.model import SelfAttentionFlowDecoder

PARAMS_NAME = "params.npz"
CHECKPOINT_NAME = "checkpoint.json"


def _flat_parameter_state(model: SelfAttentionFlowDecoder):
    state = nnx.state(model, nnx.Param)
    return state, traverse_util.flatten_dict(state.to_pure_dict())


def _path_name(path: tuple[Any, ...]) -> str:
    return "/".join(f"{type(part).__name__}:{part}" for part in path)


def save_checkpoint(
    directory: pathlib.Path,
    model: SelfAttentionFlowDecoder,
    *,
    epoch: int,
    metrics: dict[str, float],
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _, flat = _flat_parameter_state(model)
    ordered = sorted(flat.items(), key=lambda item: _path_name(item[0]))
    arrays = {f"p{index:05d}": np.asarray(value) for index, (_, value) in enumerate(ordered)}
    temporary = directory / (PARAMS_NAME + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(directory / PARAMS_NAME)
    metadata = {
        "version": 1,
        "epoch": int(epoch),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "decoder_config": dataclasses.asdict(model.config),
        "parameter_paths": [_path_name(path) for path, _ in ordered],
    }
    if extra_metadata is not None:
        metadata["extra_metadata"] = extra_metadata
    atomic_write_json(directory / CHECKPOINT_NAME, metadata)


def load_checkpoint(directory: pathlib.Path) -> tuple[SelfAttentionFlowDecoder, dict[str, Any]]:
    import json

    with (directory / CHECKPOINT_NAME).open(encoding="utf-8") as file:
        metadata = json.load(file)
    config = DecoderConfig(**metadata["decoder_config"])
    model = SelfAttentionFlowDecoder(config, rngs=nnx.Rngs(0))
    state, flat_template = _flat_parameter_state(model)
    ordered_paths = sorted(flat_template, key=_path_name)
    actual_names = [_path_name(path) for path in ordered_paths]
    if actual_names != metadata["parameter_paths"]:
        raise ValueError("Checkpoint parameter structure does not match the decoder implementation.")

    with np.load(directory / PARAMS_NAME) as archive:
        restored_flat = {
            path: jnp.asarray(archive[f"p{index:05d}"])
            for index, path in enumerate(ordered_paths)
        }
    state.replace_by_pure_dict(traverse_util.unflatten_dict(restored_flat))
    nnx.update(model, state)
    return model, metadata
