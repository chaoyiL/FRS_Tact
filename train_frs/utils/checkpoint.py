from __future__ import annotations

import dataclasses
import pathlib
import pickle
import uuid
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax import traverse_util

from utils.cache import atomic_write_json
from train_frs.utils.model import DecoderConfig
from train_frs.utils.model import TactileConditionedFlowDecoder

PARAMS_NAME = "params.npz"
OPT_STATE_NAME = "opt_state.npz"
OPT_STATE_TREEDEF_NAME = "opt_state.treedef.pkl"
CHECKPOINT_NAME = "checkpoint.json"


def _flat_parameter_state(model: TactileConditionedFlowDecoder):
    state = nnx.state(model, nnx.Param)
    return state, traverse_util.flatten_dict(state.to_pure_dict())


def _path_name(path: tuple[Any, ...]) -> str:
    return "/".join(f"{type(part).__name__}:{part}" for part in path)


def _atomic_savez(path: pathlib.Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.parent / (path.name + ".writing.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def _leaf_to_numpy(leaf: Any) -> np.ndarray:
    if isinstance(leaf, (np.bool_, bool)):
        return np.asarray(leaf, dtype=np.bool_)
    if isinstance(leaf, (np.integer, int)):
        return np.asarray(leaf, dtype=np.int32)
    if isinstance(leaf, (np.floating, float)):
        return np.asarray(leaf, dtype=np.float32)
    dtype = getattr(leaf, "dtype", None)
    if dtype is not None and str(dtype) == "bfloat16":
        return np.asarray(jnp.asarray(leaf).astype(jnp.float32))
    array = np.asarray(jax.device_get(leaf))
    if array.dtype.kind == "V":
        raise TypeError(f"Cannot serialize void-dtype leaf with shape {array.shape}.")
    if np.issubdtype(array.dtype, np.floating):
        return np.asarray(array, dtype=np.float32)
    if np.issubdtype(array.dtype, np.integer):
        return np.asarray(array, dtype=np.int32 if array.dtype != np.int64 else np.int64)
    if array.dtype == np.bool_:
        return array
    raise TypeError(f"Unsupported opt_state leaf dtype {array.dtype} shape {array.shape}.")


def _numpy_to_leaf(array: np.ndarray) -> Any:
    array = np.asarray(array)
    if array.dtype.kind == "V":
        raise TypeError(f"Unsupported void dtype {array.dtype} in checkpoint leaf.")
    return jnp.asarray(array)


def _optimizer_step_value(optimizer: nnx.Optimizer) -> int:
    return int(np.asarray(jax.device_get(optimizer.step[...])))


def _serialize_metric_value(value: Any) -> float | str | int | bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (str,)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    return float(value)


def save_checkpoint(
    directory: pathlib.Path,
    model: TactileConditionedFlowDecoder,
    *,
    epoch: int,
    metrics: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
    optimizer: nnx.Optimizer | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    params_name = f"params-{generation}.npz"
    opt_state_name = f"opt_state-{generation}.npz"
    opt_tree_name = f"opt_state-{generation}.treedef.pkl"
    _, flat = _flat_parameter_state(model)
    ordered = sorted(flat.items(), key=lambda item: _path_name(item[0]))
    arrays = {f"p{index:05d}": np.asarray(value) for index, (_, value) in enumerate(ordered)}
    _atomic_savez(directory / params_name, arrays)
    metadata: dict[str, Any] = {
        "version": 2,
        "epoch": int(epoch),
        "metrics": {key: _serialize_metric_value(value) for key, value in metrics.items()},
        "decoder_config": dataclasses.asdict(model.config),
        "parameter_paths": [_path_name(path) for path, _ in ordered],
        "has_opt_state": optimizer is not None,
        "params_file": params_name,
    }
    if optimizer is not None:
        leaves, treedef = jax.tree_util.tree_flatten(optimizer.opt_state)
        opt_arrays = {f"p{index:05d}": _leaf_to_numpy(leaf) for index, leaf in enumerate(leaves)}
        _atomic_savez(directory / opt_state_name, opt_arrays)
        treedef_path = directory / opt_tree_name
        temporary = treedef_path.with_suffix(treedef_path.suffix + ".tmp")
        with temporary.open("wb") as file:
            pickle.dump(treedef, file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(treedef_path)
        metadata["opt_state_leaf_count"] = len(leaves)
        metadata["optimizer_step"] = _optimizer_step_value(optimizer)
        metadata["opt_state_file"] = opt_state_name
        metadata["opt_state_treedef_file"] = opt_tree_name
    if extra_metadata is not None:
        metadata["extra_metadata"] = extra_metadata
    atomic_write_json(directory / CHECKPOINT_NAME, metadata)
    keep = {params_name, opt_state_name, opt_tree_name}
    for pattern in ("params-*.npz", "opt_state-*.npz", "opt_state-*.treedef.pkl"):
        for path in directory.glob(pattern):
            if path.name not in keep:
                path.unlink(missing_ok=True)


def load_checkpoint(directory: pathlib.Path) -> tuple[TactileConditionedFlowDecoder, dict[str, Any]]:
    import json

    with (directory / CHECKPOINT_NAME).open(encoding="utf-8") as file:
        metadata = json.load(file)
    config = DecoderConfig(**metadata["decoder_config"])
    model = TactileConditionedFlowDecoder(config, rngs=nnx.Rngs(0))
    state, flat_template = _flat_parameter_state(model)
    ordered_paths = sorted(flat_template, key=_path_name)
    actual_names = [_path_name(path) for path in ordered_paths]
    if actual_names != metadata["parameter_paths"]:
        raise ValueError("Checkpoint parameter structure does not match the decoder implementation.")

    params_path = directory / str(metadata.get("params_file", PARAMS_NAME))
    with np.load(params_path) as archive:
        restored_flat = {
            path: jnp.asarray(archive[f"p{index:05d}"])
            for index, path in enumerate(ordered_paths)
        }
    state.replace_by_pure_dict(traverse_util.unflatten_dict(restored_flat))
    nnx.update(model, state)
    return model, metadata


def load_optimizer_state(directory: pathlib.Path) -> tuple[Any | None, int | None]:
    """Load optimizer pytree and step count when present; otherwise ``(None, None)``."""

    import json

    directory = pathlib.Path(directory)
    with (directory / CHECKPOINT_NAME).open(encoding="utf-8") as file:
        metadata = json.load(file)
    opt_state_path = directory / str(metadata.get("opt_state_file", OPT_STATE_NAME))
    treedef_path = directory / str(
        metadata.get("opt_state_treedef_file", OPT_STATE_TREEDEF_NAME)
    )
    if not metadata.get("has_opt_state"):
        return None, None
    missing = [path for path in (opt_state_path, treedef_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"checkpoint optimizer files are missing: {missing}")
    try:
        with treedef_path.open("rb") as file:
            treedef = pickle.load(file)
        with np.load(opt_state_path) as archive:
            leaf_count = int(metadata.get("opt_state_leaf_count", len(archive.files)))
            leaves = [_numpy_to_leaf(archive[f"p{index:05d}"]) for index in range(leaf_count)]
        opt_state = jax.tree_util.tree_unflatten(treedef, leaves)
        step = metadata.get("optimizer_step")
        return opt_state, (int(step) if step is not None else None)
    except Exception as exc:  # noqa: BLE001 - checkpoint corruption must be explicit
        raise RuntimeError(f"failed to restore optimizer state from {directory}") from exc


def restore_optimizer_state(
    optimizer: nnx.Optimizer,
    *,
    opt_state: Any,
    step: int | None,
) -> None:
    optimizer.opt_state = opt_state
    if step is not None:
        optimizer.step[...] = jnp.asarray(step, dtype=jnp.uint32)
