from __future__ import annotations

import dataclasses
import json
import pathlib
import pickle
import uuid
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from utils.cache import atomic_write_json
from tactile_encoder.utils.model import TactileClipConfig
from tactile_encoder.utils.model import encode_tactile_embedding
from tactile_encoder.utils.model import tactile_clip_config_from_dict

PARAMS_NAME = "params.npz"
OPT_STATE_NAME = "opt_state.npz"
OPT_STATE_TREEDEF_NAME = "opt_state.treedef.pkl"
MEMORY_BANK_NAME = "memory_bank.npz"
CHECKPOINT_NAME = "checkpoint.json"

_MEMORY_BANK_ARRAY_KEYS = (
    "keys",
    "future_dataset_index",
    "episode_index",
    "side_id",
    "valid",
    "ptr",
)


def _path_name(path: tuple[Any, ...]) -> str:
    return "/".join(str(part) for part in path)


def _flatten_params(params: dict[str, Any]) -> list[tuple[tuple[Any, ...], Any]]:
    flat = traverse_util.flatten_dict(params)
    return sorted(flat.items(), key=lambda item: _path_name(item[0]))


def _atomic_savez(path: pathlib.Path, arrays: dict[str, np.ndarray]) -> None:
    # np.savez appends ".npz" unless the path already ends with it.
    temporary = path.parent / (path.name + ".writing.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def _leaf_to_numpy(leaf: Any) -> np.ndarray:
    """Convert a pytree leaf to a NumPy array that round-trips through npz."""

    if isinstance(leaf, (np.bool_, bool)):
        return np.asarray(leaf, dtype=np.bool_)
    if isinstance(leaf, (np.integer, int)):
        return np.asarray(leaf, dtype=np.int32)
    if isinstance(leaf, (np.floating, float)):
        return np.asarray(leaf, dtype=np.float32)

    # JAX bfloat16 becomes dtype('|V2') under np.asarray; cast to float32 for storage.
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
    """Restore a leaf saved by ``_leaf_to_numpy``, including legacy |V2 bfloat16 blobs."""

    array = np.asarray(array)
    # Legacy checkpoints: np.asarray(bfloat16) wrote void[2] payloads.
    if array.dtype.kind == "V" and array.dtype.itemsize == 2:
        bits = np.asarray(array).view(np.dtype("<u2"))
        return jax.lax.bitcast_convert_type(jnp.asarray(bits), jnp.bfloat16).astype(jnp.float32)
    if array.dtype.kind == "V":
        raise TypeError(f"Unsupported void dtype {array.dtype} in checkpoint leaf.")
    return jnp.asarray(array)


def save_memory_bank(
    directory: pathlib.Path,
    memory_bank: dict[str, Any] | None,
    *,
    filename: str = MEMORY_BANK_NAME,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if memory_bank is None or int(np.asarray(memory_bank["keys"]).shape[0]) == 0:
        if path.exists():
            path.unlink()
        return
    arrays = {key: _leaf_to_numpy(memory_bank[key]) for key in _MEMORY_BANK_ARRAY_KEYS}
    _atomic_savez(path, arrays)


def load_memory_bank(
    directory: str | pathlib.Path,
    *,
    filename: str = MEMORY_BANK_NAME,
) -> dict[str, Any] | None:
    directory = pathlib.Path(directory)
    path = directory / filename
    if not path.exists():
        return None
    with np.load(path) as archive:
        return {key: _numpy_to_leaf(archive[key]) for key in _MEMORY_BANK_ARRAY_KEYS}


def save_checkpoint(
    directory: pathlib.Path,
    params: dict[str, Any],
    *,
    epoch: int,
    metrics: dict[str, float],
    model_id: str,
    config: TactileClipConfig,
    extra_metadata: dict[str, Any] | None = None,
    opt_state: Any | None = None,
    memory_bank: dict[str, Any] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    params_name = f"params-{generation}.npz"
    opt_state_name = f"opt_state-{generation}.npz"
    opt_tree_name = f"opt_state-{generation}.treedef.pkl"
    memory_bank_name = f"memory_bank-{generation}.npz"
    ordered = _flatten_params(params)
    arrays = {f"p{index:05d}": _leaf_to_numpy(value) for index, (_, value) in enumerate(ordered)}
    _atomic_savez(directory / params_name, arrays)
    has_memory_bank = (
        memory_bank is not None and int(np.asarray(memory_bank["keys"]).shape[0]) > 0
    )
    metadata: dict[str, Any] = {
        "version": 4,
        "epoch": int(epoch),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "clip_model_id": model_id,
        "tactile_backbone": "resnet18",
        "tactile_clip_config": dataclasses.asdict(config),
        "parameter_paths": [_path_name(path) for path, _ in ordered],
        "has_opt_state": opt_state is not None,
        "has_memory_bank": has_memory_bank,
        "params_file": params_name,
    }
    if opt_state is not None:
        leaves, treedef = jax.tree_util.tree_flatten(opt_state)
        opt_arrays = {f"p{index:05d}": _leaf_to_numpy(leaf) for index, leaf in enumerate(leaves)}
        _atomic_savez(directory / opt_state_name, opt_arrays)
        treedef_path = directory / opt_tree_name
        temporary = treedef_path.with_suffix(treedef_path.suffix + ".tmp")
        with temporary.open("wb") as file:
            pickle.dump(treedef, file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(treedef_path)
        metadata["opt_state_leaf_count"] = len(leaves)
        metadata["opt_state_file"] = opt_state_name
        metadata["opt_state_treedef_file"] = opt_tree_name
    if has_memory_bank:
        save_memory_bank(directory, memory_bank, filename=memory_bank_name)
        metadata["memory_bank_file"] = memory_bank_name
    if extra_metadata is not None:
        metadata["extra_metadata"] = extra_metadata
    atomic_write_json(directory / CHECKPOINT_NAME, metadata)
    keep = {params_name, opt_state_name, opt_tree_name, memory_bank_name}
    for pattern in (
        "params-*.npz",
        "opt_state-*.npz",
        "opt_state-*.treedef.pkl",
        "memory_bank-*.npz",
    ):
        for path in directory.glob(pattern):
            if path.name not in keep:
                path.unlink(missing_ok=True)


def load_checkpoint(directory: str | pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = pathlib.Path(directory)
    with (directory / CHECKPOINT_NAME).open(encoding="utf-8") as file:
        metadata = json.load(file)
    restored_flat: dict[tuple[str, ...], Any] = {}
    params_path = directory / str(metadata.get("params_file", PARAMS_NAME))
    with np.load(params_path) as archive:
        for index, path_name in enumerate(metadata["parameter_paths"]):
            restored_flat[tuple(path_name.split("/"))] = _numpy_to_leaf(archive[f"p{index:05d}"])
    return traverse_util.unflatten_dict(restored_flat), metadata


def load_train_state(
    directory: str | pathlib.Path,
) -> tuple[dict[str, Any], Any | None, dict[str, Any], dict[str, Any] | None]:
    """Load params, optional optimizer state, metadata, and optional memory bank."""

    directory = pathlib.Path(directory)
    params, metadata = load_checkpoint(directory)
    opt_state = None
    opt_state_path = directory / str(metadata.get("opt_state_file", OPT_STATE_NAME))
    treedef_path = directory / str(
        metadata.get("opt_state_treedef_file", OPT_STATE_TREEDEF_NAME)
    )
    if metadata.get("has_opt_state"):
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
        except Exception as exc:  # noqa: BLE001 - checkpoint corruption must be explicit
            raise RuntimeError(f"failed to restore optimizer state from {directory}") from exc
    memory_bank = None
    if metadata.get("has_memory_bank"):
        memory_bank_filename = str(metadata.get("memory_bank_file", MEMORY_BANK_NAME))
        memory_bank = load_memory_bank(directory, filename=memory_bank_filename)
        if memory_bank is None:
            raise FileNotFoundError(
                f"checkpoint reports a memory bank but {memory_bank_filename} is missing "
                f"in {directory}"
            )
    return params, opt_state, metadata, memory_bank


@dataclasses.dataclass(frozen=True)
class TactileEncoderBundle:
    params: dict[str, Any]
    metadata: dict[str, Any]

    def encode(self, tactile_images, *, train: bool = False):
        config = tactile_clip_config_from_dict(self.metadata["tactile_clip_config"])
        embedding, _ = encode_tactile_embedding(
            self.params,
            jnp.asarray(tactile_images, dtype=jnp.float32),
            train=train,
            config=config,
        )
        return embedding


def load_tactile_encoder(checkpoint_dir: str | pathlib.Path) -> TactileEncoderBundle:
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    params, metadata = load_checkpoint(checkpoint_dir)
    return TactileEncoderBundle(params=params, metadata=metadata)
