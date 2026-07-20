from __future__ import annotations

import dataclasses
import json
import pathlib
import pickle
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


def save_memory_bank(directory: pathlib.Path, memory_bank: dict[str, Any] | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MEMORY_BANK_NAME
    if memory_bank is None or int(np.asarray(memory_bank["keys"]).shape[0]) == 0:
        if path.exists():
            path.unlink()
        return
    arrays = {key: _leaf_to_numpy(memory_bank[key]) for key in _MEMORY_BANK_ARRAY_KEYS}
    _atomic_savez(path, arrays)


def load_memory_bank(directory: str | pathlib.Path) -> dict[str, Any] | None:
    directory = pathlib.Path(directory)
    path = directory / MEMORY_BANK_NAME
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
    ordered = _flatten_params(params)
    arrays = {f"p{index:05d}": _leaf_to_numpy(value) for index, (_, value) in enumerate(ordered)}
    _atomic_savez(directory / PARAMS_NAME, arrays)
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
    }
    if opt_state is not None:
        leaves, treedef = jax.tree_util.tree_flatten(opt_state)
        opt_arrays = {f"p{index:05d}": _leaf_to_numpy(leaf) for index, leaf in enumerate(leaves)}
        _atomic_savez(directory / OPT_STATE_NAME, opt_arrays)
        treedef_path = directory / OPT_STATE_TREEDEF_NAME
        temporary = treedef_path.with_suffix(treedef_path.suffix + ".tmp")
        with temporary.open("wb") as file:
            pickle.dump(treedef, file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(treedef_path)
        metadata["opt_state_leaf_count"] = len(leaves)
    save_memory_bank(directory, memory_bank if has_memory_bank else None)
    if extra_metadata is not None:
        metadata["extra_metadata"] = extra_metadata
    atomic_write_json(directory / CHECKPOINT_NAME, metadata)


def load_checkpoint(directory: str | pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = pathlib.Path(directory)
    with (directory / CHECKPOINT_NAME).open(encoding="utf-8") as file:
        metadata = json.load(file)
    restored_flat: dict[tuple[str, ...], Any] = {}
    with np.load(directory / PARAMS_NAME) as archive:
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
    opt_state_path = directory / OPT_STATE_NAME
    treedef_path = directory / OPT_STATE_TREEDEF_NAME
    if metadata.get("has_opt_state") and opt_state_path.exists() and treedef_path.exists():
        try:
            with treedef_path.open("rb") as file:
                treedef = pickle.load(file)
            with np.load(opt_state_path) as archive:
                leaf_count = int(metadata.get("opt_state_leaf_count", len(archive.files)))
                leaves = [_numpy_to_leaf(archive[f"p{index:05d}"]) for index in range(leaf_count)]
            opt_state = jax.tree_util.tree_unflatten(treedef, leaves)
        except Exception as exc:  # noqa: BLE001 - fall back to params-only resume
            print(
                f"warning: failed to restore optimizer state from {directory}: {exc}; "
                "reinitializing Adam state.",
                flush=True,
            )
            opt_state = None
    memory_bank = None
    if metadata.get("has_memory_bank"):
        memory_bank = load_memory_bank(directory)
        if memory_bank is None:
            print(
                f"warning: checkpoint reports has_memory_bank but {MEMORY_BANK_NAME} is missing "
                f"in {directory}; starting with an empty bank.",
                flush=True,
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
