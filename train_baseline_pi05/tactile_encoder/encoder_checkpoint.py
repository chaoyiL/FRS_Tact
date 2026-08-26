"""Load the frozen tactile encoder used by online FRS deployment."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util


PARAMS_NAME = "params.npz"
CHECKPOINT_NAME = "checkpoint.json"


def _resolve_checkpoint_file(directory: Path, filename: str) -> Path:
    """Accept both native names and Hugging Face content-hashed checkpoint names."""

    expected = directory / filename
    if expected.exists():
        return expected
    suffix = expected.suffix
    candidates = sorted(directory.glob(f"{expected.stem}-*{suffix}"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise FileNotFoundError(f"Multiple candidates for {filename} in {directory}: {names}")
    return expected


def _numpy_to_leaf(array: np.ndarray) -> Any:
    """Restore a checkpoint leaf, including legacy bfloat16 blobs."""

    array = np.asarray(array)
    if array.dtype.kind == "V" and array.dtype.itemsize == 2:
        bits = np.asarray(array).view(np.dtype("<u2"))
        return jax.lax.bitcast_convert_type(jnp.asarray(bits), jnp.bfloat16).astype(jnp.float32)
    if array.dtype.kind == "V":
        raise TypeError(f"Unsupported void dtype {array.dtype} in checkpoint leaf.")
    return jnp.asarray(array)


def _load_checkpoint(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with (directory / CHECKPOINT_NAME).open(encoding="utf-8") as file:
        metadata = json.load(file)
    restored_flat: dict[tuple[str, ...], Any] = {}
    with np.load(_resolve_checkpoint_file(directory, PARAMS_NAME)) as archive:
        for index, path_name in enumerate(metadata["parameter_paths"]):
            restored_flat[tuple(path_name.split("/"))] = _numpy_to_leaf(archive[f"p{index:05d}"])
    return traverse_util.unflatten_dict(restored_flat), metadata


@dataclasses.dataclass(frozen=True)
class TactileEncoderBundle:
    params: dict[str, Any]
    metadata: dict[str, Any]


def load_tactile_encoder(checkpoint_dir: str | Path) -> TactileEncoderBundle:
    params, metadata = _load_checkpoint(Path(checkpoint_dir))
    return TactileEncoderBundle(params=params, metadata=metadata)
