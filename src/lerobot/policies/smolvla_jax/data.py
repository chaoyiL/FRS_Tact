from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REQUIRED_TRAINING_KEYS = (
    "images",
    "image_masks",
    "language_tokens",
    "language_masks",
    "state",
    "actions",
)


def load_training_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    missing = set(REQUIRED_TRAINING_KEYS) - set(data)
    if missing:
        raise ValueError(f"training NPZ is missing keys: {sorted(missing)}")
    lengths = {key: value.shape[0] for key, value in data.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"all training arrays need the same first dimension, got {lengths}")
    return data


def numpy_batches(
    data: Mapping[str, np.ndarray],
    batch_size: int,
    *,
    seed: int = 0,
) -> Iterator[dict[str, jax.Array]]:
    size = next(iter(data.values())).shape[0]
    if size < batch_size:
        raise ValueError(f"dataset contains {size} samples, smaller than batch size {batch_size}")
    rng = np.random.default_rng(seed)
    while True:
        indices = rng.integers(0, size, size=batch_size)
        yield {key: jnp.asarray(value[indices]) for key, value in data.items()}
