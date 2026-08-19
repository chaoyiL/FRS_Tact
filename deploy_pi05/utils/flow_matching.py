"""Base-model-agnostic flow-matching helpers, shared by every FRS action_cache producer.

`deterministic_noise` and `inversion_mse` only touch plain jax/numpy arrays, so they live here
rather than next to any one base model's velocity-field glue (`utils/pi05_source_model.py`).
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np


def deterministic_noise(indices: Sequence[int], shape: tuple[int, int], *, seed: int) -> jax.Array:
    base_key = jax.random.key(seed)
    index_arr = jnp.asarray(list(indices), dtype=jnp.int32)

    def one(index: jax.Array) -> jax.Array:
        return jax.random.normal(jax.random.fold_in(base_key, index), shape, dtype=jnp.float32)

    return jax.vmap(one)(index_arr)


def inversion_mse(x_base: jax.Array, initial_noise: jax.Array) -> np.ndarray:
    axes = tuple(range(1, x_base.ndim))
    return np.asarray(jax.device_get(jnp.mean(jnp.square(x_base - initial_noise), axis=axes)), dtype=np.float32)
