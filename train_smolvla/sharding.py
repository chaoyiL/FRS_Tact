from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def create_data_parallel_mesh(devices: Sequence[jax.Device] | None = None) -> Mesh:
    devices = tuple(jax.devices() if devices is None else devices)
    if not devices:
        raise RuntimeError("JAX did not expose any devices")
    return Mesh(np.asarray(devices), ("data",))


def replicate_tree(tree: Any, mesh: Mesh) -> Any:
    replicated = NamedSharding(mesh, P())
    return jax.tree.map(
        lambda value: jax.device_put(value, replicated) if hasattr(value, "shape") else value,
        tree,
    )


def shard_batch(batch: Mapping[str, Any], mesh: Mesh) -> dict[str, Any]:
    device_count = mesh.size
    result = {}
    for key, value in batch.items():
        if value.ndim == 0:
            result[key] = jax.device_put(value, NamedSharding(mesh, P()))
            continue
        if value.shape[0] % device_count:
            raise ValueError(
                f"batch dimension for {key!r} ({value.shape[0]}) is not divisible by {device_count} devices"
            )
        result[key] = jax.device_put(value, NamedSharding(mesh, P("data", *([None] * (value.ndim - 1)))))
    return result
