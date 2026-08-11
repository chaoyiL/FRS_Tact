"""Minimal stand-in for openpi.training.sharding, trimmed to what gemma.py/siglip.py call.

NOT vendored from openpi -- openpi's real training/sharding.py sets up multi-host FSDP device
meshes for distributed training. FRS only ever runs pi0.5 for single-device inference (producing
an action_cache, see ../../../../prepare.py), so `activation_sharding_constraint` is always a
no-op here: this is exactly what upstream's own implementation reduces to when no mesh has been
set via `sharding.set_mesh(...)` (see `_MeshState.active_mesh is None` in openpi's
src/openpi/training/sharding.py, commit 15a9616, 2026-06-16) -- we simply never call that, so
there's nothing to vendor. If FRS ever needs multi-device pi0.5 inference, this is the place to
add a real mesh instead of guessing at one.
"""

from __future__ import annotations

from typing import TypeVar

PyTree = TypeVar("PyTree")


def activation_sharding_constraint(pytree: PyTree) -> PyTree:
    return pytree
