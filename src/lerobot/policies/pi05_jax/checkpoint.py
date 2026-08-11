"""Load a native JAX/orbax pi0.5 checkpoint.

NOT vendored from openpi -- this is new glue code, following the exact pattern openpi's own
`openpi.policies.policy_config.create_trained_policy` uses (see pi05_frs_plan.md's API notes),
but skipping everything unrelated to loading params (transforms/normalization/policy wrapping --
FRS builds its own Observation and normalization, see README.md's TODO list).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax.numpy as jnp

from . import download
from .model import restore_params
from .pi0 import Pi0
from .pi0_config import Pi0Config


def resolve_checkpoint(path_or_url: str | Path) -> Path:
    """Resolve a local path, `gs://...` URL, or other fsspec URL to a local directory.

    Thin wrapper around `download.maybe_download` (see its docstring): caches remote checkpoints
    under `~/.cache/openpi` (override with `OPENPI_DATA_HOME`), matching where a real openpi
    install would put them, so this and an upstream openpi checkout can share a download cache.
    """
    return download.maybe_download(str(path_or_url))


def load_pi0(
    checkpoint_dir: str | Path,
    *,
    config: Pi0Config | None = None,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Pi0:
    """Load a pi0.5 (or pi0) model from a native JAX/orbax checkpoint directory.

    Args:
        checkpoint_dir: local path, or a URL `resolve_checkpoint`/`download.maybe_download`
            understands (e.g. `gs://openpi-assets/checkpoints/pi05_base`). Must contain a
            `params/` orbax checkpoint directory (i.e. pass the checkpoint root, not
            `.../params` itself -- this appends "params" the same way
            `openpi.policies.policy_config.create_trained_policy` does).
        config: model architecture config. Defaults to `Pi0Config(pi05=True)`, matching the
            official `pi05_base` checkpoint. Pass an explicit config if loading a pi0 (not
            pi0.5) checkpoint, or one with non-default `action_dim`/`action_horizon`/variants.
        dtype: dtype to restore params as. bfloat16 matches how openpi itself loads checkpoints
            for inference (see `policy_config.create_trained_policy`).

    Returns:
        A `Pi0` nnx.Module with restored parameters, ready for `sample_actions` /
        `build_prefix_cache` + `denoise_step` (see pi0.py).
    """
    if config is None:
        config = Pi0Config(pi05=True)
    checkpoint_dir = resolve_checkpoint(checkpoint_dir)
    params = restore_params(checkpoint_dir / "params", dtype=dtype)
    return config.load(params)


@dataclasses.dataclass(frozen=True)
class Pi05CheckpointInfo:
    """Where the official checkpoints live, for reference (see pi05_frs_plan.md)."""

    base: str = "gs://openpi-assets/checkpoints/pi05_base"
    droid: str = "gs://openpi-assets/checkpoints/pi05_droid"
