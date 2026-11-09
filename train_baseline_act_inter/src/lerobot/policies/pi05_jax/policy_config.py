"""Load a native JAX/orbax pi0.5 checkpoint for inference.

NOT vendored: this is the trimmed counterpart of openpi's `src/openpi/policies/policy_config.py`.
Upstream's `create_trained_policy` wraps the model in a `Policy` object together with the full
input/output transform pipeline, because openpi serves policies over a websocket to a robot. FRS
never serves; it drives the model directly (`utils/pi05_source_model.py`) after building its own
`Observation`. So only the weight-loading half is kept here, following upstream's pattern exactly:

    model = config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

`load_norm_stats` additionally accepts remote URLs, which upstream's does not need (it always
reads a checkpoint that `download.maybe_download` already localized).
"""

from __future__ import annotations

import dataclasses
import pathlib

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp

from . import download
from . import normalize as _normalize
from .model import restore_params
from .pi0 import Pi0
from .pi0_config import Pi0Config


def resolve_checkpoint(path_or_url: str | pathlib.Path) -> pathlib.Path:
    """Resolve a local path, `gs://...` URL, or other fsspec URL to a local directory.

    Thin wrapper around `download.maybe_download` (see its docstring): caches remote checkpoints
    under `~/.cache/openpi` (override with `OPENPI_DATA_HOME`), matching where a real openpi
    install would put them, so this and an upstream openpi checkout can share a download cache.
    """
    return download.maybe_download(str(path_or_url))


def _reject_unused_params(config: Pi0Config, params: dict, checkpoint_dir: pathlib.Path) -> None:
    """Fail if the checkpoint contains parameters `config` has no slot for.

    `BaseModelConfig.load` defaults to `remove_extra_params=True`, which *silently drops* any
    checkpoint parameter the model structure does not declare. That is the right behaviour
    upstream (openpi always pairs a checkpoint with the `TrainConfig` that produced it), but here
    it hides a specific, expensive mistake: loading a **LoRA** fine-tune with a default
    `Pi0Config`. The LoRA weights get dropped, the frozen base weights load cleanly, nothing
    raises -- and you have silently reverted to the un-fine-tuned base model.
    """
    expected = nnx.split(nnx.eval_shape(config.create, jax.random.key(0)))[1].to_pure_dict()
    extra = sorted(
        set(flax.traverse_util.flatten_dict(params, sep="/"))
        - set(flax.traverse_util.flatten_dict(expected, sep="/"))
    )
    if not extra:
        return
    hint = ""
    if any("lora" in key for key in extra):
        hint = (
            " This looks like a LoRA fine-tune being loaded with a non-LoRA config: pass "
            "`paligemma_variant`/`action_expert_variant` matching the TrainConfig it was trained "
            "with (e.g. gemma_2b_lora / gemma_300m_lora). Loading it as-is would discard every "
            "LoRA weight and leave you with the frozen base model."
        )
    raise ValueError(
        f"checkpoint {checkpoint_dir} has {len(extra)} parameters this config cannot hold, "
        f"e.g. {extra[:4]}.{hint} Pass allow_extra_params=True to drop them on purpose."
    )


def load_pi0(
    checkpoint_dir: str | pathlib.Path,
    *,
    config: Pi0Config | None = None,
    dtype: jnp.dtype = jnp.bfloat16,
    allow_extra_params: bool = False,
) -> Pi0:
    """Load a pi0.5 (or pi0) model from a native JAX/orbax checkpoint directory.

    Args:
        checkpoint_dir: local path, or a URL `resolve_checkpoint`/`download.maybe_download`
            understands (e.g. `gs://openpi-assets/checkpoints/pi05_base`). Must contain a
            `params/` orbax checkpoint directory (i.e. pass the checkpoint root, not
            `.../params` itself -- this appends "params" the same way upstream's
            `create_trained_policy` does). A checkpoint written by `tools/train_pi05_jax.py`
            works too: pass `<checkpoint_base_dir>/<config>/<exp>/<step>`.
        config: model architecture config. Defaults to `Pi0Config(pi05=True)`, matching the
            official `pi05_base` checkpoint. **Loading your own fine-tune requires passing the
            same config it was trained with** -- in particular the LoRA variants, or the LoRA
            weights are silently dropped (see `_reject_unused_params`).
        dtype: dtype to restore params as. bfloat16 matches how openpi itself loads checkpoints
            for inference.
        allow_extra_params: skip that check and let `config.load` drop unmatched parameters.

    Returns:
        A `Pi0` nnx.Module with restored parameters, ready for `sample_actions` or for
        `frs.build_prefix_cache` + `frs.denoise_step`.
    """
    if config is None:
        config = Pi0Config(pi05=True)
    checkpoint_dir = resolve_checkpoint(checkpoint_dir)
    params = restore_params(checkpoint_dir / "params", dtype=dtype)
    if not allow_extra_params:
        _reject_unused_params(config, params, checkpoint_dir)
    return config.load(params)


def load_norm_stats(assets_dir: str | pathlib.Path, asset_id: str) -> dict[str, _normalize.NormStats]:
    """Load `<assets_dir>/<asset_id>/norm_stats.json`, as written by openpi training/tooling.

    `assets_dir` may be a local path or a URL `download.maybe_download` understands (e.g.
    `gs://openpi-assets/checkpoints/pi05_base/assets`, to reuse an official checkpoint's stats).
    The join happens on plain strings, not `pathlib.Path`: `Path("gs://x") / "y"` collapses the
    `//` and corrupts the URL -- the same trap `prepare_pi05.py:_is_local_path` documents.
    """
    joined = f"{str(assets_dir).rstrip('/')}/{asset_id}"
    return _normalize.load(download.maybe_download(joined))


@dataclasses.dataclass(frozen=True)
class Pi05CheckpointInfo:
    """Where the official checkpoints live, for reference (see pi05_frs_plan.md)."""

    base: str = "gs://openpi-assets/checkpoints/pi05_base"
    droid: str = "gs://openpi-assets/checkpoints/pi05_droid"
