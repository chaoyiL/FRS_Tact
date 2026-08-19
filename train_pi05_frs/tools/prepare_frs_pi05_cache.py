#!/usr/bin/env python
"""Prepare one Pi0.5 FRS action cache per configured dataset."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from train_pi05_frs.tools.train_frs import (
    DEFAULT_CONFIG,
    load_config,
    resolve_local_path,
    resolve_url_or_local_path,
    source_cache_dir,
    validate_config,
)


@dataclass(frozen=True)
class Pi0ConfigSpec:
    pi05: bool
    action_dim: int
    action_horizon: int
    paligemma_variant: str
    action_expert_variant: str


def load_pi0(checkpoint: str, *, config: Pi0ConfigSpec):
    """Dependency-lazy Task 2 model loader, kept patchable for config tests."""
    from train_pi05_frs.pi05_cache import prepare as _prepare_boundary  # noqa: F401
    from lerobot.policies.pi05_jax import Pi0Config, load_pi0 as _load_pi0

    return _load_pi0(
        checkpoint,
        config=Pi0Config(
            pi05=config.pi05,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            paligemma_variant=config.paligemma_variant,
            action_expert_variant=config.action_expert_variant,
        ),
    )


def prepare_cache(**kwargs: object) -> Path:
    """Dependency-lazy Task 2 cache producer."""
    from train_pi05_frs.pi05_cache.prepare import prepare_cache as _prepare_cache

    return _prepare_cache(**kwargs)  # type: ignore[arg-type]


def _pi0_config(model_config: Mapping[str, Any]) -> Pi0ConfigSpec:
    return Pi0ConfigSpec(
        pi05=True,
        action_dim=int(model_config.get("action_dim", 32)),
        action_horizon=int(model_config.get("action_horizon", 50)),
        paligemma_variant=str(model_config.get("paligemma_variant", "gemma_2b")),
        action_expert_variant=str(model_config.get("action_expert_variant", "gemma_300m")),
    )


def _load_shared_model(config: Mapping[str, Any]):
    # Keep URL checkpoint strings out of pathlib: Path("gs://...") corrupts the scheme separator.
    checkpoint = resolve_url_or_local_path(str(config["checkpoint"]))
    return load_pi0(checkpoint, config=_pi0_config(config["model"]))


def checkpoint_smoke(config: Mapping[str, Any]) -> None:
    validate_config(config, check_paths=True)
    import jax

    devices = jax.devices()
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"JAX GPU device is required for the training pipeline; devices={devices}")
    model_config = config["model"]
    model = _load_shared_model(config)
    expected = (
        int(model_config.get("action_dim", 32)),
        int(model_config.get("action_horizon", 50)),
    )
    actual = (int(model.action_dim), int(model.action_horizon))
    if actual != expected or not bool(model.pi05):
        raise ValueError(
            "Pi0.5 checkpoint shape/config mismatch: "
            f"expected action_dim/horizon={expected}, got {actual}, pi05={model.pi05}"
        )
    print(
        f"Pi0.5 checkpoint smoke passed: devices={devices} action_dim={actual[0]} "
        f"action_horizon={actual[1]}",
        flush=True,
    )


def prepare_from_config(config: Mapping[str, Any]) -> list[Path]:
    validate_config(config, check_paths=True)
    checkpoint = resolve_url_or_local_path(str(config["checkpoint"]))
    datasets = config["datasets"]
    cache_config = config["action_cache"]
    model_config = config["model"]
    norm_stats_config = config["norm_stats"]
    camera_map = model_config["camera_map"]
    shared_model = _load_shared_model(config)

    outputs: list[Path] = []
    for source_index, source in enumerate(datasets):
        repo_id = str(source["repo_id"])
        output = source_cache_dir(str(cache_config["root"]), repo_id)
        dataset_root = resolve_local_path(str(source["root"]))
        print(f"prepare_source={source_index}:{repo_id} cache={output}", flush=True)
        prepare_cache(
            checkpoint_dir=checkpoint,
            cache_dir=output,
            dataset_repo_id=repo_id,
            dataset_root=dataset_root,
            dataset_revision=source.get("revision"),
            action_key=source.get("action_key"),
            rename_map=dict(source.get("rename_map", {})),
            camera_map=dict(camera_map),
            norm_stats_dir=resolve_url_or_local_path(str(norm_stats_config["dir"])),
            norm_stats_asset_id=str(norm_stats_config["asset_id"]),
            use_quantile_norm=norm_stats_config.get("use_quantile_norm", True),
            action_dim=int(model_config.get("action_dim", 32)),
            action_horizon=int(model_config.get("action_horizon", 50)),
            paligemma_variant=str(model_config.get("paligemma_variant", "gemma_2b")),
            action_expert_variant=str(model_config.get("action_expert_variant", "gemma_300m")),
            model_sample_steps=int(cache_config.get("model_sample_steps", 10)),
            reverse_steps=int(cache_config.get("reverse_steps", 50)),
            reverse_solver=str(cache_config.get("reverse_solver", "fireflow")),
            batch_size=int(cache_config.get("batch_size", 16)),
            load_workers=int(cache_config.get("load_workers", 4)),
            inference_seed=int(cache_config.get("inference_seed", 0)),
            split_seed=int(cache_config.get("split_seed", 42)),
            val_fraction=float(cache_config.get("val_fraction", 0.1)),
            frame_stride=int(cache_config.get("frame_stride", 3)),
            max_episodes=cache_config.get("max_episodes"),
            max_samples=cache_config.get("max_samples"),
            drop_tail_action_chunks=int(cache_config.get("drop_tail_action_chunks", 1)),
            flush_every=int(cache_config.get("flush_every", 8)),
            loaded_model=shared_model,
        )
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.checkpoint_smoke:
        checkpoint_smoke(config)
        return
    outputs = prepare_from_config(config)
    for output in outputs:
        print(f"action_cache={output}")


if __name__ == "__main__":
    main()
