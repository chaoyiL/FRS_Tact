#!/usr/bin/env python
"""Prepare one pi0.5 FRS action cache per dataset from a YAML config.

pi0.5 analogue of tools/prepare_frs_caches.py -- same config file
(configs/train_frs_pick_tube_pi05.yaml), same output cache format (utils/cache.py), just backed
by prepare_pi05.prepare_cache() instead of prepare.prepare_cache().

UNTESTED, like the rest of this branch's pi0.5 integration -- see
src/lerobot/policies/pi05_jax/README.md and pi05_frs_plan.md.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_pi05 import prepare_cache

DEFAULT_CONFIG = ROOT / "configs" / "train_frs_pick_tube_pi05.yaml"


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return value


def source_cache_dir(cache_root: str | Path, repo_id: str) -> Path:
    parts = [part for part in str(repo_id).split("/") if part not in ("", ".", "..")]
    if not parts:
        raise ValueError(f"invalid repo id: {repo_id!r}")
    return Path(cache_root).expanduser().joinpath(*parts)


def prepare_from_config(config: Mapping[str, Any]) -> list[Path]:
    # Kept as a plain string, not wrapped in pathlib.Path: Path("gs://bucket/x") silently
    # collapses the "//" to "/", corrupting URL checkpoints (see prepare_pi05.py:_is_local_path).
    checkpoint = str(config["checkpoint"])
    datasets = config.get("datasets") or []
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("config.datasets must be a non-empty list")
    cache_config = config.get("action_cache") or {}
    if not isinstance(cache_config, Mapping) or not cache_config.get("root"):
        raise ValueError("config.action_cache.root is required")
    cache_root = Path(str(cache_config["root"])).expanduser()

    model_config = config.get("model") or {}
    camera_map = model_config.get("camera_map")
    if not isinstance(camera_map, Mapping) or not camera_map:
        raise ValueError(
            "config.model.camera_map is required: map pi0.5 image keys "
            "(base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb) to dataset observation keys. "
            "See configs/train_frs_pick_tube_pi05.yaml and pi05_frs_plan.md."
        )
    norm_stats_config = config.get("norm_stats") or {}
    if not norm_stats_config.get("dir") or not norm_stats_config.get("asset_id"):
        raise ValueError(
            "config.norm_stats.dir and config.norm_stats.asset_id are required. There is no "
            "default -- see pi05_frs_plan.md for why (no norm stats exist for a brand-new "
            "dataset in the pretrained pi05_base checkpoint's assets)."
        )

    outputs: list[Path] = []
    for source_index, source in enumerate(datasets):
        if not isinstance(source, Mapping):
            raise ValueError(f"datasets[{source_index}] must be a mapping")
        repo_id = str(source["repo_id"])
        output = source_cache_dir(cache_root, repo_id)
        root_value = source.get("root")
        dataset_root = None if root_value in (None, "") else Path(str(root_value)).expanduser()
        print(f"prepare_source={source_index}:{repo_id} cache={output}", flush=True)
        prepare_cache(
            checkpoint_dir=checkpoint,
            cache_dir=output,
            dataset_repo_id=repo_id,
            dataset_root=dataset_root,
            dataset_revision=source.get("revision"),
            action_key=source.get("action_key"),
            rename_map=dict(source.get("rename_map") or {}),
            camera_map=dict(camera_map),
            norm_stats_dir=str(norm_stats_config["dir"]),
            norm_stats_asset_id=str(norm_stats_config["asset_id"]),
            use_quantile_norm=bool(norm_stats_config.get("use_quantile_norm", True)),
            action_dim=int(model_config.get("action_dim", 32)),
            action_horizon=int(model_config.get("action_horizon", 50)),
            model_sample_steps=int(cache_config.get("model_sample_steps", 10)),
            reverse_steps=int(cache_config.get("reverse_steps", 50)),
            reverse_solver=str(cache_config.get("reverse_solver", "fireflow")),
            batch_size=int(cache_config.get("batch_size", 16)),
            inference_seed=int(cache_config.get("inference_seed", 0)),
            split_seed=int(cache_config.get("split_seed", 42)),
            val_fraction=float(cache_config.get("val_fraction", 0.1)),
            frame_stride=int(cache_config.get("frame_stride", 3)),
            max_episodes=(None if cache_config.get("max_episodes") is None else int(cache_config["max_episodes"])),
            max_samples=(None if cache_config.get("max_samples") is None else int(cache_config["max_samples"])),
            drop_tail_action_chunks=int(cache_config.get("drop_tail_action_chunks", 1)),
            flush_every=int(cache_config.get("flush_every", 8)),
        )
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    outputs = prepare_from_config(load_config(args.config))
    for output in outputs:
        print(f"action_cache={output}")


if __name__ == "__main__":
    main()
