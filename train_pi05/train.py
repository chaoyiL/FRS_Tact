"""Standalone YAML launcher for pure-vision JAX pi0.5 fine-tuning."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_pi05.yaml"
PURE_VISION_PROFILES = {"pi05_single", "pi05_bi", "pi05_bi_no_state"}


def _fit_schedule_steps(total_steps: int, configured_warmup_steps: int) -> tuple[int, int]:
    """Return a valid warmup/decay pair for any requested training length."""

    if total_steps <= 0:
        raise ValueError(f"training.steps must be positive, got {total_steps}")
    warmup_steps = min(configured_warmup_steps, max(total_steps - 1, 0))
    decay_steps = max(total_steps, warmup_steps + 1)
    return warmup_steps, decay_steps


def _feature_dim(features: dict[str, Any], key: str, *, dataset_index: int) -> int:
    feature = features.get(key)
    if not isinstance(feature, dict):
        raise ValueError(f"datasets[{dataset_index}] is missing required feature {key!r}")
    shape = feature.get("shape")
    if not isinstance(shape, list) or not shape or not isinstance(shape[-1], int):
        raise ValueError(f"datasets[{dataset_index}] feature {key!r} has invalid shape: {shape!r}")
    return shape[-1]


def _validate_dataset_contract(
    config: dict[str, Any], source: dict[str, Any], info_path: Path, *, dataset_index: int
) -> None:
    """Validate the optional state/action/camera contract from LeRobot metadata."""
    contract = config.get("dataset_contract")
    if contract is None:
        return
    if not isinstance(contract, dict):
        raise ValueError("dataset_contract must be a mapping")
    with info_path.open(encoding="utf-8") as info_file:
        info = json.load(info_file)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"datasets[{dataset_index}] meta/info.json has no features mapping")

    state_key = str(contract.get("state_key", "observation.state"))
    expected_state_dim = int(contract["state_dim"])
    actual_state_dim = _feature_dim(features, state_key, dataset_index=dataset_index)
    if actual_state_dim != expected_state_dim:
        raise ValueError(
            f"datasets[{dataset_index}] {state_key} must be {expected_state_dim}D, got {actual_state_dim}D"
        )

    action_key = str(source.get("action_key", "action"))
    expected_action_dim = int(contract["action_dim"])
    actual_action_dim = _feature_dim(features, action_key, dataset_index=dataset_index)
    if actual_action_dim != expected_action_dim:
        raise ValueError(
            f"datasets[{dataset_index}] {action_key} must be {expected_action_dim}D, got {actual_action_dim}D"
        )

    image_keys = contract.get("image_keys", [])
    if not isinstance(image_keys, list) or not all(isinstance(key, str) for key in image_keys):
        raise ValueError("dataset_contract.image_keys must be a list of feature names")
    missing_images = [key for key in image_keys if key not in features]
    if missing_images:
        raise ValueError(f"datasets[{dataset_index}] is missing required image features: {missing_images}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    if config.get("profile") not in PURE_VISION_PROFILES:
        raise ValueError(f"profile must be one of {sorted(PURE_VISION_PROFILES)}")
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list")
    for index, source in enumerate(datasets):
        if not isinstance(source, dict) or not source.get("repo_id") or not source.get("root"):
            raise ValueError(f"datasets[{index}] requires repo_id and root")
        root = Path(str(source["root"])).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"datasets[{index}] root does not exist: {root}")
        info = root / "meta" / "info.json"
        if not info.is_file():
            raise FileNotFoundError(f"datasets[{index}] is not LeRobot v3: {info}")
        _validate_dataset_contract(config, source, info, dataset_index=index)
    training = config.get("training")
    if not isinstance(training, dict) or not training.get("output"):
        raise ValueError("training.output is required")
    if bool(training.get("resume")) and bool(training.get("overwrite")):
        raise ValueError("training.resume and training.overwrite cannot both be true")
    validation = config.get("validation") or {}
    if not isinstance(validation, dict):
        raise ValueError("validation must be a mapping")
    validation_ratio = float(validation.get("ratio", 0.0))
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation.ratio must satisfy 0 <= ratio < 1")
    if validation_ratio > 0 and int(validation.get("interval", 2000)) <= 0:
        raise ValueError("validation.interval must be positive")
    return config


def _make_source(openpi_config, item: dict[str, Any], episodes: list[int] | None):
    return openpi_config.DatasetSource(
        repo_id=str(item["repo_id"]),
        root=str(Path(str(item["root"])).expanduser().resolve()),
        revision=item.get("revision"),
        episodes=episodes,
        action_key=str(item.get("action_key", "action")),
    )


def _split_sources(raw: dict[str, Any], openpi_config):
    validation = raw.get("validation") or {}
    ratio = float(validation.get("ratio", 0.0))
    split_seed = int(validation.get("seed", raw["training"].get("seed", 42)))
    train_sources = []
    validation_sources = []

    for index, item in enumerate(raw["datasets"]):
        info_path = Path(str(item["root"])).expanduser() / "meta" / "info.json"
        with info_path.open(encoding="utf-8") as info_file:
            info = json.load(info_file)
        all_episodes = item.get("episodes")
        episode_pool = (
            [int(episode) for episode in all_episodes]
            if all_episodes is not None
            else list(range(int(info["total_episodes"])))
        )

        if ratio == 0.0:
            train_sources.append(_make_source(openpi_config, item, episode_pool))
            continue
        if len(episode_pool) < 2:
            raise ValueError(f"datasets[{index}] needs at least 2 episodes for a validation split")

        shuffled = episode_pool.copy()
        random.Random(split_seed + index).shuffle(shuffled)
        validation_count = min(len(shuffled) - 1, max(1, int(len(shuffled) * ratio + 0.5)))
        held_out = set(shuffled[:validation_count])
        train_episodes = sorted(episode for episode in episode_pool if episode not in held_out)
        validation_episodes = sorted(held_out)
        train_sources.append(_make_source(openpi_config, item, train_episodes))
        validation_sources.append(_make_source(openpi_config, item, validation_episodes))

    return tuple(train_sources), tuple(validation_sources)


def build_train_config(raw: dict[str, Any]):
    from openpi.training import config as openpi_config
    from openpi.training import optimizer, weight_loaders

    base = openpi_config.get_config(str(raw["profile"]))
    sources, validation_sources = _split_sources(raw, openpi_config)
    norm = raw.get("norm_stats") or {}
    assets_dir = str(norm.get("dir", "./assets"))
    if "://" not in assets_dir:
        assets_path = Path(assets_dir).expanduser()
        if not assets_path.is_absolute():
            assets_path = Path(__file__).resolve().parent / assets_path
        assets_dir = str(assets_path.resolve())
    assets = replace(
        base.data.assets,
        assets_dir=assets_dir,
        asset_id=str(norm.get("asset_id", sources[0].repo_id)),
    )
    contract = raw.get("dataset_contract") or {}
    visual_keys = tuple(str(key) for key in contract.get("image_keys", ())) or None
    data = replace(
        base.data,
        repo_id=sources[0].repo_id,
        sources=sources,
        assets=assets,
        visual_keys=visual_keys,
    )
    validation_data = (
        replace(
            base.data,
            repo_id=validation_sources[0].repo_id,
            sources=validation_sources,
            assets=assets,
            visual_keys=visual_keys,
        )
        if validation_sources
        else None
    )
    training = raw["training"]
    wandb = raw.get("wandb") or {}
    validation = raw.get("validation") or {}
    steps = int(training.get("steps", base.num_train_steps))
    lr = base.lr_schedule
    if isinstance(lr, optimizer.CosineDecaySchedule):
        warmup_steps, decay_steps = _fit_schedule_steps(steps, lr.warmup_steps)
        lr = replace(lr, warmup_steps=warmup_steps, decay_steps=decay_steps)
    return replace(
        base,
        data=data,
        validation_data=validation_data,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(raw["checkpoint"])),
        lr_schedule=lr,
        checkpoint_dir_override=str(training["output"]),
        exp_name=str(wandb.get("exp_name", base.exp_name)),
        project_name=str(wandb.get("project", base.project_name)),
        wandb_enabled=bool(wandb.get("enable", True)),
        batch_size=int(training.get("batch_size", base.batch_size)),
        num_workers=int(training.get("num_workers", base.num_workers)),
        num_train_steps=steps,
        log_interval=int(training.get("log_interval", base.log_interval)),
        validation_interval=int(validation.get("interval", base.validation_interval)),
        save_interval=int(training.get("save_interval", base.save_interval)),
        keep_period=training.get("keep_period", base.keep_period),
        seed=int(training.get("seed", base.seed)),
        fsdp_devices=int(training.get("fsdp_devices", base.fsdp_devices)),
        overwrite=bool(training.get("overwrite", False)),
        resume=bool(training.get("resume", False)),
    )


def norm_stats_path(raw: dict[str, Any]) -> Path:
    """Return the local norm_stats.json path selected by a YAML config."""

    norm = raw.get("norm_stats") or {}
    assets_dir = str(norm.get("dir", "./assets"))
    if "://" in assets_dir:
        raise ValueError("automatic norm stats generation requires a local norm_stats.dir")
    assets_path = Path(assets_dir).expanduser()
    if not assets_path.is_absolute():
        assets_path = Path(__file__).resolve().parent / assets_path
    asset_id = str(norm.get("asset_id", raw["datasets"][0]["repo_id"]))
    return (assets_path / asset_id / "norm_stats.json").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument("--print-norm-stats", action="store_true")
    args = parser.parse_args()
    raw = load_config(args.config.expanduser().resolve())
    output = Path(str(raw["training"]["output"])).expanduser().resolve()
    if args.print_output:
        print(output)
    if args.print_norm_stats:
        print(norm_stats_path(raw))
    if args.check:
        return
    from tools.train_core import main as train_main

    train_main(build_train_config(raw))


if __name__ == "__main__":
    main()
