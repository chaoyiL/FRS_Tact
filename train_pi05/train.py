"""Standalone YAML launcher for pure-vision JAX pi0.5 fine-tuning."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_pi05.yaml"
PURE_VISION_PROFILES = {"pi05_single", "pi05_bi", "pi05_bi_no_state"}


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
    training = config.get("training")
    if not isinstance(training, dict) or not training.get("output"):
        raise ValueError("training.output is required")
    if bool(training.get("resume")) and bool(training.get("overwrite")):
        raise ValueError("training.resume and training.overwrite cannot both be true")
    return config


def build_train_config(raw: dict[str, Any]):
    from openpi.training import config as openpi_config
    from openpi.training import optimizer, weight_loaders

    base = openpi_config.get_config(str(raw["profile"]))
    sources = tuple(
        openpi_config.DatasetSource(
            repo_id=str(item["repo_id"]),
            root=str(Path(str(item["root"])).expanduser().resolve()),
            revision=item.get("revision"),
            episodes=item.get("episodes"),
            action_key=str(item.get("action_key", "action")),
        )
        for item in raw["datasets"]
    )
    norm = raw.get("norm_stats") or {}
    assets = replace(
        base.data.assets,
        assets_dir=str(norm.get("dir", "./assets")),
        asset_id=str(norm.get("asset_id", sources[0].repo_id)),
    )
    data = replace(base.data, repo_id=sources[0].repo_id, sources=sources, assets=assets)
    training = raw["training"]
    wandb = raw.get("wandb") or {}
    steps = int(training.get("steps", base.num_train_steps))
    lr = base.lr_schedule
    if isinstance(lr, optimizer.CosineDecaySchedule):
        lr = replace(lr, decay_steps=steps)
    return replace(
        base,
        data=data,
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
        save_interval=int(training.get("save_interval", base.save_interval)),
        keep_period=training.get("keep_period", base.keep_period),
        seed=int(training.get("seed", base.seed)),
        fsdp_devices=int(training.get("fsdp_devices", base.fsdp_devices)),
        overwrite=bool(training.get("overwrite", False)),
        resume=bool(training.get("resume", False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-output", action="store_true")
    args = parser.parse_args()
    raw = load_config(args.config.expanduser().resolve())
    output = Path(str(raw["training"]["output"])).expanduser().resolve()
    if args.print_output:
        print(output)
    if args.check:
        return
    from scripts.train import main as train_main

    train_main(build_train_config(raw))


if __name__ == "__main__":
    main()

