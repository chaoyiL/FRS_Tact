#!/usr/bin/env python
"""Fine-tune JAX SmolVLA directly from a LeRobotDataset."""

from __future__ import annotations

import argparse
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import jax
import yaml

from lerobot.policies.smolvla_jax import JaxSmolVLA, JaxSmolVLAConfig
from lerobot.policies.smolvla_jax.checkpoint import load_params, resolve_checkpoint
from lerobot.policies.smolvla_jax.data import LeRobotJaxDataLoader, parse_dataset_sources
from lerobot.policies.smolvla_jax.lora import resolve_module_modes
from lerobot.policies.smolvla_jax.training import JaxSmolVLATrainer

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train_smolvla_jax.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--checkpoint", help="Override YAML: local path or Hugging Face repo id")
    parser.add_argument("--revision")
    parser.add_argument("--allow-download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--video-backend")
    parser.add_argument("--allow-tokenizer-download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-freq", type=int)
    parser.add_argument("--save-freq", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--data-parallel", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def merge_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = dict(cfg)
    cli = {
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "allow_download": args.allow_download,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "video_backend": args.video_backend,
        "allow_tokenizer_download": args.allow_tokenizer_download,
        "output": args.output,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "log_freq": args.log_freq,
        "save_freq": args.save_freq,
        "resume": args.resume,
        "data_parallel": args.data_parallel,
    }
    for key, value in cli.items():
        if value is not None:
            merged[key] = value
    return merged


def require(cfg: dict[str, Any], key: str) -> Any:
    if key not in cfg or cfg[key] in (None, ""):
        raise ValueError(f"missing required config field: {key}")
    return cfg[key]


def apply_model_overrides(config: JaxSmolVLAConfig, overrides: dict[str, Any] | None) -> JaxSmolVLAConfig:
    if not overrides:
        return config
    if not isinstance(overrides, dict):
        raise ValueError("model overrides must be a mapping")
    known = {field.name for field in fields(JaxSmolVLAConfig)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"unknown model override fields: {unknown}")
    cleaned: dict[str, Any] = {}
    for key, value in overrides.items():
        if key == "image_keys" and value is not None:
            cleaned[key] = tuple(value)
        else:
            cleaned[key] = value
    return replace(config, **cleaned)


def init_wandb(cfg: dict[str, Any], *, config_path: Path, checkpoint: Path, model: JaxSmolVLAConfig):
    wandb_cfg = cfg.get("wandb") or {}
    if not bool(wandb_cfg.get("enabled", False)):
        return None

    import wandb

    mode = str(wandb_cfg.get("mode", "online"))
    run = wandb.init(
        project=wandb_cfg.get("project", "smolvla-jax"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name"),
        group=wandb_cfg.get("group"),
        tags=list(wandb_cfg.get("tags") or []),
        notes=wandb_cfg.get("notes"),
        dir=str(Path(require(cfg, "output"))),
        mode=mode,
        config={
            "config_path": str(config_path.resolve()),
            "checkpoint": str(checkpoint),
            "datasets": cfg.get("datasets"),
            "batch_size": cfg.get("batch_size"),
            "steps": cfg.get("steps"),
            "seed": cfg.get("seed"),
            "data_parallel": cfg.get("data_parallel"),
            "model": model.to_dict(),
            "wandb": {k: v for k, v in wandb_cfg.items() if k != "api_key"},
        },
    )
    print(f"wandb={run.url if run is not None else mode}")
    return run


def main() -> None:
    args = parse_args()
    cfg = merge_cli_overrides(load_yaml_config(args.config), args)

    checkpoint = resolve_checkpoint(
        require(cfg, "checkpoint"),
        revision=cfg.get("revision"),
        local_files_only=not bool(cfg.get("allow_download", False)),
    )
    print(f"config={args.config.resolve()}")
    print(f"checkpoint={checkpoint}")

    config = apply_model_overrides(
        JaxSmolVLAConfig.from_pretrained(checkpoint),
        cfg.get("model"),
    )
    print(
        f"model overrides: action_dim={config.action_dim} state_dim={config.state_dim} "
        f"image_keys={list(config.image_keys)}"
    )

    model = JaxSmolVLA(config)
    trainer = JaxSmolVLATrainer(
        model,
        load_params(checkpoint),
        seed=int(cfg.get("seed", 0)),
        total_steps=int(require(cfg, "steps")),
    )
    trainable_count = sum(int(value.size) for value in trainer.state.params.values())
    frozen_count = sum(int(value.size) for value in trainer.frozen_params.values())
    print(f"module_modes={resolve_module_modes(config)}")
    print(
        f"parameters: trainable={trainable_count:,} frozen={frozen_count:,} "
        f"trainable_ratio={trainable_count / max(trainable_count + frozen_count, 1):.4%}"
    )
    resume = cfg.get("resume")
    if resume:
        trainer.restore(Path(resume))
    if bool(cfg.get("data_parallel", False)):
        trainer.enable_data_parallel()

    allow_download = bool(cfg.get("allow_download", False))
    allow_tokenizer_download = bool(cfg.get("allow_tokenizer_download", False))
    sources = parse_dataset_sources(cfg)
    data = LeRobotJaxDataLoader(
        checkpoint,
        config,
        sources=sources,
        batch_size=int(cfg.get("batch_size", 8)),
        num_workers=int(cfg.get("num_workers", 4)),
        prefetch_factor=int(cfg.get("prefetch_factor", 2)),
        video_backend=cfg.get("video_backend"),
        seed=int(cfg.get("seed", 0)),
        local_files_only=not (allow_tokenizer_download or allow_download),
    )
    batches = data.batches()
    for summary in data.dataset_summaries:
        print(
            f"dataset={summary['repo_id']} frames={summary['frames']} "
            f"episodes={summary['episodes']} fps={summary['fps']} "
            f"action_key={summary['action_key']!r} weight={summary['weight']}"
        )
    print(f"combined_frames={len(data.dataset)}")

    output = Path(require(cfg, "output"))
    output.mkdir(parents=True, exist_ok=True)
    steps = int(require(cfg, "steps"))
    log_freq = int(cfg.get("log_freq", 10))
    save_freq = int(cfg.get("save_freq", 1000))
    wandb_cfg = cfg.get("wandb") or {}
    wandb_run = init_wandb(cfg, config_path=args.config, checkpoint=checkpoint, model=config)
    log_checkpoints = bool(wandb_cfg.get("log_checkpoints", False))

    start = time.perf_counter()
    try:
        while int(trainer.state.step) < steps:
            metrics = trainer.step(next(batches))
            step = int(trainer.state.step)
            if step == 1 or step % log_freq == 0:
                metrics = jax.device_get(metrics)
                elapsed = time.perf_counter() - start
                loss = float(metrics["loss"])
                grad_norm = float(metrics["grad_norm"])
                lr = float(metrics["learning_rate"])
                print(
                    f"step={step} loss={loss:.6f} "
                    f"grad_norm={grad_norm:.4f} "
                    f"lr={lr:.3e} elapsed={elapsed:.1f}s"
                )
                if wandb_run is not None:
                    import wandb

                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/grad_norm": grad_norm,
                            "train/learning_rate": lr,
                            "train/elapsed_s": elapsed,
                        },
                        step=step,
                    )
            if step % save_freq == 0 or step == steps:
                path = output / f"checkpoint-{step:08d}"
                trainer.save(path, source_dir=checkpoint)
                data.preprocessor.save_normalization_assets(path)
                print(f"saved checkpoint: {path}")
                if wandb_run is not None:
                    import wandb

                    wandb.log({"train/checkpoint_step": step}, step=step)
                    if log_checkpoints:
                        wandb.save(str(path / "*"), base_path=str(output))
    finally:
        if wandb_run is not None:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
