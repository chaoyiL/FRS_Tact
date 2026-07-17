#!/usr/bin/env python
"""Fine-tune JAX SmolVLA directly from a LeRobotDataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax

from lerobot.policies.smolvla_jax import JaxSmolVLA, JaxSmolVLAConfig
from lerobot.policies.smolvla_jax.checkpoint import load_params
from lerobot.policies.smolvla_jax.data import LeRobotJaxDataLoader
from lerobot.policies.smolvla_jax.training import JaxSmolVLATrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--action-key", help="Defaults to auto-detecting action/actions")
    parser.add_argument("--rename-map", help="JSON object overriding checkpoint observation renames")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--video-backend")
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=1_000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--data-parallel", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rename_map = json.loads(args.rename_map) if args.rename_map else None
    if rename_map is not None and not isinstance(rename_map, dict):
        raise ValueError("--rename-map must be a JSON object")
    config = JaxSmolVLAConfig.from_pretrained(args.checkpoint)
    model = JaxSmolVLA(config)
    trainer = JaxSmolVLATrainer(
        model,
        load_params(args.checkpoint),
        seed=args.seed,
        total_steps=args.steps,
    )
    if args.resume:
        trainer.restore(args.resume)
    if args.data_parallel:
        trainer.enable_data_parallel()
    data = LeRobotJaxDataLoader(
        args.checkpoint,
        config,
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        revision=args.dataset_revision,
        episodes=args.episodes,
        action_key=args.action_key,
        rename_map=rename_map,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        video_backend=args.video_backend,
        seed=args.seed,
        local_files_only=not args.allow_tokenizer_download,
    )
    batches = data.batches()
    print(
        f"dataset={args.dataset_repo_id} frames={len(data.dataset)} "
        f"episodes={data.dataset.num_episodes} fps={data.dataset.fps} "
        f"action_key={data.action_key!r}"
    )
    start = time.perf_counter()
    while int(trainer.state.step) < args.steps:
        metrics = trainer.step(next(batches))
        step = int(trainer.state.step)
        if step == 1 or step % args.log_freq == 0:
            metrics = jax.device_get(metrics)
            elapsed = time.perf_counter() - start
            print(
                f"step={step} loss={float(metrics['loss']):.6f} "
                f"grad_norm={float(metrics['grad_norm']):.4f} "
                f"lr={float(metrics['learning_rate']):.3e} elapsed={elapsed:.1f}s"
            )
        if step % args.save_freq == 0 or step == args.steps:
            path = args.output / f"checkpoint-{step:08d}"
            trainer.save(path, source_dir=args.checkpoint)
            data.preprocessor.save_normalization_assets(path)
            print(f"saved checkpoint: {path}")


if __name__ == "__main__":
    main()
