#!/usr/bin/env python
"""Fine-tune JAX SmolVLA from a prepared NPZ training dataset."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax

from lerobot.policies.smolvla_jax import JaxSmolVLA, JaxSmolVLAConfig
from lerobot.policies.smolvla_jax.checkpoint import load_params
from lerobot.policies.smolvla_jax.data import load_training_npz, numpy_batches
from lerobot.policies.smolvla_jax.training import JaxSmolVLATrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path, help="Prepared training .npz")
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
    batches = numpy_batches(load_training_npz(args.dataset), args.batch_size, seed=args.seed)
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
            print(f"saved checkpoint: {path}")


if __name__ == "__main__":
    main()
