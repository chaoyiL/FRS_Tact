#!/usr/bin/env python
"""Run the JAX SmolVLA policy on an observation stored in an NPZ file.

The NPZ must contain ``observation.state`` and the camera keys expected by the
checkpoint. Images may be HWC uint8 or CHW float arrays.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from lerobot.policies.smolvla_jax import JaxSmolVLAPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint or Hugging Face repo id")
    parser.add_argument("--revision")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--observation", required=True, type=Path, help="Input .npz observation")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=Path, help="Optional .npy noise with shape [B,50,32]")
    parser.add_argument("--rename-map", help="JSON object overriding the checkpoint rename map")
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--num-steps", type=int, help="Override Euler denoising steps")
    parser.add_argument("--previous-chunk", type=Path, help="RTC previous action chunk (.npy)")
    parser.add_argument("--inference-delay", type=int)
    parser.add_argument("--execution-horizon", type=int)
    parser.add_argument("--normalized", action="store_true", help="Do not unnormalize output actions")
    parser.add_argument("--output", type=Path, help="Optional .npy output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rename_map = json.loads(args.rename_map) if args.rename_map else None
    with np.load(args.observation, allow_pickle=False) as archive:
        observation = {key: archive[key] for key in archive.files}
    noise = np.load(args.noise) if args.noise else None
    previous_chunk = np.load(args.previous_chunk) if args.previous_chunk else None
    policy = JaxSmolVLAPolicy.from_pretrained(
        args.checkpoint,
        rename_map=rename_map,
        revision=args.revision,
        local_files_only=not args.allow_download,
    )
    start = time.perf_counter()
    actions = policy.predict_action_chunk(
        observation,
        args.task,
        seed=args.seed,
        noise=noise,
        jit=not args.no_jit,
        normalized=args.normalized,
        num_steps=args.num_steps,
        previous_chunk=previous_chunk,
        inference_delay=args.inference_delay,
        execution_horizon=args.execution_horizon,
    )
    jax.block_until_ready(actions)
    elapsed = time.perf_counter() - start
    actions_numpy = np.asarray(actions)
    print(f"platform     : {jax.default_backend()}")
    print(f"action shape : {actions_numpy.shape}")
    print(f"elapsed      : {elapsed:.3f}s")
    print(f"first action : {actions_numpy[0, 0]}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, actions_numpy)
        print(f"saved        : {args.output}")


if __name__ == "__main__":
    main()
