#!/usr/bin/env python
"""Convert raw observation/action NPZ arrays into JAX SmolVLA training batches.

Input arrays use a leading sample dimension and must include ``task``,
``observation.state``, ``actions``, and camera keys. The output is accepted by
``tools/train_smolvla_jax.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lerobot.policies.smolvla_jax import JaxSmolVLAConfig
from lerobot.policies.smolvla_jax.preprocessing import JaxSmolVLAPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-dtype", choices=("float16", "float32"), default="float16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = JaxSmolVLAConfig.from_pretrained(args.checkpoint)
    processor = JaxSmolVLAPreprocessor(args.checkpoint, config)
    with np.load(args.input, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    required = {"task", "observation.state", "actions"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"raw NPZ is missing keys: {sorted(missing)}")
    sample_count = raw["observation.state"].shape[0]
    if raw["actions"].shape[:2] != (sample_count, config.chunk_size):
        raise ValueError(f"actions must have shape [N,{config.chunk_size},A], got {raw['actions'].shape}")
    observation_keys = [key for key in raw if key.startswith("observation.")]
    prepared = {key: [] for key in ("images", "image_masks", "language_tokens", "language_masks", "state")}
    for index in range(sample_count):
        observation = {key: raw[key][index] for key in observation_keys}
        task = raw["task"][index]
        if isinstance(task, bytes):
            task = task.decode("utf-8")
        batch = processor.prepare(observation, str(task))
        for key in prepared:
            prepared[key].append(np.asarray(batch[key][0]))
    output = {key: np.stack(value) for key, value in prepared.items()}
    output["images"] = output["images"].astype(args.image_dtype)
    output["actions"] = np.asarray(processor.normalize_actions(raw["actions"]), dtype=np.float32)
    if "action_is_pad" in raw:
        output["action_is_pad"] = raw["action_is_pad"].astype(np.bool_)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **output)
    print(f"prepared {sample_count} samples: {args.output}")
    print({key: value.shape for key, value in output.items()})


if __name__ == "__main__":
    main()
