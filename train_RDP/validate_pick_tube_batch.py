#!/usr/bin/env python3
"""Load one pick-tube RDP batch and print the exact AT/LDP tensor shapes."""

from __future__ import annotations

import argparse

import zarr
from torch.utils.data import DataLoader

from reactive_diffusion_policy.dataset.real_image_tactile_dataset import RealImageTactileDataset


def shape_meta(include_rgb: bool, tactile_embedding_dim: int = 30) -> dict:
    obs = {
        "observation_state": {"shape": [20], "type": "low_dim"},
        "tactile_embedding": {"shape": [tactile_embedding_dim], "type": "low_dim"},
    }
    if include_rgb:
        obs = {
            "camera1": {"shape": [3, 224, 224], "type": "rgb"},
            "camera2": {"shape": [3, 224, 224], "type": "rgb"},
            **obs,
        }
    return {
        "obs": obs,
        "extended_obs": {
            "tactile_embedding": {"shape": [tactile_embedding_dim], "type": "low_dim"}
        },
        "action": {"shape": [20]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--mode", choices=("at", "ldp"), default="ldp")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    replay_buffer = zarr.open_group(
        f"{args.dataset_path}/replay_buffer.zarr", mode="r"
    )
    tactile_embedding_dim = int(
        replay_buffer["data"]["tactile_embedding"].shape[1]
    )

    dataset = RealImageTactileDataset(
        shape_meta=shape_meta(
            include_rgb=args.mode == "ldp",
            tactile_embedding_dim=tactile_embedding_dim,
        ),
        dataset_path=args.dataset_path,
        horizon=32,
        pad_before=3,
        pad_after=28,
        n_obs_steps=4,
        obs_temporal_downsample_ratio=2,
        seed=42,
        val_ratio=0.0,
        use_episode_repeats=False,
        delta_action=False,
        relative_action=False,
        load_to_memory=False,
        bimanual_contiguous_action=True,
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)))
    for group in ("obs", "extended_obs"):
        for key, value in batch[group].items():
            print(f"{group}.{key}: {tuple(value.shape)} {value.dtype}")
    print(f"action: {tuple(batch['action'].shape)} {batch['action'].dtype}")


if __name__ == "__main__":
    main()
