#!/usr/bin/env python
"""Compute normalization statistics for a pi0.5 training config.

openpi's `scripts/compute_norm_stats.py` (commit 15a9616), adapted only where it must be:
imports point at `lerobot.policies.pi05_jax`; the RLDS/DROID branch is dropped (not vendored);
and the output directory is keyed by `data_config.asset_id` rather than `repo_id`.

That last point is a real upstream bug, not a preference: `DataConfigFactory.create_base_config`
loads stats from `assets_dir / asset_id`, so writing them to `assets_dir / repo_id` only works
when the two happen to be equal. This repo's pick_tube config sets `asset_id="pick_tube"` with
`repo_id="KaiyueChen/pick_tube"`, so writing by `repo_id` would produce a file the trainer then
reports as missing.

Run this once before training on a new dataset:

    python tools/compute_pi05_norm_stats.py --config-name=pi05_pick_tube
"""

import sys
from pathlib import Path

import numpy as np
import tqdm
import tyro

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lerobot.policies.pi05_jax import (  # noqa: E402  (must follow the sys.path insert above)
    model as _model,
    normalize,
    transforms,
)
from lerobot.policies.pi05_jax.training import (  # noqa: E402
    config as _config,
    data_loader as _data_loader,
)


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    data_loader, num_batches = create_torch_dataloader(
        data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # See the module docstring: the trainer reads these back from `assets_dirs / asset_id`.
    output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
