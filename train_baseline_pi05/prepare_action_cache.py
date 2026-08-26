"""Produce a direct, frozen Pi0.5 action cache using only forward sampling."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .action_cache import ActionCacheWriter, SampleRecord, build_records
from .source_model import fixed_noise, load_pi05_source_model, sample_coarse_actions, validate_pi05_model


def _field(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for name in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(name, default)
        else:
            current = getattr(current, name, default)
        if current is default:
            return default
    return current


def _stack(values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, Mapping):
        return {key: _stack([value[key] for value in values]) for key in first}
    return np.stack([np.asarray(value) for value in values], axis=0)


def _window(dataset: Any, record: SampleRecord, metadata: Any, action_key: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    end = int(metadata.episodes[record.episode_index]["dataset_to_index"])
    rows = [np.asarray(dataset[min(record.dataset_index + step, end - 1)][action_key]) for step in range(horizon)]
    valid = np.arange(horizon) < (end - record.dataset_index)
    return np.stack(rows).astype(np.float32), valid.astype(np.bool_)


def _default_dependencies(config: Any) -> dict[str, Any]:
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.pi05_jax import load_norm_stats
    from .policy_inputs import Pi05SampleProcessor

    dataset_info = _field(config, "dataset")
    source_info = _field(config, "source")
    metadata = LeRobotDatasetMetadata(dataset_info.repo_id, root=dataset_info.root, revision=getattr(dataset_info, "revision", None))
    dataset = LeRobotDataset(dataset_info.repo_id, root=dataset_info.root, revision=getattr(dataset_info, "revision", None))
    processor = Pi05SampleProcessor(
        dataset_repo_id=dataset_info.repo_id, dataset_root=dataset_info.root,
        dataset_revision=getattr(dataset_info, "revision", None), action_key=dataset_info.action_key,
        camera_map=getattr(dataset_info, "camera_map", {}),
        state_stats=load_norm_stats(source_info.norm_stats_dir, source_info.norm_stats_asset_id)["state"],
        action_stats=load_norm_stats(source_info.norm_stats_dir, source_info.norm_stats_asset_id)["actions"],
        action_horizon=source_info.action_horizon,
    )
    model, width = load_pi05_source_model(source_info.checkpoint, config=processor.config)
    return {"metadata": metadata, "dataset": dataset, "processor": processor, "model": model, "params": None, "source_width": width}


def prepare_action_cache(config: Any, dependencies: Mapping[str, Any] | None = None) -> Path:
    """Write `[N, 50, 20]` normalized coarse/expert cache arrays and finalize them."""
    deps = dict(_default_dependencies(config) if dependencies is None else dependencies)
    metadata, dataset, processor, model = (deps[name] for name in ("metadata", "dataset", "processor", "model"))
    source = _field(config, "source", config)
    decoder = _field(config, "decoder", config)
    output = Path(_field(config, "cache.action_root", _field(config, "action_root")))
    horizon = int(getattr(source, "action_horizon", 50))
    action_dim = int(getattr(decoder, "action_dim", 20))
    if horizon != 50 or action_dim != 20:
        raise ValueError("direct decoder cache contract is [N, 50, 20]")
    records = tuple(deps.get("records") or build_records(
        metadata, split_seed=int(_field(config, "dataset.split_seed", 0)),
        fractions=(float(_field(config, "dataset.train_fraction", 0.8)), float(_field(config, "dataset.validation_fraction", 0.1)), float(_field(config, "dataset.test_fraction", 0.1))),
        frame_stride=int(_field(config, "dataset.frame_stride", 1)),
    ))
    source_width = int(deps.get("source_width", validate_pi05_model(model, action_horizon=horizon)))
    if source_width < action_dim:
        raise ValueError("source model is narrower than the decoder action width")
    action_key = str(_field(config, "dataset.action_key", "actions"))
    manifest = {
        "dataset_identity": {"repo_id": str(_field(config, "dataset.repo_id", "injected")), "revision": _field(config, "dataset.revision")},
        "split": {"seed": int(_field(config, "dataset.split_seed", 0))},
        "source_checkpoint": str(_field(config, "source.checkpoint", "injected")), "norm_stats": {},
        "sample_steps": int(getattr(source, "sample_steps", 10)), "noise_seed": int(getattr(source, "seed", 0)),
        "source_model_action_width": source_width, "decoder_action_width": action_dim, "action_space": "normalized_pi05",
    }
    batch_size = int(deps.get("batch_size", 1))
    with ActionCacheWriter.create(output, sample_count=len(records), horizon=horizon, action_dim=action_dim, manifest=manifest) as writer:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            observations, experts, valid = [], [], []
            for record in batch:
                sample = dict(dataset[record.dataset_index])
                action_window, row_valid = _window(dataset, record, metadata, action_key, horizon)
                sample[action_key] = action_window
                observation, expert, _ = processor.prepare_sample(sample)
                observations.append(observation)
                experts.append(np.asarray(expert, dtype=np.float32)[..., :action_dim])
                valid.append(row_valid)
            observation_batch = _stack(observations)
            noise = fixed_noise(len(batch), seed=manifest["noise_seed"], horizon=horizon, action_dim=source_width)
            coarse = sample_coarse_actions(model, deps.get("params"), observation_batch, noise, manifest["sample_steps"])[..., :action_dim]
            writer.write_batch(start, coarse=coarse.astype(np.float32), expert=np.stack(experts).astype(np.float32), valid=np.stack(valid), records=batch)
        writer.finalize()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    from .config import load_config
    prepare_action_cache(load_config(args.config))


if __name__ == "__main__":
    main()
