"""Produce a direct, frozen Pi0.5 action cache using only forward sampling."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .action_cache import ActionCacheWriter, SampleRecord, build_records
from .config import TACTILE_KEYS
from .source_model import fixed_noise, load_pi05_source_model, make_frozen_sampler, sample_coarse_actions, validate_pi05_model


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


def _stack_mapping(values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, Mapping):
        return {key: _stack_mapping([value[key] for value in values]) for key in first}
    return np.stack([np.asarray(value) for value in values], axis=0)


def _stack(values: Sequence[Any]) -> Any:
    if isinstance(values[0], Mapping):
        return _stack_mapping(values)
    from .policy_inputs import stack_observations

    return stack_observations(list(values))


def _window(dataset: Any, record: SampleRecord, metadata: Any, action_key: str, horizon: int, *, action_rows: Any = None) -> tuple[np.ndarray, np.ndarray]:
    end = int(metadata.episodes[record.episode_index]["dataset_to_index"])
    indices = np.minimum(record.dataset_index + np.arange(horizon), end - 1).tolist()
    raw_dataset = getattr(dataset, "hf_dataset", None)
    if raw_dataset is not None:
        # LeRobot __getitem__ decodes every selected video. Action windows only
        # need the numeric column, including when image columns use HF decoders.
        relative_indices = getattr(dataset, "absolute_to_relative_idx", None)
        if relative_indices is not None:
            indices = [relative_indices[index] for index in indices]
        if action_rows is None:
            action_rows = raw_dataset.select_columns([action_key])
        rows = action_rows[indices][action_key]
    else:
        # Keep small in-memory/custom dataset adapters usable without Hugging Face.
        rows = [dataset[index][action_key] for index in indices]
    rows = [np.asarray(row) for row in rows]
    actions = np.stack(rows).astype(np.float32)
    within_episode = np.arange(horizon) < (end - record.dataset_index)
    non_terminal = np.any(actions != 0, axis=-1)
    return actions, (within_episode & non_terminal).astype(np.bool_)


def _camera_visual_keys(dataset_info: Any) -> list[str]:
    """Resolve configured post-rename camera names to raw dataset columns."""
    rename_map = dict(dataset_info.rename_map)
    selected = []
    for target in dataset_info.camera_map.values():
        sources = [source for source, destination in rename_map.items() if destination == target]
        selected.extend(sources or [target])
    return list(dict.fromkeys(selected))


def _prefetched_batches(starts: Iterable[int], prepare: Callable[[int], Any], *, enabled: bool) -> Iterator[tuple[int, Any, float]]:
    """Yield in order with at most one prepared batch ahead of the consumer."""
    if not enabled:
        for start in starts:
            waited = time.perf_counter()
            result = prepare(start)
            yield start, result, time.perf_counter() - waited
        return
    iterator = iter(starts)
    start = next(iterator, None)
    if start is None:
        return
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="action-prefetch")
    try:
        future = executor.submit(prepare, start)
        while start is not None:
            waited = time.perf_counter()
            result = future.result()
            wait_seconds = time.perf_counter() - waited
            next_start = next(iterator, None)
            if next_start is not None:
                future = executor.submit(prepare, next_start)
            yield start, result, wait_seconds
            start = next_start
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _default_dependencies(config: Any) -> dict[str, Any]:
    import torch

    # This dedicated JAX cache process only uses Torch for small dataset tensors.
    # Large intra-op pools make PIL conversion slower and delay feeding the GPU.
    torch.set_num_threads(1)
    from .runtime_path import activate_vendored_lerobot
    activate_vendored_lerobot()
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.pi05_jax import load_norm_stats
    from .policy_inputs import Pi05SampleProcessor

    dataset_info = _field(config, "dataset")
    source_info = _field(config, "source")
    metadata = LeRobotDatasetMetadata(dataset_info.repo_id, root=dataset_info.root, revision=getattr(dataset_info, "revision", None))
    dataset = LeRobotDataset(
        dataset_info.repo_id, root=dataset_info.root,
        revision=getattr(dataset_info, "revision", None), visual_keys=_camera_visual_keys(dataset_info),
    )
    processor = Pi05SampleProcessor(
        dataset_repo_id=dataset_info.repo_id, dataset_root=dataset_info.root,
        dataset_revision=getattr(dataset_info, "revision", None), action_key=dataset_info.action_key,
        rename_map=dataset_info.rename_map, camera_map=dataset_info.camera_map,
        state_stats=load_norm_stats(source_info.norm_stats_dir, source_info.norm_stats_asset_id)["state"],
        action_stats=load_norm_stats(source_info.norm_stats_dir, source_info.norm_stats_asset_id)["actions"],
        use_quantile_norm=source_info.use_quantile_norm, action_dim=source_info.model_action_dim,
        action_horizon=source_info.action_horizon, paligemma_variant=source_info.paligemma_variant,
        action_expert_variant=source_info.action_expert_variant,
    )
    print(f"[Action cache] Loading Pi0.5 checkpoint: {source_info.checkpoint}", flush=True)
    model, width = load_pi05_source_model(source_info.checkpoint, config=processor.config)
    print("[Action cache] Pi0.5 checkpoint loaded.", flush=True)
    import jax
    return {"metadata": metadata, "dataset": dataset, "processor": processor, "model": model, "params": jax.random.key(source_info.seed), "source_width": width}


def prepare_action_cache(config: Any, dependencies: Mapping[str, Any] | None = None, *, max_samples: int | None = None) -> Path:
    """Write normalized `[N, 50, 10 or 20]` coarse/expert cache arrays."""
    print("[Action cache] Loading dataset and preparing source inputs...", flush=True)
    deps = dict(_default_dependencies(config) if dependencies is None else dependencies)
    metadata, dataset, processor, model = (deps[name] for name in ("metadata", "dataset", "processor", "model"))
    source = _field(config, "source", config)
    decoder = _field(config, "decoder", config)
    output = Path(_field(config, "cache.action_root", _field(config, "action_root")))
    horizon = int(getattr(source, "action_horizon", 50))
    action_dim = int(getattr(decoder, "action_dim", 20))
    if horizon != 50 or action_dim not in (10, 20):
        raise ValueError("direct decoder cache contract is [N, 50, 10 or 20]")
    records = tuple(deps.get("records") or build_records(
        metadata, split_seed=int(_field(config, "dataset.split_seed", 0)),
        fractions=(float(_field(config, "dataset.train_fraction", 0.8)), float(_field(config, "dataset.validation_fraction", 0.1)), float(_field(config, "dataset.test_fraction", 0.1))),
        frame_stride=int(_field(config, "dataset.frame_stride", 1)),
    ))
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        records = records[:max_samples]
    source_width = int(deps.get("source_width", validate_pi05_model(model, action_horizon=horizon)))
    model_action_dim = int(getattr(source, "model_action_dim", source_width))
    if source_width != model_action_dim:
        raise ValueError("loaded source model width does not match source.model_action_dim")
    if source_width < action_dim:
        raise ValueError("source model is narrower than the decoder action width")
    action_key = str(_field(config, "dataset.action_key", "actions"))
    fractions = [float(_field(config, "dataset.train_fraction", 0.8)), float(_field(config, "dataset.validation_fraction", 0.1)), float(_field(config, "dataset.test_fraction", 0.1))]
    frame_stride = int(_field(config, "dataset.frame_stride", 1))
    manifest = {
        "dataset_identity": {
            "repo_id": str(_field(config, "dataset.repo_id", "injected")),
            "root": str(Path(_field(config, "dataset.root", "injected")).expanduser().resolve()),
            "revision": _field(config, "dataset.revision"),
        },
        "split": {"seed": int(_field(config, "dataset.split_seed", 0)), "fractions": fractions, "frame_stride": frame_stride},
        "source_checkpoint": str(_field(config, "source.checkpoint", "injected")),
        "source_variant": {"paligemma_variant": _field(config, "source.paligemma_variant"), "action_expert_variant": _field(config, "source.action_expert_variant")},
        "norm_stats": {"dir": str(_field(config, "source.norm_stats_dir")) if _field(config, "source.norm_stats_dir") is not None else None, "asset_id": _field(config, "source.norm_stats_asset_id"), "use_quantile_norm": bool(_field(config, "source.use_quantile_norm", True))},
        "sample_steps": int(getattr(source, "sample_steps", 10)), "noise_seed": int(getattr(source, "seed", 0)),
        "source_model_action_width": source_width, "decoder_action_width": action_dim, "action_space": "normalized_pi05",
        "camera_map": dict(_field(config, "dataset.camera_map", {})),
        "rename_map": dict(_field(config, "dataset.rename_map", {})),
        "action_key": action_key,
    }
    batch_size = int(deps.get("batch_size", _field(config, "cache.action_batch_size", 64)))
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        writer = ActionCacheWriter.create(output, sample_count=len(records), horizon=horizon, action_dim=action_dim, manifest=manifest)
    else:
        status = __import__("json").load(manifest_path.open(encoding="utf-8")).get("status")
        if status == "incomplete":
            writer = ActionCacheWriter.resume(output, manifest)
        elif status == "complete":
            from .action_cache import ActionCache
            cache = ActionCache.open(output)
            if cache.manifest.get("immutable_manifest") != {
                "cache_version": 1, "sample_count": len(records), "horizon": horizon, "action_dim": action_dim,
                **manifest,
            }:
                raise ValueError("complete action cache does not match requested producer contract")
            print(f"[Action cache] Reusing complete cache: {len(records):,} samples at {output}", flush=True)
            return output
        else:
            raise ValueError("action cache manifest has an invalid status")
    completed = int(writer.manifest["completed_samples"])
    print(
        f"[Action cache] {completed:,}/{len(records):,} samples complete, batch size {batch_size}; "
        "the first batch includes JAX compilation.",
        flush=True,
    )
    prefetch = bool(_field(config, "cache.action_prefetch", False))
    with writer, tqdm(total=len(records), initial=completed, desc="Pi0.5 action cache", unit="sample", mininterval=1.0, dynamic_ncols=True, disable=False) as progress:
        sampler = make_frozen_sampler(model)
        # Selecting HF columns copies Arrow metadata. Reuse this numeric-only
        # view across windows instead of rebuilding it for every sample.
        raw_dataset = getattr(dataset, "hf_dataset", None)
        action_rows = raw_dataset.select_columns([action_key]) if raw_dataset is not None else None
        cpu_device = None
        if prefetch:
            import jax
            cpu_device = jax.devices("cpu")[0]
            print("[Action cache] CPU preprocessing enabled; prefetching one batch ahead.", flush=True)

        def prepare(start):
            import jax

            batch_started = time.perf_counter()
            batch = records[start : start + batch_size]
            observations, experts, valid = [], [], []
            # Keep resize and small tensor operations off the inference GPU.
            # Dataset/processor access stays on this single producer thread.
            with jax.default_device(cpu_device) if cpu_device is not None else nullcontext():
                for record in batch:
                    sample = {
                        key: value for key, value in dict(dataset[record.dataset_index]).items()
                        if key not in TACTILE_KEYS
                    }
                    action_window, row_valid = _window(dataset, record, metadata, action_key, horizon, action_rows=action_rows)
                    sample[action_key] = action_window
                    observation, expert, _ = processor.prepare_sample(sample)
                    observations.append(jax.tree.map(np.asarray, observation) if prefetch else observation)
                    experts.append(np.asarray(expert, dtype=np.float32)[..., :action_dim])
                    valid.append(row_valid)
                return observations, np.stack(experts), np.stack(valid), time.perf_counter() - batch_started

        starts = range(completed, len(records), batch_size)
        with closing(_prefetched_batches(starts, prepare, enabled=prefetch)) as batches:
            for start, (observations, experts, valid, data_seconds), wait_seconds in batches:
                batch = records[start : start + batch_size]
                transfer_started = time.perf_counter()
                observation_batch = _stack(observations)
                noise = fixed_noise(len(batch), seed=manifest["noise_seed"], horizon=horizon, action_dim=model_action_dim)
                data_ready = time.perf_counter()
                coarse = sample_coarse_actions(model, deps.get("params"), observation_batch, noise, manifest["sample_steps"], sampler=sampler)[..., :action_dim]
                inference_done = time.perf_counter()
                writer.write_batch(start, coarse=coarse.astype(np.float32), expert=experts.astype(np.float32), valid=valid, records=batch)
                progress.set_postfix(
                    data=f"{data_seconds:.2f}s",
                    wait=f"{wait_seconds:.2f}s",
                    pack=f"{data_ready - transfer_started:.2f}s",
                    infer=f"{inference_done - data_ready:.2f}s",
                    write=f"{time.perf_counter() - inference_done:.2f}s",
                    refresh=False,
                )
                progress.update(len(batch))
        progress.close()
        print("[Action cache] Finalizing and verifying cache...", flush=True)
        writer.finalize()
    print(f"[Action cache] Complete: {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    from .config import load_config
    prepare_action_cache(load_config(args.config), max_samples=args.max_samples)


if __name__ == "__main__":
    main()
