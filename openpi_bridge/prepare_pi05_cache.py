#!/usr/bin/env python
"""Precompute a pi0.5 action_cache for FRS, in the same on-disk format SmolVLA's
tools/prepare_frs_caches.py produces.

This runs in the independent `openpi_bridge` venv (see README.md / ../pi05_frs_plan.md for why),
so it must not import anything from the main repo's `lerobot` package. The only thing it borrows
from the main repo is ../utils/cache.py, which is pure numpy/stdlib and defines the on-disk cache
format that tools/train_frs.py already knows how to read.

STATUS: scaffold only. The three pi0.5-specific pieces are marked TODO / NotImplementedError below:
  1. loading a pi0.5 policy for our pick_tube robot setup (needs a custom openpi TrainConfig/DataConfig)
  2. mapping one LeRobot dataset sample -> the observation dict pi0.5 expects
  3. exposing a per-step velocity_fn so the euler/fireflow reverse solver can run t:0->1
See ../pi05_frs_plan.md for the concrete API references and why each of these is non-trivial.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

BRIDGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BRIDGE_ROOT.parent
# Only for utils.cache, which has no dependency on this repo's `lerobot` package (see module docstring).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cache import (  # noqa: E402
    CACHE_VERSION,
    MANIFEST_NAME,
    SampleRecord,
    atomic_write_json,
    build_records,
    create_cache_arrays,
    flush_arrays,
    load_manifest,
    open_cache_arrays,
    records_digest,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "train_frs_pick_tube_pi05.yaml"


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return value


def source_cache_dir(cache_root: str | Path, repo_id: str) -> Path:
    parts = [part for part in str(repo_id).split("/") if part not in ("", ".", "..")]
    if not parts:
        raise ValueError(f"invalid repo id: {repo_id!r}")
    return Path(cache_root).expanduser().joinpath(*parts)


@dataclasses.dataclass
class Pi05CacheModel:
    """Everything prepare_pi05_cache() needs from the pi0.5 side.

    Fill this in once the openpi TrainConfig / observation mapping / exposed velocity_fn exist
    (see module docstring). Keeping it as one small object (instead of inlining pi0.5 calls into
    the batching loop below) means the loop, resume logic, and manifest handling never need to
    change again once this class is implemented.
    """

    checkpoint_dir: Path
    dataset_repo_id: str
    dataset_root: Path | None
    dataset_revision: str | None
    action_key: str | None
    action_horizon: int
    action_dim: int

    @classmethod
    def load(
        cls,
        checkpoint_dir: Path,
        *,
        dataset_repo_id: str,
        dataset_root: Path | None,
        dataset_revision: str | None,
        action_key: str | None,
        rename_map: Mapping[str, str] | None,
        allow_download: bool,
    ) -> "Pi05CacheModel":
        # TODO: build/select an openpi TrainConfig for our pick_tube robot setup and call
        #   openpi.policies.policy_config.create_trained_policy(train_config, checkpoint_dir)
        # See ../pi05_frs_plan.md "checkpoint 来源" for why there is no ready-made "run pi05_base
        # zero-shot" TrainConfig in the openpi repo yet -- one has to be written, mirroring
        # pi05_aloha / pi05_droid / pi05_libero in openpi's src/openpi/training/config.py.
        raise NotImplementedError(
            "Pi05CacheModel.load: write a pick_tube TrainConfig + DataConfig and call "
            "openpi.policies.policy_config.create_trained_policy(...) here."
        )

    def prepare_sample(self, sample: Mapping[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        """Map one LeRobotDataset sample into pi0.5's observation dict, plus the GT action chunk.

        pi0.5 expects observation keys like `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`
        + `state` + a tokenized prompt (see openpi src/openpi/policies/*_policy.py for the pattern
        used by each of openpi's own robots). Our pick_tube cameras are
        `observation.images.camera1` / `camera2` (already renamed from `camera0`/`camera1`, see
        configs/train_frs_pick_tube_pi05.yaml `rename_map`) plus 4 tactile cameras that pi0.5 has
        no slot for -- tactile only ever goes through tactile_encoder downstream, so it should be
        fine to simply not pass it to pi0.5 here.
        """
        raise NotImplementedError("Pi05CacheModel.prepare_sample: map dataset sample -> pi0.5 observation dict.")

    def sample_and_reverse(
        self,
        observations: Sequence[dict[str, Any]],
        noise: np.ndarray,
        *,
        sample_steps: int,
        reverse_steps: int,
        solver: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample actions t:1->0 (like openpi Policy.infer / Pi0.sample_actions), then
        reverse-integrate them t:0->1 (like utils/source_model.sample_and_reverse) to get x_base.

        pi0.5 uses the same flow-matching convention as SmolVLA here (t=1 is noise, t=0 is data,
        dt = -1/num_steps -- see openpi src/openpi/models/pi0.py Pi0.sample_actions), so
        ../utils/integration.py's euler_integrate_velocity / fireflow_integrate_velocity can be
        reused verbatim once there is a per-step velocity_fn(x, t) -> v to hand them. openpi's
        Pi0.sample_actions does not expose that step function on its own (it is inlined in a
        jax.lax loop with a KV cache) -- build one from Pi0.embed_prefix / embed_suffix /
        action_out_proj, following how SmolVLA's denoise_step is used by
        utils/source_model.py:_jitted_reverse_from_context.
        """
        raise NotImplementedError(
            "Pi05CacheModel.sample_and_reverse: expose a velocity_fn(x, t) for pi0.5 and reuse "
            "utils/integration.py's euler/fireflow solver."
        )


def _deterministic_noise(indices: Sequence[int], shape: tuple[int, int], *, seed: int) -> np.ndarray:
    """numpy port of utils/source_model.deterministic_noise (kept local: see module docstring
    for why this bridge does not import that module).

    Each dataset index gets its own independent stream (via SeedSequence, passing `[seed, index]`
    as entropy), mirroring jax.random.fold_in(base_key, index) in utils/source_model.py: noise
    only depends on (seed, dataset_index), not on batch order or batch size.
    """
    out = np.empty((len(indices), *shape), dtype=np.float32)
    for row, index in enumerate(indices):
        out[row] = np.random.default_rng([seed, int(index)]).standard_normal(shape)
    return out


def _inversion_mse(x_base: np.ndarray, noise: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, x_base.ndim))
    return np.mean(np.square(x_base - noise), axis=axes).astype(np.float32)


def _checkpoint_fingerprint(checkpoint_dir: Path) -> str:
    """pi0.5 analogue of prepare.py:_checkpoint_fingerprint.

    TODO: verify against an actual downloaded pi0.5 checkpoint -- openpi stores params as an
    orbax checkpoint directory (not a single model.safetensors like the merged SmolVLA JAX
    checkpoints), so the file layout walked here may need adjusting once one is on disk.
    """
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    digest = hashlib.sha256()
    candidates = sorted(path for path in checkpoint_dir.rglob("*") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"no checkpoint files found under {checkpoint_dir}")
    for path in candidates:
        stat = path.stat()
        digest.update(str(path.relative_to(checkpoint_dir)).encode())
        digest.update(f":{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def prepare_cache(
    *,
    checkpoint_dir: Path,
    cache_dir: Path,
    dataset_repo_id: str,
    dataset_root: Path | None = None,
    dataset_revision: str | None = None,
    action_key: str | None = None,
    rename_map: Mapping[str, str] | None = None,
    allow_download: bool = False,
    model_sample_steps: int,
    reverse_steps: int,
    reverse_solver: str,
    batch_size: int,
    inference_seed: int,
    split_seed: int,
    val_fraction: float,
    frame_stride: int,
    max_episodes: int | None,
    max_samples: int | None,
    drop_tail_action_chunks: int = 1,
    flush_every: int = 8,
) -> dict[str, Any]:
    """pi0.5 analogue of prepare.py:prepare_cache. Structurally identical (records selection,
    manifest/resume handling, memmap arrays are all shared via utils/cache.py); the only
    difference is that batches are produced by Pi05CacheModel instead of SmolVLAEvalModel.
    """
    if model_sample_steps <= 0 or reverse_steps <= 0 or batch_size <= 0:
        raise ValueError("model_sample_steps, reverse_steps, and batch_size must all be positive.")
    if reverse_solver not in ("euler", "fireflow"):
        raise ValueError(f"reverse_solver must be 'euler' or 'fireflow', got {reverse_solver!r}.")
    if flush_every <= 0:
        raise ValueError(f"flush_every must be positive, got {flush_every}.")

    model = Pi05CacheModel.load(
        checkpoint_dir,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        dataset_revision=dataset_revision,
        action_key=action_key,
        rename_map=rename_map,
        allow_download=allow_download,
    )

    # TODO: once Pi05CacheModel.load exists, get `metadata` from openpi's own (pinned) lerobot
    # package -- e.g. `from lerobot.datasets import LeRobotDatasetMetadata` inside this bridge
    # venv, NOT from the main repo. build_records() only needs total_episodes/episodes[i][...],
    # so it works unmodified with that upstream class (see utils/cache.py:build_records docstring).
    metadata: Any = None  # placeholder until the TODO above is done
    records, train_episodes, val_episodes = build_records(
        metadata,
        val_fraction=val_fraction,
        split_seed=split_seed,
        frame_stride=frame_stride,
        max_episodes=max_episodes,
        max_samples=max_samples,
        action_horizon=model.action_horizon,
        drop_tail_action_chunks=drop_tail_action_chunks,
    )

    configuration = {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_fingerprint": _checkpoint_fingerprint(checkpoint_dir),
        "dataset_repo_id": dataset_repo_id,
        "dataset_root": str(dataset_root.resolve()) if dataset_root is not None else None,
        "dataset_revision": dataset_revision,
        "action_key": action_key,
        "rename_map": dict(rename_map) if rename_map is not None else None,
        "base_model": "pi0.5",
        "model_sample_steps": model_sample_steps,
        "reverse_steps": reverse_steps,
        "reverse_solver": reverse_solver,
        "inference_seed": inference_seed,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "frame_stride": frame_stride,
        "max_episodes": max_episodes,
        "max_samples": max_samples,
        "drop_tail_action_chunks": drop_tail_action_chunks,
    }
    digest = records_digest(records)
    manifest_path = cache_dir / MANIFEST_NAME
    action_horizon = model.action_horizon
    action_dim = model.action_dim

    if manifest_path.exists():
        manifest = load_manifest(cache_dir, require_complete=False)
        if manifest.get("configuration") != configuration or manifest.get("records_sha256") != digest:
            raise ValueError(
                f"Existing cache at {cache_dir} was created with different inputs. "
                "Choose a new cache directory instead of mixing runs."
            )
        if manifest.get("status") == "complete":
            print(f"cache already complete: {cache_dir}")
            return manifest
        arrays = open_cache_arrays(cache_dir, mode="r+")
        completed = int(manifest.get("completed_samples", 0))
        print(f"resuming cache at sample {completed}/{len(records)}")
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        arrays = create_cache_arrays(cache_dir, records, action_horizon=action_horizon, action_dim=action_dim)
        completed = 0
        manifest = {
            "version": CACHE_VERSION,
            "status": "incomplete",
            "completed_samples": 0,
            "sample_count": len(records),
            "train_sample_count": sum(record.split == "train" for record in records),
            "val_sample_count": sum(record.split == "val" for record in records),
            "train_episodes": list(train_episodes),
            "val_episodes": list(val_episodes),
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "configuration": configuration,
            "records_sha256": digest,
        }
        atomic_write_json(manifest_path, manifest)

    if completed == len(records):
        manifest["status"] = "complete"
        manifest["mean_source_inversion_mse"] = float(np.mean(np.asarray(arrays["inversion_mse"])))
        atomic_write_json(manifest_path, manifest)
        print(f"cache complete: {cache_dir}")
        return manifest

    # TODO: build a dataset iterator here (openpi's own pinned `lerobot.LeRobotDataset`, not this
    # repo's) to fetch `records[i].dataset_index` -> sample dict for model.prepare_sample().
    raise NotImplementedError(
        "prepare_cache: dataset loading + batching loop not implemented yet -- see "
        "Pi05CacheModel.prepare_sample / sample_and_reverse TODOs above, and "
        "prepare.py:prepare_cache in the main repo for the batching/flush pattern to mirror."
    )


def prepare_from_config(config: Mapping[str, Any]) -> list[Path]:
    checkpoint = Path(str(config["checkpoint"])).expanduser()
    datasets = config.get("datasets") or []
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("config.datasets must be a non-empty list")
    cache_config = config.get("action_cache") or {}
    if not isinstance(cache_config, Mapping) or not cache_config.get("root"):
        raise ValueError("config.action_cache.root is required")
    cache_root = Path(str(cache_config["root"])).expanduser()
    outputs: list[Path] = []

    for source_index, source in enumerate(datasets):
        if not isinstance(source, Mapping):
            raise ValueError(f"datasets[{source_index}] must be a mapping")
        repo_id = str(source["repo_id"])
        output = source_cache_dir(cache_root, repo_id)
        root_value = source.get("root")
        dataset_root = None if root_value in (None, "") else Path(str(root_value)).expanduser()
        print(f"prepare_source={source_index}:{repo_id} cache={output}", flush=True)
        prepare_cache(
            checkpoint_dir=checkpoint,
            cache_dir=output,
            dataset_repo_id=repo_id,
            dataset_root=dataset_root,
            dataset_revision=source.get("revision"),
            action_key=source.get("action_key"),
            rename_map=dict(source.get("rename_map") or {}),
            allow_download=bool(config.get("allow_download", False)),
            model_sample_steps=int(cache_config.get("model_sample_steps", 10)),
            reverse_steps=int(cache_config.get("reverse_steps", 50)),
            reverse_solver=str(cache_config.get("reverse_solver", "fireflow")),
            batch_size=int(cache_config.get("batch_size", 16)),
            inference_seed=int(cache_config.get("inference_seed", 0)),
            split_seed=int(cache_config.get("split_seed", 42)),
            val_fraction=float(cache_config.get("val_fraction", 0.1)),
            frame_stride=int(cache_config.get("frame_stride", 3)),
            max_episodes=(None if cache_config.get("max_episodes") is None else int(cache_config["max_episodes"])),
            max_samples=(None if cache_config.get("max_samples") is None else int(cache_config["max_samples"])),
            drop_tail_action_chunks=int(cache_config.get("drop_tail_action_chunks", 1)),
            flush_every=int(cache_config.get("flush_every", 8)),
        )
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    start = time.perf_counter()
    outputs = prepare_from_config(load_config(args.config))
    for output in outputs:
        print(f"action_cache={output}")
    print(f"done in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
