#!/usr/bin/env python
"""Run a source-matched FireFlow/SlerpFlow inversion smoke test before FRS training."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_frs.prepare_frs_caches import prepare_cache
from utils.cache import atomic_write_json, load_manifest, open_cache_arrays

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_frs.yaml"
SOLVERS = ("fireflow", "slerpflow")


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


def summarize_inversion_mse(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    summary: dict[str, float | int] = {
        "sample_count": int(values.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(values.size - finite.size),
    }
    if finite.size == 0:
        summary.update(
            {
                "mean": math.inf,
                "median": math.inf,
                "p95": math.inf,
                "p99": math.inf,
                "max": math.inf,
            }
        )
        return summary
    summary.update(
        {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "p95": float(np.quantile(finite, 0.95)),
            "p99": float(np.quantile(finite, 0.99)),
            "max": float(np.max(finite)),
        }
    )
    return summary


def mean_ratio(numerator: Mapping[str, float | int], denominator: Mapping[str, float | int]) -> float:
    top = float(numerator["mean"])
    bottom = float(denominator["mean"])
    if bottom <= 1e-12:
        return 1.0 if top <= 1e-12 else math.inf
    return top / bottom


def _prepare_one(
    *,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    solver: str,
    output: Path,
    max_samples: int,
    max_episodes: int | None,
) -> dict[str, Any]:
    action_cache = config["action_cache"]
    checkpoint = Path(str(config["checkpoint"])).expanduser()
    root_value = source.get("root")
    dataset_root = None if root_value in (None, "") else Path(str(root_value)).expanduser()
    return prepare_cache(
        checkpoint_dir=checkpoint,
        cache_dir=output,
        dataset_repo_id=str(source["repo_id"]),
        dataset_root=dataset_root,
        dataset_revision=source.get("revision"),
        action_key=source.get("action_key"),
        rename_map=dict(source.get("rename_map") or {}),
        normalization_source="checkpoint",
        allow_download=bool(config.get("allow_download", False)),
        model_sample_steps=int(action_cache.get("model_sample_steps", 10)),
        reverse_steps=int(action_cache.get("reverse_steps", 50)),
        reverse_solver=solver,
        batch_size=int(action_cache.get("batch_size", 16)),
        inference_seed=int(action_cache.get("inference_seed", 0)),
        split_seed=int(action_cache.get("split_seed", 42)),
        val_fraction=float(action_cache.get("val_fraction", 0.1)),
        frame_stride=int(action_cache.get("frame_stride", 5)),
        max_episodes=max_episodes,
        max_samples=max_samples,
        drop_tail_action_chunks=int(action_cache.get("drop_tail_action_chunks", 1)),
        flush_every=int(action_cache.get("flush_every", 8)),
        num_workers=int(action_cache.get("num_workers", 0)),
        prefetch_factor=int(action_cache.get("prefetch_factor", 2)),
        video_backend=action_cache.get("video_backend"),
        worker_timeout_seconds=float(action_cache.get("worker_timeout_seconds", 300.0)),
    )


def compare_from_config(
    config: Mapping[str, Any],
    *,
    no_fail: bool = False,
) -> dict[str, Any]:
    settings = config.get("reverse_solver_ab") or {}
    if not isinstance(settings, Mapping):
        raise ValueError("config.reverse_solver_ab must be a mapping")
    if not bool(settings.get("enabled", True)):
        print("reverse_solver_ab disabled; skip")
        return {"status": "disabled"}
    if not settings.get("root"):
        raise ValueError("config.reverse_solver_ab.root is required")
    datasets = config.get("datasets") or []
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("config.datasets must be a non-empty list")
    action_cache = config.get("action_cache") or {}
    if not isinstance(action_cache, Mapping):
        raise ValueError("config.action_cache must be a mapping")
    max_samples = int(settings.get("max_samples_per_dataset", 512))
    if max_samples < 2:
        raise ValueError("reverse_solver_ab.max_samples_per_dataset must be at least 2")
    max_episodes_value = settings.get("max_episodes")
    max_episodes = None if max_episodes_value is None else int(max_episodes_value)
    max_mean_ratio = float(settings.get("max_mean_ratio", 1.0))
    if max_mean_ratio <= 0:
        raise ValueError("reverse_solver_ab.max_mean_ratio must be positive")
    root = Path(str(settings["root"])).expanduser()
    started = time.perf_counter()

    cache_dirs: dict[str, dict[str, Path]] = {solver: {} for solver in SOLVERS}
    for solver in SOLVERS:
        for source_index, source in enumerate(datasets):
            if not isinstance(source, Mapping):
                raise ValueError(f"datasets[{source_index}] must be a mapping")
            repo_id = str(source["repo_id"])
            output = source_cache_dir(root / solver, repo_id)
            cache_dirs[solver][repo_id] = output
            print(
                f"solver_ab prepare solver={solver} source={source_index}:{repo_id} "
                f"samples={max_samples} cache={output}",
                flush=True,
            )
            _prepare_one(
                config=config,
                source=source,
                solver=solver,
                output=output,
                max_samples=max_samples,
                max_episodes=max_episodes,
            )

    per_dataset: dict[str, Any] = {}
    combined: dict[str, list[np.ndarray]] = {solver: [] for solver in SOLVERS}
    failures: list[str] = []
    for source in datasets:
        repo_id = str(source["repo_id"])
        manifests = {
            solver: load_manifest(cache_dirs[solver][repo_id]) for solver in SOLVERS
        }
        if manifests["fireflow"]["records_sha256"] != manifests["slerpflow"]["records_sha256"]:
            failures.append(f"{repo_id}: solver caches contain different sample records")
        summaries: dict[str, Any] = {}
        arrays_by_solver = {
            solver: open_cache_arrays(cache_dirs[solver][repo_id])
            for solver in SOLVERS
        }
        prediction_difference = np.max(
            np.abs(
                np.asarray(arrays_by_solver["fireflow"]["target"], dtype=np.float32)
                - np.asarray(arrays_by_solver["slerpflow"]["target"], dtype=np.float32)
            )
        )
        summaries["prediction_max_abs_diff"] = float(prediction_difference)
        if not np.isfinite(prediction_difference) or prediction_difference > 1e-6:
            failures.append(
                f"{repo_id}: solver caches do not share identical predictions "
                f"(max abs diff={prediction_difference:.8g})"
            )
        for solver in SOLVERS:
            values = np.asarray(
                arrays_by_solver[solver]["inversion_mse"],
                dtype=np.float64,
            )
            combined[solver].append(values)
            summaries[solver] = summarize_inversion_mse(values)
            if int(summaries[solver]["nonfinite_count"]) != 0:
                failures.append(f"{repo_id}: {solver} produced non-finite inversion MSE")
        summaries["slerp_to_fire_mean_ratio"] = mean_ratio(
            summaries["slerpflow"], summaries["fireflow"]
        )
        per_dataset[repo_id] = summaries

    global_summaries = {
        solver: summarize_inversion_mse(np.concatenate(combined[solver]))
        for solver in SOLVERS
    }
    ratio = mean_ratio(global_summaries["slerpflow"], global_summaries["fireflow"])
    if ratio > max_mean_ratio:
        failures.append(
            f"global SlerpFlow/FireFlow mean ratio {ratio:.6f} exceeds {max_mean_ratio:.6f}"
        )
    report: dict[str, Any] = {
        "status": "passed" if not failures else "failed",
        "max_mean_ratio": max_mean_ratio,
        "slerp_to_fire_mean_ratio": ratio,
        "elapsed_seconds": float(time.perf_counter() - started),
        "max_samples_per_dataset": max_samples,
        "per_dataset": per_dataset,
        "global": global_summaries,
        "failures": failures,
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "comparison.json", report)

    print("reverse_solver_ab results:", flush=True)
    for repo_id, summaries in per_dataset.items():
        fire = summaries["fireflow"]
        slerp = summaries["slerpflow"]
        print(
            f"  {repo_id}: fire_mean={fire['mean']:.8g} slerp_mean={slerp['mean']:.8g} "
            f"ratio={summaries['slerp_to_fire_mean_ratio']:.6f} "
            f"fire_p95={fire['p95']:.8g} slerp_p95={slerp['p95']:.8g}",
            flush=True,
        )
    print(
        f"  global: fire_mean={global_summaries['fireflow']['mean']:.8g} "
        f"slerp_mean={global_summaries['slerpflow']['mean']:.8g} ratio={ratio:.6f} "
        f"status={report['status']} report={root / 'comparison.json'}",
        flush=True,
    )
    if failures and not no_fail:
        raise RuntimeError("reverse solver A/B failed: " + "; ".join(failures))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Write and print the comparison without failing when SlerpFlow is worse.",
    )
    args = parser.parse_args()
    compare_from_config(load_config(args.config), no_fail=args.no_fail)


if __name__ == "__main__":
    main()
