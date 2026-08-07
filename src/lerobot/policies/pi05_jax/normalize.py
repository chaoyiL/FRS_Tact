"""Normalization stats + apply/unapply formulas, adapted from openpi (Apache-2.0):
src/openpi/shared/normalize.py + the Normalize/Unnormalize transforms in src/openpi/transforms.py,
commit 15a9616a00943ada6c20a0f158e3adb39df2ccac (2026-06-16).

Deliberately NOT a byte-for-byte vendor: upstream's `NormStats`/JSON (de)serialization goes
through `pydantic` + `numpydantic`, and `load_norm_stats` through `etils.epath` -- three more
dependencies just to read a small JSON file. `NormStats` here is a plain dataclass and
`load_norm_stats`/`save_norm_stats` do the same `<assets_dir>/<asset_id>/norm_stats.json` layout
with stdlib `json`, byte-compatible with files written by upstream openpi (same key names/nesting)
so checkpoints downloaded via `../../../../pi05_frs_plan.md`'s `gs://openpi-assets/...` URLs load
with either implementation.

The z-score/quantile normalize math in `apply`/`unapply` is copied verbatim from
`openpi.transforms.Normalize`/`Unnormalize`.

IMPORTANT -- what this file does *not* decide: pi0.5's officially released checkpoints only ship
norm stats for the datasets they were pretrained/finetuned on (keyed by `asset_id`, e.g. "droid").
There is no `asset_id` for a brand-new robot/dataset like pick_tube. Whoever wires this up needs
to decide: (a) reuse one of the shipped asset_id's stats as an approximation, or (b) compute fresh
stats from the pick_tube dataset itself (openpi's own `scripts/compute_norm_stats.py` does this
for training data -- not vendored here). See pi05_frs_plan.md.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None  # 1st percentile
    q99: np.ndarray | None = None  # 99th percentile


def _to_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _from_json_stats(value: dict) -> NormStats:
    return NormStats(
        mean=_to_array(value["mean"]),
        std=_to_array(value["std"]),
        q01=None if value.get("q01") is None else _to_array(value["q01"]),
        q99=None if value.get("q99") is None else _to_array(value["q99"]),
    )


def _to_json_stats(stats: NormStats) -> dict:
    return {
        "mean": stats.mean.tolist(),
        "std": stats.std.tolist(),
        "q01": None if stats.q01 is None else stats.q01.tolist(),
        "q99": None if stats.q99 is None else stats.q99.tolist(),
    }


def load_norm_stats(assets_dir: str | Path, asset_id: str) -> dict[str, NormStats]:
    """Load `<assets_dir>/<asset_id>/norm_stats.json`, as written by openpi training/tooling."""
    path = Path(assets_dir) / asset_id / "norm_stats.json"
    if not path.is_file():
        raise FileNotFoundError(f"norm stats file not found: {path}")
    payload = json.loads(path.read_text())
    return {key: _from_json_stats(value) for key, value in payload["norm_stats"].items()}


def save_norm_stats(assets_dir: str | Path, asset_id: str, stats: dict[str, NormStats]) -> None:
    path = Path(assets_dir) / asset_id / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"norm_stats": {key: _to_json_stats(value) for key, value in stats.items()}}
    path.write_text(json.dumps(payload, indent=2))


def apply(x: np.ndarray, stats: NormStats, *, use_quantiles: bool = False) -> np.ndarray:
    """Normalize `x` into model space. Matches `openpi.transforms.Normalize`."""
    if use_quantiles:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("use_quantiles=True requires stats.q01/q99")
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
    return (x - mean) / (std + 1e-6)


def _pad_to_dim(x: np.ndarray, dim: int, *, value: float) -> np.ndarray:
    if x.shape[-1] >= dim:
        return x
    pad_width = [(0, 0)] * (x.ndim - 1) + [(0, dim - x.shape[-1])]
    return np.pad(x, pad_width, constant_values=value)


def unapply(x: np.ndarray, stats: NormStats, *, use_quantiles: bool = False) -> np.ndarray:
    """Undo `apply` (model space -> real units). Matches `openpi.transforms.Unnormalize`."""
    if use_quantiles:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("use_quantiles=True requires stats.q01/q99")
        q01, q99 = stats.q01, stats.q99
        dim = q01.shape[-1]
        if dim < x.shape[-1]:
            return np.concatenate(
                [(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1
            )
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    mean = _pad_to_dim(stats.mean, x.shape[-1], value=0.0)
    std = _pad_to_dim(stats.std, x.shape[-1], value=1.0)
    return x * (std + 1e-6) + mean
