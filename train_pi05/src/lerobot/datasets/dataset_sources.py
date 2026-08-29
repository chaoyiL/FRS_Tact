"""Model-neutral parsing helpers for multi-source LeRobot datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSource:
    """One LeRobot dataset declared in a training YAML file."""

    repo_id: str
    root: str | Path | None = None
    revision: str | None = None
    episodes: Sequence[int] | None = None
    action_key: str | None = None
    rename_map: Mapping[str, str] | None = None
    weight: float = 1.0


def resolve_source_visual_keys(
    model_image_keys: Sequence[str],
    rename_map: Mapping[str, str] | None,
    available_cameras: Sequence[str],
) -> list[str]:
    """Map post-rename model image keys back to dataset camera names."""

    rename_map = dict(rename_map or {})
    inverse = {dst: src for src, dst in rename_map.items()}
    available = set(available_cameras)
    resolved: list[str] = []
    missing: list[str] = []
    for key in model_image_keys:
        source = inverse.get(key, key)
        if source in available:
            resolved.append(source)
        elif key in available:
            resolved.append(key)
        else:
            missing.append(key)
    if missing:
        raise KeyError(
            f"could not resolve model image keys {missing} via rename_map={rename_map} "
            f"against cameras={list(available_cameras)}"
        )
    return list(dict.fromkeys(resolved))


def parse_dataset_sources(cfg: Mapping[str, Any]) -> list[DatasetSource]:
    """Build dataset sources from YAML ``datasets: [{repo_id, ...}, ...]``."""

    raw_datasets = cfg.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("datasets must be a non-empty list of dataset mappings")
    sources: list[DatasetSource] = []
    for index, item in enumerate(raw_datasets):
        if not isinstance(item, Mapping):
            raise ValueError(f"datasets[{index}] must be a mapping")
        if "repo_id" not in item or not item["repo_id"]:
            raise ValueError(f"datasets[{index}].repo_id is required")
        weight = float(item.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"datasets[{index}].weight must be positive")
        rename_map = item.get("rename_map") or {}
        if not isinstance(rename_map, Mapping):
            raise ValueError(f"datasets[{index}].rename_map must be a mapping")
        action_key = item.get("action_key")
        sources.append(
            DatasetSource(
                repo_id=str(item["repo_id"]),
                root=item.get("root"),
                revision=item.get("revision"),
                episodes=item.get("episodes"),
                action_key=None if action_key is None else str(action_key),
                rename_map=dict(rename_map),
                weight=weight,
            )
        )
    return sources
