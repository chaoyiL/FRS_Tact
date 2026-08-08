from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from safetensors.numpy import load_file as load_safetensors_file
from safetensors.numpy import save_file as save_safetensors_file

from lerobot.datasets import LeRobotDatasetMetadata, aggregate_stats
from lerobot.datasets.io_utils import cast_stats_to_numpy, load_info, load_nested_dataset
from lerobot.datasets.utils import EPISODES_DIR, INFO_PATH
from lerobot.utils.utils import unflatten_dict

from .data import (
    CANONICAL_ACTION_KEY,
    DatasetSource,
    canonicalize_dataset_stats,
    rename_dataset_stats,
    resolve_action_key,
)

DATA_SPLIT_FILENAME = "data_split.json"
NORMALIZATION_MANIFEST_FILENAME = "normalization_manifest.json"
PREPROCESSOR_STATS_FILENAME = "policy_preprocessor_step_5_normalizer_processor.safetensors"
POSTPROCESSOR_STATS_FILENAME = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
NORMALIZATION_ALGORITHM_VERSION = 1
_CANONICAL_FEATURES = ("observation.state", CANONICAL_ACTION_KEY)
_REQUIRED_STATS = ("min", "max", "mean", "std", "count")


class NormalizationProtocolResult(NamedTuple):
    stats: dict[str, dict[str, np.ndarray]]
    split_path: Path
    manifest_path: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_float32_stats(
    stats: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    return {
        feature: {
            stat: np.ascontiguousarray(np.asarray(value, dtype="<f4"))
            for stat, value in sorted(feature_stats.items())
        }
        for feature, feature_stats in sorted(stats.items())
    }


def _stats_digest(stats: Mapping[str, Mapping[str, Any]]) -> str:
    canonical = _canonical_float32_stats(stats)
    payload = {
        feature: {
            stat: {
                "dtype": "float32",
                "shape": list(value.shape),
                "bytes": value.tobytes(order="C").hex(),
            }
            for stat, value in feature_stats.items()
        }
        for feature, feature_stats in canonical.items()
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selected_episode_digest(records: Sequence[tuple[int, str]]) -> str:
    payload = [
        {"episode_index": int(episode_index), "stats_sha256": stats_sha256}
        for episode_index, stats_sha256 in records
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _flatten_stats(stats: Mapping[str, Mapping[str, Any]]) -> dict[str, np.ndarray]:
    return {
        f"{feature}.{stat}": np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        for feature, feature_stats in sorted(stats.items())
        for stat, value in sorted(feature_stats.items())
    }


def _unflatten_stats(stats: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    nested: dict[str, dict[str, np.ndarray]] = {}
    for key, value in stats.items():
        try:
            feature, stat = key.rsplit(".", 1)
        except ValueError as exc:
            raise ValueError(f"invalid normalization asset key: {key!r}") from exc
        nested.setdefault(feature, {})[stat] = np.asarray(value, dtype=np.float32)
    return _canonical_float32_stats(nested)


def _validate_feature_stats(
    stats: Mapping[str, Mapping[str, Any]],
    *,
    state_dim: int,
    action_dim: int,
    context: str,
) -> None:
    dimensions = {"observation.state": state_dim, CANONICAL_ACTION_KEY: action_dim}
    if set(stats) != set(_CANONICAL_FEATURES):
        raise ValueError(
            f"{context} stats must canonicalize exactly to {_CANONICAL_FEATURES}, got {sorted(stats)}"
        )
    for feature, dimension in dimensions.items():
        feature_stats = stats[feature]
        missing = set(_REQUIRED_STATS) - set(feature_stats)
        if missing:
            raise ValueError(f"{context} stats for {feature!r} are missing {sorted(missing)}")
        for stat in _REQUIRED_STATS:
            try:
                value = np.asarray(feature_stats[stat])
                finite = bool(np.all(np.isfinite(value)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{context} stats for {feature!r}.{stat} are not numeric") from exc
            expected_shape = (1,) if stat == "count" else (dimension,)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{context} stats for {feature!r}.{stat} have shape {value.shape}, "
                    f"expected {expected_shape}"
                )
            if not finite:
                raise ValueError(f"{context} stats for {feature!r}.{stat} must be finite")
            if stat == "count" and float(value[0]) <= 0:
                raise ValueError(f"{context} stats for {feature!r}.count must be positive")


def _load_split(path: Path, sources: Sequence[DatasetSource]) -> None:
    try:
        split = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid persisted episode split {path}: {exc}") from exc
    if not isinstance(split, Mapping) or int(split.get("version", -1)) != 1:
        raise ValueError(f"invalid persisted episode split version in {path}")
    entries = split.get("datasets")
    if not isinstance(entries, list) or len(entries) != len(sources):
        raise ValueError("persisted episode split source count mismatch")
    for source, entry in zip(sources, entries, strict=True):
        if not isinstance(entry, Mapping) or entry.get("repo_id") != source.repo_id:
            raise ValueError(f"persisted episode split source order mismatch for {source.repo_id!r}")
        if entry.get("revision") != source.revision:
            raise ValueError(f"persisted episode split revision mismatch for {source.repo_id!r}")
        requested = [int(value) for value in source.episodes or ()]
        train_ids = [int(value) for value in entry.get("train_episodes", [])]
        val_ids = [int(value) for value in entry.get("val_episodes", [])]
        if not requested:
            raise ValueError(f"normalization requires explicit train episodes for {source.repo_id!r}")
        if len(requested) != len(set(requested)):
            raise ValueError(f"requested train episodes must be unique for {source.repo_id!r}")
        if len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids)):
            raise ValueError(f"persisted episode split contains duplicate episodes for {source.repo_id!r}")
        if set(train_ids) & set(val_ids):
            raise ValueError(f"persisted episode split overlaps train/val for {source.repo_id!r}")
        if sorted(requested) != sorted(train_ids):
            raise ValueError(f"persisted episode split train coverage mismatch for {source.repo_id!r}")


def _selected_source_stats(
    source: DatasetSource,
    *,
    state_dim: int,
    action_dim: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    requested = [int(value) for value in source.episodes or ()]
    if not requested:
        raise ValueError(f"normalization requires explicit train episodes for {source.repo_id!r}")
    if len(requested) != len(set(requested)):
        raise ValueError(f"requested train episodes must be unique for {source.repo_id!r}")
    requested = sorted(requested)

    requested_root = None if source.root is None else Path(source.root).expanduser()
    if requested_root is not None and (requested_root / INFO_PATH).is_file():
        info = load_info(requested_root)
        metadata_root = requested_root
        features = info.features
        codebase_version = info.codebase_version
        resolved_revision = source.revision or info.codebase_version
    else:
        metadata = LeRobotDatasetMetadata(
            repo_id=source.repo_id,
            root=source.root,
            revision=source.revision,
        )
        metadata_root = Path(metadata.root)
        features = metadata.features
        codebase_version = getattr(getattr(metadata, "info", None), "codebase_version", None)
        resolved_revision = getattr(metadata, "revision", source.revision)
    if codebase_version is not None and not str(codebase_version).startswith("v3."):
        raise ValueError(
            f"dataset {source.repo_id!r} must use LeRobot v3 per-episode stats, got {codebase_version!r}"
        )
    resolved_action_key = resolve_action_key(features, source.action_key)
    episodes = load_nested_dataset(metadata_root / EPISODES_DIR, episodes=requested)
    column_names = list(episodes.column_names)
    prefixes = ("stats/observation.state/", f"stats/{resolved_action_key}/")
    selected_columns = [
        "episode_index",
        *(column for column in column_names if column.startswith(prefixes)),
    ]
    episodes = episodes.select_columns(selected_columns)

    loaded_ids = [int(value) for value in episodes["episode_index"]]
    if len(loaded_ids) != len(set(loaded_ids)):
        raise ValueError(f"episode metadata contains duplicate episode indices for {source.repo_id!r}")
    if sorted(loaded_ids) != requested:
        raise ValueError(
            f"episode metadata coverage mismatch for {source.repo_id!r}: "
            f"requested={requested} loaded={sorted(loaded_ids)}"
        )

    per_episode: list[dict[str, dict[str, np.ndarray]]] = []
    episode_digests: list[tuple[int, str]] = []
    for row in sorted(episodes, key=lambda item: int(item["episode_index"])):
        flat_stats = {
            key.removeprefix("stats/"): value
            for key, value in row.items()
            if key.startswith("stats/")
        }
        raw_stats = cast_stats_to_numpy(unflatten_dict(flat_stats))
        try:
            canonical = rename_dataset_stats(
                canonicalize_dataset_stats(raw_stats, resolved_action_key),
                source.rename_map,
            )
        except KeyError as exc:
            raise ValueError(
                f"dataset {source.repo_id!r} episode {int(row['episode_index'])} has invalid stats: {exc}"
            ) from exc
        canonical = {feature: canonical[feature] for feature in _CANONICAL_FEATURES if feature in canonical}
        context = f"dataset {source.repo_id!r} episode {int(row['episode_index'])}"
        _validate_feature_stats(
            canonical,
            state_dim=state_dim,
            action_dim=action_dim,
            context=context,
        )
        canonical = _canonical_float32_stats(canonical)
        per_episode.append(canonical)
        episode_digests.append((int(row["episode_index"]), _stats_digest(canonical)))

    selected_stats = _canonical_float32_stats(aggregate_stats(per_episode))
    _validate_feature_stats(
        selected_stats,
        state_dim=state_dim,
        action_dim=action_dim,
        context=f"dataset {source.repo_id!r} aggregate",
    )
    source_manifest = {
        "repo_id": source.repo_id,
        "requested_revision": source.revision,
        "resolved_revision": resolved_revision,
        "requested_action_key": source.action_key,
        "resolved_action_key": resolved_action_key,
        "rename_map": dict(sorted((source.rename_map or {}).items())),
        "train_episodes": requested,
        "selected_stats_sha256": _selected_episode_digest(episode_digests),
    }
    return selected_stats, source_manifest


def _load_and_validate_assets(
    artifact_dir: Path,
    expected_stats: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    pre_path = artifact_dir / PREPROCESSOR_STATS_FILENAME
    post_path = artifact_dir / POSTPROCESSOR_STATS_FILENAME
    try:
        pre_stats = _unflatten_stats(load_safetensors_file(pre_path))
        post_stats = _unflatten_stats(load_safetensors_file(post_path))
    except Exception as exc:
        raise ValueError(f"corrupt normalization asset in {artifact_dir}: {exc}") from exc
    expected_pre = _canonical_float32_stats(expected_stats)
    expected_post = {CANONICAL_ACTION_KEY: expected_pre[CANONICAL_ACTION_KEY]}
    for label, actual, expected in (
        ("preprocessor", pre_stats, expected_pre),
        ("postprocessor", post_stats, expected_post),
    ):
        if set(actual) != set(expected):
            raise ValueError(f"{label} normalization asset feature mismatch")
        for feature, feature_stats in expected.items():
            if set(actual[feature]) != set(feature_stats):
                raise ValueError(f"{label} normalization asset stats mismatch for {feature!r}")
            for stat, expected_value in feature_stats.items():
                if not np.array_equal(actual[feature][stat], expected_value):
                    raise ValueError(f"{label} normalization asset value mismatch for {feature}.{stat}")
    return pre_stats


def _manifest(
    *,
    split_sha256: str,
    datasets: list[dict[str, Any]],
    stats: Mapping[str, Mapping[str, Any]],
    state_dim: int,
    action_dim: int,
    asset_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "algorithm_version": NORMALIZATION_ALGORITHM_VERSION,
        "split_sha256": split_sha256,
        "datasets": datasets,
        "dimensions": {"observation.state": int(state_dim), "action": int(action_dim)},
        "canonical_stats_sha256": _stats_digest(stats),
        "assets": dict(sorted(asset_hashes.items())),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def build_or_validate_normalization_protocol(
    protocol_dir: str | Path,
    *,
    split_path: str | Path,
    sources: Sequence[DatasetSource],
    state_dim: int = 20,
    action_dim: int = 20,
    allow_create: bool = True,
) -> NormalizationProtocolResult:
    """Build or strictly validate immutable train-episode normalization assets."""

    protocol_dir = Path(protocol_dir).expanduser()
    split_path = Path(split_path).expanduser()
    if state_dim != 20 or action_dim != 20:
        raise ValueError(
            f"VT normalization protocol requires state_dim=action_dim=20, got {state_dim}/{action_dim}"
        )
    if not sources:
        raise ValueError("normalization protocol requires at least one dataset source")
    if len({source.repo_id for source in sources}) != len(sources):
        raise ValueError("normalization protocol requires unique dataset repo_id values")
    if not split_path.is_file():
        raise ValueError(f"normalization protocol split is missing: {split_path}")
    _load_split(split_path, sources)

    source_stats: list[dict[str, dict[str, np.ndarray]]] = []
    dataset_manifests: list[dict[str, Any]] = []
    for source in sources:
        stats, source_manifest = _selected_source_stats(
            source,
            state_dim=state_dim,
            action_dim=action_dim,
        )
        source_stats.append(stats)
        dataset_manifests.append(source_manifest)
    canonical_stats = _canonical_float32_stats(aggregate_stats(source_stats))
    _validate_feature_stats(
        canonical_stats,
        state_dim=state_dim,
        action_dim=action_dim,
        context="final canonical",
    )

    required_paths = {
        DATA_SPLIT_FILENAME: protocol_dir / DATA_SPLIT_FILENAME,
        PREPROCESSOR_STATS_FILENAME: protocol_dir / PREPROCESSOR_STATS_FILENAME,
        POSTPROCESSOR_STATS_FILENAME: protocol_dir / POSTPROCESSOR_STATS_FILENAME,
        NORMALIZATION_MANIFEST_FILENAME: protocol_dir / NORMALIZATION_MANIFEST_FILENAME,
    }
    if protocol_dir.exists():
        if not protocol_dir.is_dir():
            raise ValueError(f"normalization protocol artifact is not a directory: {protocol_dir}")
        missing = [name for name, path in required_paths.items() if not path.is_file()]
        if missing:
            raise ValueError(f"normalization protocol artifact is missing files: {missing}")
        try:
            existing_manifest = json.loads(
                required_paths[NORMALIZATION_MANIFEST_FILENAME].read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"corrupt normalization manifest: {exc}") from exc
        if required_paths[DATA_SPLIT_FILENAME].read_bytes() != split_path.read_bytes():
            raise ValueError("normalization protocol split content mismatch")
        asset_hashes = {
            name: _sha256_file(required_paths[name])
            for name in (PREPROCESSOR_STATS_FILENAME, POSTPROCESSOR_STATS_FILENAME)
        }
        expected_manifest = _manifest(
            split_sha256=_sha256_file(split_path),
            datasets=dataset_manifests,
            stats=canonical_stats,
            state_dim=state_dim,
            action_dim=action_dim,
            asset_hashes=asset_hashes,
        )
        if existing_manifest != expected_manifest:
            raise ValueError("normalization protocol manifest digest/source mismatch")
        authoritative_stats = _load_and_validate_assets(protocol_dir, canonical_stats)
        return NormalizationProtocolResult(
            stats=authoritative_stats,
            split_path=required_paths[DATA_SPLIT_FILENAME],
            manifest_path=required_paths[NORMALIZATION_MANIFEST_FILENAME],
        )

    if not allow_create:
        raise ValueError(f"normalization protocol artifact is missing: {protocol_dir}")
    protocol_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{protocol_dir.name}.staging-", dir=protocol_dir.parent))
    try:
        shutil.copy2(split_path, staging / DATA_SPLIT_FILENAME)
        save_safetensors_file(_flatten_stats(canonical_stats), staging / PREPROCESSOR_STATS_FILENAME)
        save_safetensors_file(
            _flatten_stats({CANONICAL_ACTION_KEY: canonical_stats[CANONICAL_ACTION_KEY]}),
            staging / POSTPROCESSOR_STATS_FILENAME,
        )
        asset_hashes = {
            name: _sha256_file(staging / name)
            for name in (PREPROCESSOR_STATS_FILENAME, POSTPROCESSOR_STATS_FILENAME)
        }
        manifest = _manifest(
            split_sha256=_sha256_file(staging / DATA_SPLIT_FILENAME),
            datasets=dataset_manifests,
            stats=canonical_stats,
            state_dim=state_dim,
            action_dim=action_dim,
            asset_hashes=asset_hashes,
        )
        _write_json(staging / NORMALIZATION_MANIFEST_FILENAME, manifest)
        _load_and_validate_assets(staging, canonical_stats)
        if protocol_dir.exists():
            raise ValueError(f"normalization protocol appeared during creation: {protocol_dir}")
        os.replace(staging, protocol_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return NormalizationProtocolResult(
        stats=canonical_stats,
        split_path=protocol_dir / DATA_SPLIT_FILENAME,
        manifest_path=protocol_dir / NORMALIZATION_MANIFEST_FILENAME,
    )
