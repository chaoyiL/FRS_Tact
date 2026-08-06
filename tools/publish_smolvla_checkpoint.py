#!/usr/bin/env python
"""Validate, assemble, repair, and safely publish SmolVLA inference bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lerobot.policies.smolvla_jax.validation import (
    CheckpointContract,
    CheckpointValidationReport,
    validate_checkpoint,
)

MODEL_FILENAME = "model.safetensors"
MANIFEST_FILENAME = "conversion_manifest.json"
SIDECAR_FILENAMES = (
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
INFERENCE_FILENAMES = (MODEL_FILENAME, *SIDECAR_FILENAMES, MANIFEST_FILENAME)
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_LEGACY_METADATA_ALLOW_PATTERNS = ("meta/info.json", "meta/episodes_stats.jsonl")


@dataclass(frozen=True)
class _MetadataStats:
    """Minimal metadata surface needed to reconstruct normalization sidecars."""

    total_frames: int
    features: Mapping[str, Any]
    stats: Mapping[str, Mapping[str, Any]]
    metadata_source: str
    legacy_conversion_proof: str | None = None


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _contract_dict(contract: CheckpointContract) -> dict[str, Any]:
    value = asdict(contract)
    for key in ("image_keys", "tactile_keys", "vlm_lora_target_modules"):
        value[key] = list(value[key])
    return value


def _contract_from_dict(value: Mapping[str, Any]) -> CheckpointContract:
    fields = dict(value)
    for key in ("image_keys", "tactile_keys", "vlm_lora_target_modules"):
        fields[key] = tuple(fields.get(key) or ())
    return CheckpointContract(**fields)


def _report_dict(report: CheckpointValidationReport) -> dict[str, Any]:
    return {"ok": report.ok, "path": str(report.path), "issues": list(report.issues)}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("PyYAML is required to read a training config") from exc

    if not path.is_file():
        raise FileNotFoundError(f"training config does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"training config root must be a mapping: {path}")
    return value


def contract_from_training_yaml(path: str | Path) -> CheckpointContract:
    """Read the authoritative inference contract from a training YAML model block."""

    config = _load_yaml(Path(path).expanduser().resolve())
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("training config must contain a model mapping")
    required = ("state_dim", "action_dim", "chunk_size", "image_keys")
    missing = [key for key in required if key not in model]
    if missing:
        raise ValueError(f"training model config is missing contract fields: {missing}")
    use_tactile = bool(model.get("use_tactile_encoder", False))
    tactile_keys = tuple(model.get("tactile_keys") or ()) if use_tactile else ()
    tactile_tokens = int(model.get("tactile_num_tokens", 0)) if use_tactile else 0
    if use_tactile and len(tactile_keys) != tactile_tokens:
        raise ValueError("training tactile_keys length must equal tactile_num_tokens")
    image_keys = tuple(model["image_keys"])
    overlap = sorted(set(image_keys) & set(tactile_keys))
    if overlap:
        raise ValueError(f"training RGB and tactile keys overlap: {overlap}")
    return CheckpointContract(
        state_dim=int(model["state_dim"]),
        action_dim=int(model["action_dim"]),
        chunk_size=int(model["chunk_size"]),
        image_keys=image_keys,
        tactile_keys=tactile_keys,
        tactile_embedding_dim=int(model.get("tactile_embedding_dim", 512)),
        tactile_num_tokens=tactile_tokens,
        lora_rank=int(model.get("lora_rank", 0)),
        vlm_lora_target_modules=tuple(model.get("vlm_lora_target_modules") or ()),
    )


def _reject_incomplete(path: Path) -> None:
    if any(part.endswith(".incomplete") for part in path.parts):
        raise ValueError(f"refusing an incomplete checkpoint or bundle: {path}")


def _copy_payload(source: Path, destination: Path) -> None:
    """Copy payload bytes so the bundle cannot be mutated through a shared inode."""

    shutil.copy2(source, destination)


def _manifest(
    staging: Path,
    *,
    source: str,
    source_weight_sha256: str,
    expected: CheckpointContract,
    dataset_revisions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    payload_files = sorted(path.name for path in staging.iterdir() if path.is_file())
    return {
        "format_version": 1,
        "source": source,
        "source_weight_sha256": source_weight_sha256,
        "contract": _contract_dict(expected),
        "datasets": [dict(item) for item in dataset_revisions],
        # A manifest cannot contain its own stable digest. This covers every payload file.
        "files": {
            name: {"sha256": _sha256(staging / name), "size": (staging / name).stat().st_size}
            for name in payload_files
        },
    }


def build_inference_bundle(
    source: str | Path,
    destination: str | Path,
    *,
    expected: CheckpointContract,
    include_model: bool = True,
    dataset_revisions: Iterable[Mapping[str, Any]] = (),
    base_sidecars: str | Path | None = None,
) -> Path:
    """Atomically build a validated bundle containing only inference artifacts."""

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    _reject_incomplete(source_path)
    if destination_path.exists():
        raise FileExistsError(f"bundle destination already exists: {destination_path}")
    source_report = validate_checkpoint(
        source_path,
        expected=expected,
        base_sidecars=base_sidecars,
        require_weight=True,
    )
    source_report.require_valid()

    staging = destination_path.with_name(f".{destination_path.name}.{uuid.uuid4().hex}.incomplete")
    if staging.exists():
        raise FileExistsError(f"bundle staging path already exists: {staging}")
    staging.mkdir(parents=True)
    for filename in SIDECAR_FILENAMES:
        shutil.copy2(source_path / filename, staging / filename)
    source_weight = source_path / MODEL_FILENAME
    weight_sha = _sha256(source_weight)
    if include_model:
        _copy_payload(source_weight, staging / MODEL_FILENAME)
    _write_json(
        staging / MANIFEST_FILENAME,
        _manifest(
            staging,
            source=str(source_path),
            source_weight_sha256=weight_sha,
            expected=expected,
            dataset_revisions=dataset_revisions,
        ),
    )
    staged_report = validate_checkpoint(staging, expected=expected, require_weight=include_model)
    staged_report.require_valid()
    staging.replace(destination_path)
    return destination_path


def _as_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=value.tzinfo or UTC).astimezone(UTC)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            return None
    return None


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def resolve_dataset_revisions(
    datasets: Sequence[Mapping[str, Any]],
    *,
    model_weight_uploaded_at: datetime | None,
    api: Any,
) -> list[dict[str, Any]]:
    """Resolve immutable dataset SHAs, refusing revisions that cannot be proven."""

    resolved: list[dict[str, Any]] = []
    weight_time = _as_datetime(model_weight_uploaded_at)
    for source in datasets:
        repo_id = str(source.get("repo_id") or "")
        if not repo_id:
            raise ValueError("dataset entry is missing repo_id")
        requested = source.get("revision")
        if requested is not None:
            requested = str(requested)
            if not _COMMIT_SHA_RE.fullmatch(requested):
                raise ValueError(
                    f"dataset {repo_id!r} revision must be an immutable commit SHA, got {requested!r}"
                )
            if weight_time is None:
                raise ValueError(
                    f"cannot prove training-time dataset revision for {repo_id!r}: "
                    "model weight upload time is unavailable"
                )
            info = api.dataset_info(repo_id, revision=requested)
            actual = str(_attribute(info, "sha", ""))
            if actual != requested:
                raise ValueError(
                    f"dataset {repo_id!r} resolved to {actual!r}, expected explicit SHA {requested!r}"
                )
            commits = list(
                api.list_repo_commits(
                    repo_id,
                    repo_type="dataset",
                    revision=requested,
                )
            )
            matching = next(
                (commit for commit in commits if str(_attribute(commit, "commit_id", "")) == requested),
                None,
            )
            created_at = _as_datetime(_attribute(matching, "created_at"))
            if created_at is None:
                raise ValueError(
                    f"cannot prove explicit dataset revision {requested} for {repo_id!r}: "
                    "commit timestamp is unavailable"
                )
            if created_at > weight_time:
                raise ValueError(
                    f"explicit dataset revision {requested} for {repo_id!r} postdates model weight upload"
                )
            proof = "explicit immutable revision"
        else:
            if weight_time is None:
                raise ValueError(
                    f"cannot prove training-time dataset revision for {repo_id!r}: "
                    "model weight upload time is unavailable"
                )
            info = api.dataset_info(repo_id, revision=None)
            actual = str(_attribute(info, "sha", ""))
            commits = list(api.list_repo_commits(repo_id, repo_type="dataset"))
            head = str(_attribute(commits[0], "commit_id", "")) if commits else ""
            commit_time = _as_datetime(_attribute(commits[0], "created_at")) if commits else None
            modified_time = _as_datetime(_attribute(info, "last_modified"))
            if (
                not _COMMIT_SHA_RE.fullmatch(actual)
                or head != actual
                or commit_time is None
                or modified_time is None
                or commit_time > weight_time
                or modified_time > weight_time
            ):
                raise ValueError(
                    f"cannot prove training-time dataset revision for {repo_id!r}: "
                    "the current repository head does not predate the model weight"
                )
            proof = "repository head predates model weight"
        entry = {"repo_id": repo_id, "revision": actual, "revision_proof": proof}
        for key in ("action_key", "rename_map", "weight"):
            if key in source:
                entry[key] = source[key]
        resolved.append(entry)
    return resolved


def _lfs_sha(value: Any) -> str | None:
    lfs = _attribute(value, "lfs")
    sha = _attribute(lfs, "sha256") if lfs is not None else None
    return str(sha) if sha else None


def _remote_repo_sha(api: Any, repo_id: str, revision: str | None) -> str:
    info = api.repo_info(repo_id, repo_type="model", revision=revision)
    sha = str(_attribute(info, "sha", ""))
    if not _COMMIT_SHA_RE.fullmatch(sha):
        raise ValueError(f"could not resolve immutable model revision for {repo_id!r}: {sha!r}")
    return sha


def _remote_weight_info(api: Any, repo_id: str, revision: str | None = None) -> tuple[str, datetime | None]:
    files = api.get_paths_info(
        repo_id,
        [MODEL_FILENAME],
        repo_type="model",
        revision=revision,
        expand=True,
    )
    candidate = next(
        (file for file in files if _attribute(file, "path", _attribute(file, "rfilename")) == MODEL_FILENAME),
        None,
    )
    if candidate is None:
        raise FileNotFoundError(f"remote repository {repo_id!r} has no {MODEL_FILENAME}")
    sha = _lfs_sha(candidate)
    if not sha:
        raise ValueError(f"remote {MODEL_FILENAME} has no LFS SHA-256 metadata")
    last_commit = _attribute(candidate, "last_commit")
    timestamp = _as_datetime(_attribute(last_commit, "date"))
    return sha, timestamp


def _load_manifest(bundle: Path) -> tuple[dict[str, Any], CheckpointContract]:
    path = bundle / MANIFEST_FILENAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid bundle manifest: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("contract"), dict):
        raise ValueError("invalid bundle manifest: missing contract")
    return manifest, _contract_from_dict(manifest["contract"])


def _safe_bundle_files(bundle: Path) -> set[str]:
    entries = list(bundle.iterdir())
    names = {entry.name for entry in entries}
    for entry in entries:
        if entry.is_symlink():
            raise ValueError(f"bundle payload must not contain symbolic links: {entry.name}")
        try:
            resolved = entry.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"could not resolve bundle payload {entry.name!r}: {exc}") from exc
        if resolved.parent != bundle or not entry.is_file():
            raise ValueError(f"bundle payload must be a regular child file: {entry.name}")
    return names


def _snapshot_regular_child(source: Path, destination: Path) -> None:
    """Copy one already-screened child through a no-follow file descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValueError(f"could not securely open bundle payload {source.name!r}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"bundle payload must be a regular file: {source.name}")
        with (
            os.fdopen(descriptor, "rb", closefd=False) as input_file,
            destination.open("xb") as output_file,
        ):
            shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
    finally:
        os.close(descriptor)


def _verify_manifest_files(
    bundle: Path,
    manifest: Mapping[str, Any],
    *,
    payload_names: set[str],
) -> None:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("invalid bundle manifest: files must be a mapping")
    if set(files) != payload_names:
        raise ValueError(
            "manifest files must exactly match bundle payload allowlist "
            f"(manifest={sorted(files)}, payload={sorted(payload_names)})"
        )
    for filename, metadata in files.items():
        if filename == MANIFEST_FILENAME or filename not in INFERENCE_FILENAMES:
            raise ValueError(f"invalid bundle manifest file: {filename!r}")
        path = bundle / filename
        if not path.is_file():
            raise ValueError(f"bundle manifest references missing file: {filename}")
        expected_sha = _attribute(metadata, "sha256")
        expected_size = _attribute(metadata, "size")
        if expected_sha != _sha256(path) or expected_size != path.stat().st_size:
            raise ValueError(f"bundle file does not match manifest: {filename}")


def publish_bundle(
    bundle: str | Path,
    *,
    repo_id: str,
    expected: CheckpointContract,
    api: Any | None = None,
    revision: str | None = None,
    sidecars_only: bool = True,
    commit_message: str = "Repair VT-SmolVLA inference sidecars",
) -> dict[str, Any]:
    """Publish a validated bundle, preserving remote weights in sidecar-only mode."""

    bundle_path = Path(bundle).expanduser().resolve()
    _reject_incomplete(bundle_path)
    actual_files = _safe_bundle_files(bundle_path)
    unexpected = sorted(actual_files - set(INFERENCE_FILENAMES))
    if unexpected:
        raise ValueError(f"bundle contains unexpected files: {unexpected}")
    missing = sorted(set(INFERENCE_FILENAMES) - actual_files)
    if missing:
        if missing == [MODEL_FILENAME]:
            raise ValueError("sidecar artifact is not publishable without model.safetensors")
        raise ValueError(f"bundle is missing inference files: {missing}")
    manifest, manifest_contract = _load_manifest(bundle_path)
    if manifest_contract != expected:
        raise ValueError("manifest contract does not match expected contract")
    report = validate_checkpoint(bundle_path, expected=expected, require_weight=True)
    report.require_valid()
    payload_names = actual_files - {MANIFEST_FILENAME}
    _verify_manifest_files(bundle_path, manifest, payload_names=payload_names)

    expected_weight_sha = str(manifest.get("source_weight_sha256") or "")
    if expected_weight_sha != _sha256(bundle_path / MODEL_FILENAME):
        raise ValueError("bundle model.safetensors does not match source_weight_sha256")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    parent_sha = _remote_repo_sha(api, repo_id, revision)
    before_sha: str | None = None
    if sidecars_only:
        before_sha, _ = _remote_weight_info(api, repo_id, parent_sha)
        if before_sha != expected_weight_sha:
            raise ValueError(
                "remote model.safetensors SHA-256 does not match the validated source weight "
                f"({before_sha} != {expected_weight_sha})"
            )
        names = (*SIDECAR_FILENAMES, MANIFEST_FILENAME)
    else:
        names = INFERENCE_FILENAMES

    from huggingface_hub import CommitOperationAdd

    # Upload from an immutable private snapshot. Any mutation of the user-facing
    # bundle after validation cannot change the bytes sent to the Hub.
    with tempfile.TemporaryDirectory(prefix="smolvla-publish-", dir=bundle_path.parent) as temp:
        upload_snapshot = Path(temp)
        for name in names:
            _snapshot_regular_child(bundle_path / name, upload_snapshot / name)
        snapshot_manifest, snapshot_contract = _load_manifest(upload_snapshot)
        if snapshot_contract != expected or snapshot_manifest != manifest:
            raise ValueError("bundle manifest changed while preparing publication")
        snapshot_report = validate_checkpoint(
            upload_snapshot,
            expected=expected,
            require_weight=not sidecars_only,
        )
        snapshot_report.require_valid()
        snapshot_files = snapshot_manifest["files"]
        for name in names:
            if name == MANIFEST_FILENAME:
                continue
            metadata = snapshot_files[name]
            if (
                _attribute(metadata, "sha256") != _sha256(upload_snapshot / name)
                or _attribute(metadata, "size") != (upload_snapshot / name).stat().st_size
            ):
                raise ValueError(f"bundle file changed while preparing publication: {name}")
        operations = [
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(upload_snapshot / name))
            for name in names
        ]
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message=commit_message,
            revision=revision,
            parent_commit=parent_sha,
        )
    commit_sha = str(_attribute(commit, "oid", ""))
    if not _COMMIT_SHA_RE.fullmatch(commit_sha):
        raise RuntimeError(f"Hub publication returned an invalid commit SHA: {commit_sha!r}")
    after_sha, _ = (
        _remote_weight_info(api, repo_id, commit_sha) if sidecars_only else (expected_weight_sha, None)
    )
    if sidecars_only and after_sha != before_sha:
        raise RuntimeError(
            f"remote model.safetensors changed during publication ({before_sha} -> {after_sha})"
        )
    return {
        "ok": True,
        "repo_id": repo_id,
        "commit_sha": commit_sha,
        "commit_url": _attribute(commit, "commit_url"),
        "sidecars_only": sidecars_only,
        "weight_sha256_before": before_sha,
        "weight_sha256_after": after_sha,
        "uploaded_files": list(names),
    }


def _default_snapshot_resolver(repo_id: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[MODEL_FILENAME, *SIDECAR_FILENAMES],
        )
    )


def _load_legacy_metadata_stats(snapshot: Path) -> _MetadataStats:
    """Rebuild v3-style global stats from v2.1 per-episode metadata only."""

    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import cast_stats_to_numpy

    info_path = snapshot / "meta/info.json"
    episodes_stats_path = snapshot / "meta/episodes_stats.jsonl"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid legacy dataset info metadata: {exc}") from exc
    if not isinstance(info, Mapping):
        raise ValueError("invalid legacy dataset info metadata: root must be a mapping")
    features = info.get("features")
    total_frames = info.get("total_frames")
    if not isinstance(features, Mapping) or not isinstance(total_frames, int):
        raise ValueError("legacy meta/info.json must contain features and integer total_frames")

    episodes: list[tuple[int, Mapping[str, Any]]] = []
    try:
        with episodes_stats_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise ValueError(f"line {line_number} root must be a mapping")
                episode_index = record.get("episode_index")
                episode_stats = record.get("stats")
                if not isinstance(episode_index, int) or not isinstance(episode_stats, Mapping):
                    raise ValueError(
                        f"line {line_number} must contain integer episode_index and stats mapping"
                    )
                episodes.append((episode_index, episode_stats))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid legacy episode stats metadata: {exc}") from exc
    if not episodes:
        raise ValueError("legacy meta/episodes_stats.jsonl contains no episode statistics")
    if len({index for index, _ in episodes}) != len(episodes):
        raise ValueError("legacy meta/episodes_stats.jsonl contains duplicate episode indices")

    # This is deliberately identical to the statistics portion of
    # convert_dataset_v21_to_v30: cast every episode, sort by episode index,
    # then aggregate the episode dictionaries.
    per_episode = [cast_stats_to_numpy(stats) for _, stats in sorted(episodes)]
    return _MetadataStats(
        total_frames=total_frames,
        features=features,
        stats=aggregate_stats(per_episode),
        metadata_source="legacy_v2.1_episode_stats",
        legacy_conversion_proof="cast_stats_to_numpy(per episode) then aggregate_stats",
    )


def _default_metadata_loader(
    repo_id: str,
    revision: str,
    *,
    metadata_class: type[Any] | None = None,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> Any:
    """Load v3 metadata, falling back only for an explicit v2.1 compatibility error."""

    from lerobot.datasets.utils import BackwardCompatibilityError

    if metadata_class is None:
        from lerobot.datasets import LeRobotDatasetMetadata

        metadata_class = LeRobotDatasetMetadata
    try:
        metadata = metadata_class(repo_id, revision=revision, force_cache_sync=True)
    except BackwardCompatibilityError:
        if snapshot_download_fn is None:
            from huggingface_hub import snapshot_download

            snapshot_download_fn = snapshot_download
        snapshot = Path(
            snapshot_download_fn(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                allow_patterns=list(_LEGACY_METADATA_ALLOW_PATTERNS),
            )
        )
        return _load_legacy_metadata_stats(snapshot)
    return _MetadataStats(
        total_frames=int(metadata.total_frames),
        features=metadata.features,
        stats=metadata.stats,
        metadata_source="lerobot_v3_metadata",
    )


def _metadata_stats(
    datasets: Sequence[Mapping[str, Any]],
    *,
    metadata_loader: Callable[[str, str], Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    # These imports are deliberately delayed so validation/bundle/publish stay lightweight.
    from lerobot.datasets import aggregate_stats
    from lerobot.policies.smolvla_jax.data import (
        canonicalize_dataset_stats,
        ensure_stats_counts,
        rename_dataset_stats,
        resolve_action_key,
    )

    if metadata_loader is None:
        metadata_loader = _default_metadata_loader

    stats_list: list[dict[str, dict[str, Any]]] = []
    provenance: list[dict[str, Any]] = []
    for source in datasets:
        repo_id = str(source["repo_id"])
        revision = str(source["revision"])
        metadata = metadata_loader(repo_id, revision)
        frame_count = int(metadata.total_frames)
        action_key = resolve_action_key(metadata.features, source.get("action_key"))
        stats = rename_dataset_stats(
            canonicalize_dataset_stats(metadata.stats, action_key),
            source.get("rename_map"),
        )
        stats_list.append(ensure_stats_counts(stats, frame_count=frame_count))
        dataset_provenance = {
            "repo_id": repo_id,
            "revision": revision,
            "revision_proof": source["revision_proof"],
            "frames": frame_count,
            "action_key": action_key,
        }
        metadata_source = _attribute(metadata, "metadata_source")
        if metadata_source is not None:
            dataset_provenance["metadata_source"] = str(metadata_source)
        legacy_conversion_proof = _attribute(metadata, "legacy_conversion_proof")
        if legacy_conversion_proof is not None:
            dataset_provenance["legacy_conversion_proof"] = str(legacy_conversion_proof)
        provenance.append(dataset_provenance)
    merged = aggregate_stats(stats_list) if len(stats_list) > 1 else stats_list[0]
    return merged, provenance


def repair_sidecars(
    *,
    repo_id: str,
    training_config: str | Path,
    output: str | Path,
    revision: str | None = None,
    expected_weight_sha256: str | None = None,
    api: Any | None = None,
    snapshot_resolver: Callable[[str, str | None], Path] | None = None,
    metadata_loader: Callable[[str, str], Any] | None = None,
) -> Path:
    """Rebuild sidecars from immutable metadata without decoding dataset frames."""

    training_path = Path(training_config).expanduser().resolve()
    raw = _load_yaml(training_path)
    datasets = raw.get("datasets")
    model_overrides = raw.get("model")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("training config must contain a non-empty datasets list")
    if not isinstance(model_overrides, dict):
        raise ValueError("training config must contain a model mapping")
    expected = contract_from_training_yaml(training_path)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    remote_sha, weight_time = _remote_weight_info(api, repo_id, revision)
    if expected_weight_sha256 is not None and remote_sha != expected_weight_sha256:
        raise ValueError(
            f"remote model.safetensors SHA-256 mismatch ({remote_sha} != {expected_weight_sha256})"
        )
    resolved_datasets = resolve_dataset_revisions(
        datasets,
        model_weight_uploaded_at=weight_time,
        api=api,
    )
    stats, dataset_provenance = _metadata_stats(
        resolved_datasets,
        metadata_loader=metadata_loader,
    )

    resolver = snapshot_resolver or _default_snapshot_resolver
    snapshot = Path(resolver(repo_id, revision)).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"repair output already exists: {output_path}")
    staging = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.incomplete")
    staging.mkdir(parents=True)
    for filename in SIDECAR_FILENAMES:
        source_file = snapshot / filename
        if not source_file.is_file():
            raise FileNotFoundError(f"remote snapshot is missing {filename}")
        shutil.copy2(source_file, staging / filename)
    source_weight = snapshot / MODEL_FILENAME
    if _sha256(source_weight) != remote_sha:
        raise ValueError("downloaded model.safetensors does not match remote LFS SHA-256")
    _copy_payload(source_weight, staging / MODEL_FILENAME)

    from lerobot.policies.smolvla_jax.checkpoint import write_effective_config
    from lerobot.policies.smolvla_jax.configuration import JaxSmolVLAConfig
    from lerobot.policies.smolvla_jax.preprocessing import JaxSmolVLAPreprocessor

    effective_config = JaxSmolVLAConfig.from_pretrained(staging).with_overrides(model_overrides)
    write_effective_config(staging, effective_config)
    preprocessor = JaxSmolVLAPreprocessor.__new__(JaxSmolVLAPreprocessor)
    preprocessor.checkpoint = staging
    preprocessor.config = effective_config
    preprocessor.rename_map = {}
    preprocessor.stats = preprocessor._flatten_stats(stats)
    preprocessor.post_stats = dict(preprocessor.stats)
    preprocessor.save_normalization_assets(staging)
    _write_json(
        staging / MANIFEST_FILENAME,
        _manifest(
            staging,
            source=f"hf://{repo_id}@{revision or 'HEAD'}",
            source_weight_sha256=remote_sha,
            expected=expected,
            dataset_revisions=dataset_provenance,
        ),
    )
    report = validate_checkpoint(staging, expected=expected, require_weight=True)
    report.require_valid()
    staging.replace(output_path)
    return output_path


def _add_training_contract(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--training-config", type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a local checkpoint or bundle")
    validate.add_argument("path", type=Path)
    _add_training_contract(validate)

    bundle = commands.add_parser("bundle", help="build a validated inference-only bundle")
    bundle.add_argument("source", type=Path)
    bundle.add_argument("destination", type=Path)
    bundle.add_argument("--training-config", type=Path, required=True)
    bundle.add_argument(
        "--without-model",
        action="store_true",
        help="build a validation-only sidecar artifact; this output cannot be published",
    )

    repair = commands.add_parser("repair-sidecars", help="reconstruct sidecars from training metadata")
    repair.add_argument("--repo-id", required=True)
    repair.add_argument("--training-config", type=Path, required=True)
    repair.add_argument("--output", type=Path, required=True)
    repair.add_argument("--revision")
    repair.add_argument("--expected-weight-sha256")

    publish = commands.add_parser("publish", help="publish an already validated bundle")
    publish.add_argument("--bundle", type=Path, required=True)
    publish.add_argument("--repo-id", required=True)
    publish.add_argument("--training-config", type=Path, required=True)
    publish.add_argument("--revision")
    publish.add_argument("--sidecars-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            expected = contract_from_training_yaml(args.training_config) if args.training_config else None
            report = validate_checkpoint(args.path, expected=expected)
            result = _report_dict(report)
            code = 0 if report.ok else 1
        elif args.command == "bundle":
            expected = contract_from_training_yaml(args.training_config)
            output = build_inference_bundle(
                args.source,
                args.destination,
                expected=expected,
                include_model=not args.without_model,
            )
            report = validate_checkpoint(output, expected=expected, require_weight=not args.without_model)
            result = _report_dict(report)
            code = 0
        elif args.command == "repair-sidecars":
            output = repair_sidecars(
                repo_id=args.repo_id,
                training_config=args.training_config,
                output=args.output,
                revision=args.revision,
                expected_weight_sha256=args.expected_weight_sha256,
            )
            manifest, expected = _load_manifest(output)
            report = validate_checkpoint(output, expected=expected)
            result = {**_report_dict(report), "manifest": manifest}
            code = 0
        else:
            expected = contract_from_training_yaml(args.training_config)
            result = publish_bundle(
                args.bundle,
                repo_id=args.repo_id,
                expected=expected,
                revision=args.revision,
                sidecars_only=args.sidecars_only,
            )
            code = 0
    except Exception as exc:  # CLI boundary: errors must be machine-readable and nonzero.
        result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        code = 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
