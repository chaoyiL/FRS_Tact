#!/usr/bin/env python
"""Validate, assemble, repair, and safely publish SmolVLA inference bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
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
_SHA_RE = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)


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


def _copy_or_link(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
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
        _copy_or_link(source_weight, staging / MODEL_FILENAME)
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
            if not _SHA_RE.fullmatch(requested):
                raise ValueError(
                    f"dataset {repo_id!r} revision must be an immutable commit SHA, got {requested!r}"
                )
            info = api.dataset_info(repo_id, revision=requested)
            actual = str(_attribute(info, "sha", ""))
            if actual != requested:
                raise ValueError(
                    f"dataset {repo_id!r} resolved to {actual!r}, expected explicit SHA {requested!r}"
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
                not _SHA_RE.fullmatch(actual)
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
    if sha is None:
        sha = _attribute(value, "blob_id")
    return str(sha) if sha else None


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


def _verify_manifest_files(bundle: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("invalid bundle manifest: files must be a mapping")
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
    api: Any | None = None,
    revision: str | None = None,
    sidecars_only: bool = True,
    commit_message: str = "Repair VT-SmolVLA inference sidecars",
) -> dict[str, Any]:
    """Publish a validated bundle, preserving remote weights in sidecar-only mode."""

    bundle_path = Path(bundle).expanduser().resolve()
    _reject_incomplete(bundle_path)
    manifest, expected = _load_manifest(bundle_path)
    report = validate_checkpoint(bundle_path, expected=expected, require_weight=True)
    report.require_valid()
    actual_files = {path.name for path in bundle_path.iterdir() if path.is_file()}
    unexpected = sorted(actual_files - set(INFERENCE_FILENAMES))
    if unexpected:
        raise ValueError(f"bundle contains unexpected files: {unexpected}")
    missing = sorted(set(INFERENCE_FILENAMES) - actual_files)
    if missing:
        raise ValueError(f"bundle is missing inference files: {missing}")
    _verify_manifest_files(bundle_path, manifest)

    expected_weight_sha = str(manifest.get("source_weight_sha256") or "")
    if expected_weight_sha != _sha256(bundle_path / MODEL_FILENAME):
        raise ValueError("bundle model.safetensors does not match source_weight_sha256")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    before_sha: str | None = None
    if sidecars_only:
        before_sha, _ = _remote_weight_info(api, repo_id, revision)
        if before_sha != expected_weight_sha:
            raise ValueError(
                "remote model.safetensors SHA-256 does not match the validated source weight "
                f"({before_sha} != {expected_weight_sha})"
            )
        names = (*SIDECAR_FILENAMES, MANIFEST_FILENAME)
    else:
        names = INFERENCE_FILENAMES

    from huggingface_hub import CommitOperationAdd

    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(bundle_path / name)) for name in names
    ]
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=commit_message,
        revision=revision,
    )
    after_sha, _ = _remote_weight_info(api, repo_id, None) if sidecars_only else (expected_weight_sha, None)
    if sidecars_only and after_sha != before_sha:
        raise RuntimeError(
            f"remote model.safetensors changed during publication ({before_sha} -> {after_sha})"
        )
    return {
        "ok": True,
        "repo_id": repo_id,
        "commit_sha": _attribute(commit, "oid"),
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


def _metadata_stats(
    datasets: Sequence[Mapping[str, Any]],
    *,
    metadata_loader: Callable[[str, str], Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    # These imports are deliberately delayed so validation/bundle/publish stay lightweight.
    from lerobot.datasets import LeRobotDatasetMetadata, aggregate_stats
    from lerobot.policies.smolvla_jax.data import (
        canonicalize_dataset_stats,
        ensure_stats_counts,
        rename_dataset_stats,
        resolve_action_key,
    )

    if metadata_loader is None:

        def metadata_loader(repo_id: str, revision: str) -> Any:
            return LeRobotDatasetMetadata(repo_id, revision=revision, force_cache_sync=True)

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
        provenance.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "revision_proof": source["revision_proof"],
                "frames": frame_count,
                "action_key": action_key,
            }
        )
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
    _copy_or_link(source_weight, staging / MODEL_FILENAME)

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
    bundle.add_argument("--without-model", action="store_true")

    repair = commands.add_parser("repair-sidecars", help="reconstruct sidecars from training metadata")
    repair.add_argument("--repo-id", required=True)
    repair.add_argument("--training-config", type=Path, required=True)
    repair.add_argument("--output", type=Path, required=True)
    repair.add_argument("--revision")
    repair.add_argument("--expected-weight-sha256")

    publish = commands.add_parser("publish", help="publish an already validated bundle")
    publish.add_argument("--bundle", type=Path, required=True)
    publish.add_argument("--repo-id", required=True)
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
            result = publish_bundle(
                args.bundle,
                repo_id=args.repo_id,
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
