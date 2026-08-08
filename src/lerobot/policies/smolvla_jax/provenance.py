from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


LOCAL_DATASET_IDENTITY_VERSION = 1
TACTILE_ENCODER_PROVENANCE_VERSION = 1
TACTILE_ENCODER_PROVENANCE_FILENAME = "encoder_provenance.json"

_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(dict(payload), file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def local_dataset_content_identity(dataset_root: str | Path) -> dict[str, Any]:
    """Return a deterministic local-v3 identity without reading large payloads.

    Metadata bytes under ``meta/`` are hashed. Other files are represented by a
    canonical relative-path/size inventory, so copying a dataset preserves its
    identity while payload additions, removals, renames, and size changes do not.
    This intentionally does not claim to detect same-size payload byte changes.
    """

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    metadata_records: list[dict[str, Any]] = []
    data_records: list[dict[str, Any]] = []
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        record: dict[str, Any] = {"path": relative, "size": int(path.stat().st_size)}
        if PurePosixPath(relative).parts[:1] == ("meta",):
            record["sha256"] = sha256_file(path)
            metadata_records.append(record)
        else:
            data_records.append(record)

    metadata = {
        "file_count": len(metadata_records),
        "total_size": sum(record["size"] for record in metadata_records),
        "sha256": _canonical_sha256(metadata_records),
    }
    data_inventory = {
        "file_count": len(data_records),
        "total_size": sum(record["size"] for record in data_records),
        "sha256": _canonical_sha256(data_records),
    }
    identity = {
        "version": LOCAL_DATASET_IDENTITY_VERSION,
        "algorithm": "lerobot-v3-metadata-sha256-data-inventory-v1",
        "metadata": metadata,
        "data_inventory": data_inventory,
    }
    return {**identity, "sha256": _canonical_sha256(identity)}


def local_dataset_identity(dataset_root: str | Path) -> dict[str, Any]:
    return {
        "kind": "local_v3",
        "content_identity": local_dataset_content_identity(dataset_root),
    }


def validate_local_dataset_content_identity_record(identity: Mapping[str, Any]) -> None:
    if identity.get("version") != LOCAL_DATASET_IDENTITY_VERSION:
        raise ValueError("local dataset content identity version is invalid")
    if identity.get("algorithm") != "lerobot-v3-metadata-sha256-data-inventory-v1":
        raise ValueError("local dataset content identity algorithm is invalid")
    for section_name in ("metadata", "data_inventory"):
        section = identity.get(section_name)
        if not isinstance(section, Mapping):
            raise ValueError(f"local dataset content identity {section_name} is invalid")
        if (
            isinstance(section.get("file_count"), bool)
            or not isinstance(section.get("file_count"), int)
            or int(section["file_count"]) < 0
            or isinstance(section.get("total_size"), bool)
            or not isinstance(section.get("total_size"), int)
            or int(section["total_size"]) < 0
            or not isinstance(section.get("sha256"), str)
            or not _SHA256_RE.fullmatch(str(section["sha256"]))
        ):
            raise ValueError(f"local dataset content identity {section_name} is invalid")
    canonical = {
        "version": identity["version"],
        "algorithm": identity["algorithm"],
        "metadata": dict(identity["metadata"]),
        "data_inventory": dict(identity["data_inventory"]),
    }
    if identity.get("sha256") != _canonical_sha256(canonical):
        raise ValueError("local dataset content identity canonical digest mismatch")


def hub_snapshot_sha(snapshot_root: str | Path) -> str:
    """Extract and validate the immutable commit from a Hub snapshot path."""

    path = Path(snapshot_root).expanduser()
    parts = path.parts
    candidates = [
        parts[index + 1]
        for index, part in enumerate(parts[:-1])
        if part == "snapshots"
    ]
    if _COMMIT_SHA_RE.fullmatch(path.name):
        candidates.append(path.name)
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1 or not _COMMIT_SHA_RE.fullmatch(candidates[0]):
        raise ValueError(
            "Hub snapshot download did not return a path containing an immutable 40-hex commit SHA: "
            f"{path}"
        )
    return candidates[0]


def _checkpoint_file_records(checkpoint_dir: str | Path) -> list[dict[str, Any]]:
    directory = Path(checkpoint_dir).expanduser().resolve()
    checkpoint_path = directory / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"tactile encoder checkpoint metadata is missing: {checkpoint_path}")
    try:
        metadata = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid tactile encoder checkpoint metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("tactile encoder checkpoint metadata must be a mapping")

    params_name = metadata.get("params_file", "params.npz")
    if not isinstance(params_name, str) or not params_name or "\\" in params_name:
        raise ValueError("tactile encoder params_file must be a safe relative path")
    params_relative = PurePosixPath(params_name)
    if params_relative.is_absolute() or ".." in params_relative.parts:
        raise ValueError("tactile encoder params_file must be a safe relative path")

    paths = (checkpoint_path, directory.joinpath(*params_relative.parts))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"tactile encoder files are missing: {missing}")
    records = [
        {
            "path": path.relative_to(directory).as_posix(),
            "size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return sorted(records, key=lambda record: record["path"])


def tactile_encoder_checkpoint_identity(checkpoint_dir: str | Path) -> dict[str, Any]:
    files = _checkpoint_file_records(checkpoint_dir)
    manifest = {
        "algorithm": "tactile-encoder-checkpoint-files-v1",
        "files": files,
    }
    return {**manifest, "sha256": _canonical_sha256(manifest)}


def write_tactile_encoder_provenance(
    checkpoint_dir: str | Path,
    *,
    repo_id: str,
    requested_revision: str,
    resolved_revision: str,
) -> dict[str, Any]:
    directory = Path(checkpoint_dir).expanduser().resolve()
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("tactile encoder repo_id must be non-empty")
    if not isinstance(requested_revision, str) or not requested_revision.strip():
        raise ValueError("tactile encoder requested_revision must be non-empty")
    resolved_revision = str(resolved_revision).lower()
    if not _COMMIT_SHA_RE.fullmatch(resolved_revision):
        raise ValueError("tactile encoder resolved_revision must be an immutable 40-hex commit SHA")

    checkpoint_identity = tactile_encoder_checkpoint_identity(directory)
    provenance = {
        "version": TACTILE_ENCODER_PROVENANCE_VERSION,
        "repo_id": repo_id.strip(),
        "requested_revision": requested_revision.strip(),
        "resolved_revision": resolved_revision,
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "checkpoint_files": checkpoint_identity["files"],
        "checkpoint_digest_algorithm": checkpoint_identity["algorithm"],
    }
    _atomic_write_json(directory / TACTILE_ENCODER_PROVENANCE_FILENAME, provenance)
    return provenance


def validate_tactile_encoder_provenance(
    checkpoint_dir: str | Path,
    *,
    expected_repo_id: str | None = None,
) -> dict[str, Any]:
    directory = Path(checkpoint_dir).expanduser().resolve()
    provenance_path = directory / TACTILE_ENCODER_PROVENANCE_FILENAME
    provenance = validate_tactile_encoder_provenance_record(
        provenance_path,
        expected_repo_id=expected_repo_id,
    )

    actual = tactile_encoder_checkpoint_identity(directory)
    if provenance.get("checkpoint_digest_algorithm") != actual["algorithm"]:
        raise ValueError("tactile encoder provenance digest algorithm mismatch")
    if provenance.get("checkpoint_files") != actual["files"]:
        raise ValueError("tactile encoder provenance file digest mismatch")
    if provenance["checkpoint_sha256"] != actual["sha256"]:
        raise ValueError("tactile encoder provenance checkpoint sha256 mismatch")
    return provenance


def validate_tactile_encoder_provenance_record(
    provenance_path: str | Path,
    *,
    expected_repo_id: str | None = None,
) -> dict[str, Any]:
    """Validate a persisted provenance record without requiring encoder payload files."""

    provenance_path = Path(provenance_path).expanduser().resolve()
    if not provenance_path.is_file():
        raise FileNotFoundError(f"tactile encoder provenance is missing: {provenance_path}")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid tactile encoder provenance: {exc}") from exc
    if not isinstance(provenance, dict):
        raise ValueError("tactile encoder provenance must be a mapping")
    if provenance.get("version") != TACTILE_ENCODER_PROVENANCE_VERSION:
        raise ValueError("unsupported tactile encoder provenance version")

    repo_id = provenance.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("tactile encoder provenance repo_id is invalid")
    if expected_repo_id is not None and repo_id != expected_repo_id:
        raise ValueError(
            f"tactile encoder provenance repo mismatch: {repo_id!r} != {expected_repo_id!r}"
        )
    requested_revision = provenance.get("requested_revision")
    if not isinstance(requested_revision, str) or not requested_revision:
        raise ValueError("tactile encoder provenance requested_revision is invalid")
    resolved_revision = provenance.get("resolved_revision")
    if not isinstance(resolved_revision, str) or not _COMMIT_SHA_RE.fullmatch(resolved_revision):
        raise ValueError("tactile encoder provenance resolved_revision is not immutable")
    checkpoint_sha256 = provenance.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or not _SHA256_RE.fullmatch(checkpoint_sha256):
        raise ValueError("tactile encoder provenance checkpoint_sha256 is invalid")

    files = provenance.get("checkpoint_files")
    if not isinstance(files, list) or not files:
        raise ValueError("tactile encoder provenance checkpoint_files are invalid")
    for record in files:
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            raise ValueError("tactile encoder provenance checkpoint file record is invalid")
        if (
            not isinstance(record["path"], str)
            or not record["path"]
            or isinstance(record["size"], bool)
            or not isinstance(record["size"], int)
            or record["size"] < 0
            or not isinstance(record["sha256"], str)
            or not _SHA256_RE.fullmatch(record["sha256"])
        ):
            raise ValueError("tactile encoder provenance checkpoint file record is invalid")
    if provenance.get("checkpoint_digest_algorithm") != "tactile-encoder-checkpoint-files-v1":
        raise ValueError("tactile encoder provenance digest algorithm is invalid")
    canonical_checkpoint = {
        "algorithm": provenance["checkpoint_digest_algorithm"],
        "files": files,
    }
    if checkpoint_sha256 != _canonical_sha256(canonical_checkpoint):
        raise ValueError("tactile encoder provenance canonical checkpoint sha256 mismatch")
    return provenance


def tactile_encoder_experiment_identity(provenance: Mapping[str, Any]) -> dict[str, str]:
    return {
        "repo_id": str(provenance["repo_id"]),
        "resolved_revision": str(provenance["resolved_revision"]),
        "checkpoint_sha256": str(provenance["checkpoint_sha256"]),
    }
