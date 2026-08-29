"""Stable artifact identities for v2 pick-tube training and deployment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf


SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of *path* without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_digest(value: Any) -> str:
    """Hash a JSON value using a deterministic, whitespace-free encoding."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"artifact manifest field {key!r} must be a non-empty string")
    return result


def _require_digest(value: Mapping[str, Any], key: str) -> str:
    result = _require_text(value, key)
    if len(result) != 64:
        raise ValueError(f"artifact manifest field {key!r} must be a SHA256 digest")
    return result


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: int
    dataset_digest: str
    split_digest: str
    action_representation_version: int
    action_contract: str
    normalizer_version: str
    normalizer_sha256: str
    pca_sha256: str
    tactile_cache_sha256: str
    git_commit: str
    at_sha256: str | None = None
    latent_target_mode: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, role: str) -> "ArtifactManifest":
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if not isinstance(value, Mapping):
            raise ValueError("artifact manifest must be a mapping")
        role = str(role).upper()
        if role not in {"AT", "LDP"}:
            raise ValueError(f"unknown artifact role: {role}")
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"artifact manifest schema_version must be {SCHEMA_VERSION}")
        action_version = value.get("action_representation_version")
        if not isinstance(action_version, int) or isinstance(action_version, bool):
            raise ValueError("artifact manifest action_representation_version must be an integer")

        kwargs = {
            "schema_version": SCHEMA_VERSION,
            "dataset_digest": _require_digest(value, "dataset_digest"),
            "split_digest": _require_digest(value, "split_digest"),
            "action_representation_version": action_version,
            "action_contract": _require_text(value, "action_contract"),
            "normalizer_version": _require_text(value, "normalizer_version"),
            "normalizer_sha256": _require_digest(value, "normalizer_sha256"),
            "pca_sha256": _require_digest(value, "pca_sha256"),
            "tactile_cache_sha256": _require_digest(value, "tactile_cache_sha256"),
            "git_commit": _require_text(value, "git_commit"),
            "at_sha256": value.get("at_sha256"),
            "latent_target_mode": value.get("latent_target_mode"),
        }
        if role == "LDP":
            kwargs["at_sha256"] = _require_digest(value, "at_sha256")
            kwargs["latent_target_mode"] = _require_text(value, "latent_target_mode")
        elif kwargs["at_sha256"] is not None or kwargs["latent_target_mode"] is not None:
            raise ValueError("AT artifact manifest cannot contain LDP-only fields")
        return cls(**kwargs)

    @classmethod
    def from_cache_signature(
        cls,
        signature: Mapping[str, Any],
        *,
        normalizer_sha256: str,
        role: str,
    ) -> "ArtifactManifest":
        value = {
            "schema_version": signature["schema_version"],
            "dataset_digest": signature["dataset"]["digest"],
            "split_digest": signature["split"]["digest"],
            "action_representation_version": signature["action"]["representation_version"],
            "action_contract": signature["action"]["contract"],
            "normalizer_version": signature["normalizer"]["version"],
            "normalizer_sha256": normalizer_sha256,
            "pca_sha256": signature["pca"]["sha256"],
            "tactile_cache_sha256": signature["tactile"]["sha256"],
            "git_commit": signature["git_commit"],
            "at_sha256": signature.get("at_sha256"),
            "latent_target_mode": signature.get("latent_target_mode"),
        }
        return cls.from_dict(value, role=role)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return {key: value for key, value in result.items() if value is not None}


def _dataset_artifact_manifest(dataset: Any) -> dict[str, Any]:
    value = getattr(dataset, "artifact_manifest", None)
    if value is None:
        replay_buffer = getattr(dataset, "replay_buffer", None)
        root = getattr(replay_buffer, "root", None)
        try:
            value = root["meta"].attrs.get("v2_manifest_json")
        except (AttributeError, KeyError, TypeError):
            value = None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("dataset v2 artifact manifest is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("dataset is missing v2 artifact manifest metadata")
    value = dict(value)
    required = (
        "dataset_digest",
        "action_representation_version",
        "action_contract",
        "normalizer_version",
        "pca_sha256",
        "tactile_cache_sha256",
        "converter_git_commit",
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"dataset v2 artifact manifest is missing fields: {missing}")
    for key in ("dataset_digest", "pca_sha256", "tactile_cache_sha256"):
        _require_digest(value, key)
    return value


def _select(cfg: Any, key: str, default: Any = None) -> Any:
    if OmegaConf.is_config(cfg):
        return OmegaConf.select(cfg, key, default=default)
    current = cfg
    for part in key.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return default
    return current


def _git_commit(cfg: Any) -> str:
    configured = _select(cfg, "artifact_git_commit")
    if configured:
        return str(configured)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def build_normalizer_cache_signature(cfg: Any, dataset: Any, at_path: Path | None) -> dict:
    """Build the complete canonical identity governing normalizer reuse."""
    manifest = _dataset_artifact_manifest(dataset)
    configured_normalizer = _select(cfg, "task.dataset.action_normalizer_version")
    if configured_normalizer and configured_normalizer != manifest["normalizer_version"]:
        raise ValueError(
            "configured action normalizer version does not match dataset manifest: "
            f"{configured_normalizer!r} != {manifest['normalizer_version']!r}"
        )
    train_mask = np.asarray(getattr(dataset, "train_mask", []), dtype=bool).tolist()
    split = {
        "train_mask": train_mask,
        "seed": _select(cfg, "task.dataset.seed"),
        "val_ratio": _select(cfg, "task.dataset.val_ratio"),
        "max_train_episodes": _select(cfg, "task.dataset.max_train_episodes"),
    }
    temporal_keys = (
        "horizon",
        "n_obs_steps",
        "dataset_obs_steps",
        "dataset_obs_temporal_downsample_ratio",
        "n_latency_steps",
        "n_action_steps",
    )
    at_path = Path(at_path) if at_path is not None else None
    latent_target_mode = None
    if at_path is not None:
        latent_target_mode = (
            "posterior_mode_pre_vq"
            if bool(_select(cfg, "use_latent_action_before_vq", False))
            else "posterior_mode_post_vq"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "digest": manifest["dataset_digest"],
            "converter_git_commit": str(manifest["converter_git_commit"]),
        },
        "split": {"digest": stable_json_digest(split), **split},
        "action": {
            "representation_version": int(manifest["action_representation_version"]),
            "contract": str(manifest["action_contract"]),
        },
        "normalizer": {"version": str(manifest["normalizer_version"])},
        "pca": {"sha256": str(manifest["pca_sha256"])},
        "tactile": {"sha256": str(manifest["tactile_cache_sha256"])},
        "temporal": {key: _select(cfg, key) for key in temporal_keys},
        "at_sha256": sha256_file(at_path) if at_path is not None else None,
        "latent_target_mode": latent_target_mode,
        "git_commit": _git_commit(cfg),
    }


def normalizer_identity_digest(normalizer: Any) -> str:
    """Hash actual shared action-normalizer parameters, excluding latent state."""
    try:
        action_state = normalizer["action"].state_dict()
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("normalizer is missing action parameters") from error
    if not action_state:
        raise ValueError("normalizer action parameters must not be empty")
    digest = hashlib.sha256()
    for key in sorted(action_state):
        value = action_state[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"normalizer action parameter {key!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        array = tensor.numpy()
        metadata = {
            "key": key,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _metadata_path(normalizer_path: Path) -> Path:
    return Path(normalizer_path).with_name("normalizer.meta.json")


def load_normalizer_cache(normalizer_path: Path, signature: Mapping[str, Any]) -> Any | None:
    """Return a cached normalizer only when its full metadata matches exactly."""
    normalizer_path = Path(normalizer_path)
    metadata_path = _metadata_path(normalizer_path)
    if not normalizer_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, Mapping) or metadata.get("signature") != signature:
        return None
    expected_sha256 = metadata.get("normalizer_sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        return None
    try:
        actual_sha256 = sha256_file(normalizer_path)
    except OSError:
        return None
    if actual_sha256 != expected_sha256:
        return None
    expected_action_sha256 = metadata.get("action_normalizer_sha256")
    if not isinstance(expected_action_sha256, str) or len(expected_action_sha256) != 64:
        return None
    try:
        with normalizer_path.open("rb") as file:
            normalizer = pickle.load(file)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    try:
        actual_action_sha256 = normalizer_identity_digest(normalizer)
    except ValueError:
        return None
    if actual_action_sha256 != expected_action_sha256:
        return None
    return normalizer


def _write_temporary(path: Path, write) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, write[0], **write[1]) as file:
            write[2](file)
            file.flush()
            os.fsync(file.fileno())
        return temporary_path
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_replace(path: Path, write) -> None:
    temporary_path = _write_temporary(path, write)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_normalizer_cache(
    normalizer_path: Path,
    normalizer: Any,
    signature: Mapping[str, Any],
) -> None:
    """Atomically publish the pickle, then its exact-match metadata marker."""
    normalizer_path = Path(normalizer_path)
    temporary_path = _write_temporary(
        normalizer_path,
        ("wb", {}, lambda file: pickle.dump(normalizer, file)),
    )
    try:
        normalizer_sha256 = sha256_file(temporary_path)
        os.replace(temporary_path, normalizer_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    metadata_path = _metadata_path(normalizer_path)
    metadata = {
        "signature": signature,
        "normalizer_sha256": normalizer_sha256,
        "action_normalizer_sha256": normalizer_identity_digest(normalizer),
    }
    _atomic_replace(
        metadata_path,
        (
            "w",
            {"encoding": "utf-8"},
            lambda file: json.dump(
                metadata,
                file,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ),
        ),
    )
