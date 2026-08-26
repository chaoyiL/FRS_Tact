"""Validate and cache conversions of the Flax tactile ResNet18 checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from train_deco.models.tactile_resnet import RESNET18_EMBEDDING_DIM, TactileResNet18


CONVERSION_VERSION = "1"
ARCHITECTURE = "resnet18"
_WEIGHTS_NAME = "encoder.safetensors"
_METADATA_NAME = "encoder.json"


@dataclass(frozen=True)
class ResolvedTactileEncoder:
    weights_path: Path
    metadata_path: Path
    source_sha256: str
    architecture: str
    embedding_dim: int


@dataclass(frozen=True)
class _Source:
    path: Path
    params_path: Path | None
    checkpoint: dict[str, Any] | None
    digest: str
    framework: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_digest(checkpoint_path: Path, params_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (checkpoint_path, params_path):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _resolve_source(source: str | Path) -> _Source:
    path = Path(source).expanduser().resolve()
    if path.is_file() and path.suffix == ".safetensors":
        return _Source(path, None, None, _sha256_file(path), "pytorch")
    if not path.is_dir():
        raise FileNotFoundError(
            "tactile encoder source must be a .safetensors file or a checkpoint directory: "
            f"{path}"
        )
    checkpoint_path = path / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"tactile encoder checkpoint directory is missing checkpoint.json: {path}")
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid tactile encoder checkpoint.json: {checkpoint_path}") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("tactile encoder checkpoint.json must contain a JSON object")
    requested = checkpoint.get("params_file")
    if requested is None:
        candidates = sorted(path.glob("params-*.npz"))
        if len(candidates) != 1:
            raise ValueError(
                "tactile encoder checkpoint directory must contain exactly one params-*.npz "
                "when checkpoint.json has no params_file"
            )
        params_path = candidates[0]
    else:
        if not isinstance(requested, str) or Path(requested).name != requested:
            raise ValueError("checkpoint.json params_file must be a simple params-*.npz filename")
        params_path = path / requested
        if not params_path.is_file():
            raise FileNotFoundError(f"checkpoint.json references missing parameter archive: {params_path}")
    if params_path.suffix != ".npz" or not params_path.name.startswith("params-"):
        raise ValueError(f"tactile encoder params_file must be params-*.npz, got {params_path.name!r}")
    return _Source(path, params_path, checkpoint, _directory_digest(checkpoint_path, params_path), "flax")


def _flax_path_for_torch(name: str) -> str | None:
    if name.endswith("num_batches_tracked"):
        return None
    parts = name.split(".")
    if parts[0].startswith("layer"):
        block = f"block{parts[0][-1]}_{parts[1]}"
        tail = "/".join(parts[2:])
    else:
        block = ""
        tail = "/".join(parts)
    is_stat = tail.endswith(("running_mean", "running_var"))
    if is_stat:
        tail = tail.replace("running_mean", "mean").replace("running_var", "var")
        prefix = "tactile_resnet/batch_stats"
    else:
        if tail.endswith("weight"):
            module_name = name.split(".")[-2]
            is_convolution = module_name.startswith("conv") or module_name == "proj_conv"
            tail = tail[:-len("weight")] + ("kernel" if is_convolution else "scale")
        prefix = "tactile_resnet/params"
    return "/".join(part for part in (prefix, block, tail) if part)


def _flax_shape_for_torch(tensor: torch.Tensor) -> tuple[int, ...]:
    if tensor.ndim == 4:
        return (tensor.shape[2], tensor.shape[3], tensor.shape[1], tensor.shape[0])
    return tuple(tensor.shape)


def _load_flax_state(source: _Source) -> dict[str, torch.Tensor]:
    assert source.params_path is not None and source.checkpoint is not None
    paths = source.checkpoint.get("parameter_paths")
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise ValueError("checkpoint.json parameter_paths must be a list of Flax leaf paths")
    if len(paths) != len(set(paths)):
        raise ValueError("checkpoint.json parameter_paths contains duplicate leaves")
    archive_by_path: dict[str, np.ndarray] = {}
    try:
        with np.load(source.params_path, allow_pickle=False) as archive:
            for index, path in enumerate(paths):
                archive_name = f"p{index:05d}"
                if archive_name not in archive:
                    raise ValueError(f"missing archive leaf {archive_name} for parameter path {path!r}")
                archive_by_path[path] = np.asarray(archive[archive_name])
    except ValueError:
        raise
    except Exception as error:  # corrupt NPZ errors vary by NumPy release
        raise ValueError(f"failed to read tactile encoder parameter archive: {source.params_path}") from error

    reference = TactileResNet18().state_dict()
    expected = {name: _flax_path_for_torch(name) for name in reference}
    expected_paths = {path for path in expected.values() if path is not None}
    present_paths = {path for path in archive_by_path if path.startswith("tactile_resnet/")}
    missing = sorted(expected_paths - present_paths)
    extra = sorted(present_paths - expected_paths)
    if missing:
        raise ValueError(f"missing required tactile ResNet leaves: {missing}")
    if extra:
        raise ValueError(f"unexpected tactile ResNet leaves: {extra}")

    converted = dict(reference)
    for name, reference_tensor in reference.items():
        flax_path = expected[name]
        if flax_path is None:
            continue
        array = archive_by_path[flax_path]
        expected_shape = _flax_shape_for_torch(reference_tensor)
        if tuple(array.shape) != expected_shape:
            raise ValueError(
                f"shape mismatch for {name}: expected Flax {expected_shape}, got {tuple(array.shape)}"
            )
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"tactile encoder leaf {flax_path!r} must contain finite numeric values")
        if reference_tensor.ndim == 4:
            array = np.transpose(array, (3, 2, 0, 1))
        converted[name] = torch.from_numpy(np.ascontiguousarray(array)).to(dtype=reference_tensor.dtype)
    model = TactileResNet18()
    model.load_state_dict(converted, strict=True)
    return converted


def _validate_pytorch_state(state: dict[str, torch.Tensor], *, source: Path) -> dict[str, torch.Tensor]:
    model = TactileResNet18()
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"converted tactile encoder does not strictly match ResNet18: {source}") from error
    for name, tensor in state.items():
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"converted tactile encoder tensor {name!r} is non-finite")
    return state


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _artifact_metadata(source: _Source, state: dict[str, torch.Tensor], weights_path: Path) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "conversion_version": CONVERSION_VERSION,
        "embedding_dim": RESNET18_EMBEDDING_DIM,
        "source_framework": source.framework,
        "source_path": str(source.path),
        "source_sha256": source.digest,
        "target_framework": "pytorch",
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in state.items()},
        "weights_sha256": _sha256_file(weights_path),
    }


def _cached_artifact(directory: Path, source: _Source) -> ResolvedTactileEncoder | None:
    weights_path = directory / _WEIGHTS_NAME
    metadata_path = directory / _METADATA_NAME
    if not weights_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("source_sha256") != source.digest
            or metadata.get("conversion_version") != CONVERSION_VERSION
            or metadata.get("architecture") != ARCHITECTURE
            or metadata.get("embedding_dim") != RESNET18_EMBEDDING_DIM
            or metadata.get("weights_sha256") != _sha256_file(weights_path)
        ):
            return None
        _validate_pytorch_state(load_file(str(weights_path)), source=weights_path)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return None
    return ResolvedTactileEncoder(weights_path, metadata_path, source.digest, ARCHITECTURE, RESNET18_EMBEDDING_DIM)



def _assert_source_unchanged(source: _Source) -> None:
    if _resolve_source(source.path).digest != source.digest:
        raise RuntimeError("tactile encoder source changed during conversion; retry resolution")

def resolve_tactile_encoder(source: str | Path, cache_root: str | Path) -> ResolvedTactileEncoder:
    """Resolve a validated, content-addressed PyTorch tactile encoder artifact."""

    resolved_source = _resolve_source(source)
    directory = Path(cache_root).expanduser().resolve() / f"v{CONVERSION_VERSION}" / resolved_source.digest
    cached = _cached_artifact(directory, resolved_source)
    if cached is not None:
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    weights_path = directory / _WEIGHTS_NAME
    metadata_path = directory / _METADATA_NAME
    if resolved_source.framework == "flax":
        state = _load_flax_state(resolved_source)
        temporary = directory / f".{_WEIGHTS_NAME}.{uuid.uuid4().hex}.tmp"
        save_file(state, str(temporary))
    else:
        state = _validate_pytorch_state(load_file(str(resolved_source.path)), source=resolved_source.path)
        temporary = directory / f".{_WEIGHTS_NAME}.{uuid.uuid4().hex}.tmp"
        shutil.copyfile(resolved_source.path, temporary)
    _validate_pytorch_state(load_file(str(temporary)), source=temporary)
    _assert_source_unchanged(resolved_source)
    os.replace(temporary, weights_path)
    _atomic_json(metadata_path, _artifact_metadata(resolved_source, state, weights_path))
    return ResolvedTactileEncoder(weights_path, metadata_path, resolved_source.digest, ARCHITECTURE, RESNET18_EMBEDDING_DIM)


def load_tactile_encoder_weights(module: torch.nn.Module, artifact: ResolvedTactileEncoder) -> None:
    """Strictly load a resolved tactile artifact into the supplied encoder module."""

    try:
        state = load_file(str(artifact.weights_path))
        module.load_state_dict(state, strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"failed to strictly load tactile encoder artifact: {artifact.weights_path}") from error


def import_jax_flax_for_cpu() -> tuple[Any, Any]:
    """Lazily import conversion-only JAX/Flax with CPU selected before import."""

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    import jax
    import flax

    jax.config.update("jax_platform_name", "cpu")
    return jax, flax
