from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp
from huggingface_hub import snapshot_download
from safetensors.flax import load_file as load_safetensors_file, save_file as save_safetensors_file

from .configuration import JaxSmolVLAConfig

Array = jax.Array

ASSET_FILENAMES = (
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "train_config.json",
)


def resolve_checkpoint(
    path_or_repo_id: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Path:
    path = Path(path_or_repo_id).expanduser()
    if path.exists():
        return path.resolve()
    snapshot = snapshot_download(
        repo_id=str(path_or_repo_id),
        revision=revision,
        local_files_only=local_files_only,
        allow_patterns=["*.json", "*.safetensors", "params/**"],
    )
    return Path(snapshot)


def load_safetensors_params(path: str | Path) -> dict[str, Array]:
    """Load a PyTorch SmolVLA safetensors file as JAX arrays without copying layouts."""

    path = Path(path)
    model_file = path / "model.safetensors" if path.is_dir() else path
    if not model_file.is_file():
        raise FileNotFoundError(f"SmolVLA model file not found: {model_file}")
    return dict(load_safetensors_file(model_file))


def parameter_summary(params: Mapping[str, Array]) -> dict[str, Any]:
    dtypes: dict[str, int] = {}
    parameter_count = 0
    for value in params.values():
        parameter_count += int(np.prod(value.shape))
        dtype = str(value.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + int(np.prod(value.shape))
    return {
        "tensor_count": len(params),
        "parameter_count": parameter_count,
        "parameters_by_dtype": dtypes,
        "layout": "pytorch_source_layout",
    }


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_orbax_params(
    params: Mapping[str, Array],
    destination: str | Path,
    *,
    source_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    params_dir = destination / "params"
    handler = ocp.StandardCheckpointHandler()
    checkpointer = ocp.Checkpointer(handler)
    checkpointer.save(params_dir, dict(params), force=overwrite)

    manifest = parameter_summary(params)
    manifest["format_version"] = 1
    manifest["backend"] = "jax"
    if source_dir is not None:
        source_dir = Path(source_dir).expanduser().resolve()
        source_model = source_dir / "model.safetensors"
        if source_model.is_file():
            manifest["source_model"] = str(source_model)
            manifest["source_sha256"] = _sha256(source_model)
        for filename in ASSET_FILENAMES:
            source = source_dir / filename
            if source.is_file():
                shutil.copy2(source, destination / filename)
    with (destination / "conversion_manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    return destination


def save_portable_params(
    params: Mapping[str, Array],
    destination: str | Path,
    *,
    source_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Save a JAX-native, framework-portable safetensors checkpoint."""

    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model_file = destination / "model.safetensors"
    if model_file.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint already exists: {model_file}")
    save_safetensors_file(dict(params), model_file)

    manifest = parameter_summary(params)
    manifest.update({"format_version": 1, "backend": "jax", "storage": "safetensors"})
    if source_dir is not None:
        source_dir = Path(source_dir).expanduser().resolve()
        source_model = source_dir / "model.safetensors"
        if source_model.is_file():
            manifest["source_model"] = str(source_model)
            manifest["source_sha256"] = _sha256(source_model)
        for filename in ASSET_FILENAMES:
            if filename == "model.safetensors":
                continue
            source = source_dir / filename
            if source.is_file():
                shutil.copy2(source, destination / filename)
    manifest["output_sha256"] = _sha256(model_file)
    with (destination / "conversion_manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    return destination


def restore_orbax_params(path: str | Path) -> dict[str, Array]:
    path = Path(path).expanduser().resolve()
    params_dir = path if path.name == "params" else path / "params"
    if not params_dir.is_dir():
        raise FileNotFoundError(f"Orbax params directory not found: {params_dir}")
    restored = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(params_dir)
    return dict(restored)


def load_params(path: str | Path) -> dict[str, Array]:
    path = Path(path).expanduser()
    if path.is_dir() and (path / "params").is_dir():
        return restore_orbax_params(path)
    return load_safetensors_params(path)


def load_config(path: str | Path) -> JaxSmolVLAConfig:
    path = Path(path).expanduser()
    if path.name == "params":
        path = path.parent
    return JaxSmolVLAConfig.from_pretrained(path)
