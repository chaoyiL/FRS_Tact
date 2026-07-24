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


def _sync_visual_input_features(
    input_features: dict[str, Any],
    *,
    image_keys: tuple[str, ...],
    resize_height: int,
    resize_width: int,
) -> None:
    """Make VISUAL entries match the image keys used at train/eval time."""

    existing_visual = {
        key: dict(feature)
        for key, feature in input_features.items()
        if isinstance(feature, Mapping) and feature.get("type") == "VISUAL"
    }
    for key in list(input_features):
        feature = input_features.get(key)
        if isinstance(feature, Mapping) and feature.get("type") == "VISUAL" and key not in image_keys:
            del input_features[key]

    default_shape = [3, int(resize_height), int(resize_width)]
    template = next(iter(existing_visual.values()), {"type": "VISUAL", "shape": default_shape})
    for key in image_keys:
        feature = dict(existing_visual.get(key, template))
        feature["type"] = "VISUAL"
        feature.setdefault("shape", default_shape)
        input_features[key] = feature


def _legacy_flags_from_module_modes(module_modes: Mapping[str, Any] | None) -> dict[str, bool] | None:
    """Derive the older boolean training switches from module_modes when present."""

    if module_modes is None:
        return None
    vision = str(module_modes.get("vision", "frozen")).lower()
    connector = str(module_modes.get("connector", "frozen")).lower()
    vlm_text = str(module_modes.get("vlm_text", "frozen")).lower()
    state_proj = str(module_modes.get("state_proj", "full")).lower()
    train_expert_only = connector == "frozen" and vlm_text == "frozen" and vision == "frozen"
    return {
        "freeze_vision_encoder": vision == "frozen",
        "train_expert_only": train_expert_only,
        "train_state_proj": state_proj != "frozen",
    }


def write_effective_config(destination: str | Path, config: JaxSmolVLAConfig) -> Path:
    """Persist training-time overrides in the checkpoint's compatible config.json."""

    destination = Path(destination).expanduser().resolve()
    config_path = destination / "config.json"
    raw: dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open() as file:
            raw = json.load(file)

    raw.update(
        {
            "chunk_size": config.chunk_size,
            "n_action_steps": config.n_action_steps,
            "max_state_dim": config.max_state_dim,
            "max_action_dim": config.max_action_dim,
            "empty_cameras": config.empty_cameras,
            "resize_imgs_with_padding": [config.resize_width, config.resize_height],
            "tokenizer_max_length": config.tokenizer_max_length,
            "pad_language_to": config.pad_language_to,
            "vlm_model_name": config.tokenizer_name,
            "num_steps": config.num_steps,
            "num_vlm_layers": config.num_vlm_layers,
            "num_expert_layers": config.num_expert_layers,
            "self_attn_every_n_layers": config.self_attn_every_n_layers,
            "attention_mode": config.attention_mode,
            "expert_width_multiplier": config.expert_width_multiplier,
            "add_image_special_tokens": config.add_image_special_tokens,
            "module_modes": config.module_modes,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "optimizer_lr": config.optimizer_lr,
            "optimizer_betas": [config.optimizer_beta1, config.optimizer_beta2],
            "optimizer_eps": config.optimizer_eps,
            "optimizer_weight_decay": config.optimizer_weight_decay,
            "optimizer_grad_clip_norm": config.optimizer_grad_clip_norm,
            "scheduler_warmup_steps": config.scheduler_warmup_steps,
            "scheduler_decay_steps": config.scheduler_decay_steps,
            "scheduler_decay_lr": config.scheduler_decay_lr,
            "freeze_vision_encoder": config.freeze_vision_encoder,
            "train_expert_only": config.train_expert_only,
            "train_state_proj": config.train_state_proj,
        }
    )
    legacy = _legacy_flags_from_module_modes(config.module_modes)
    if legacy is not None:
        # Prefer module_modes as source of truth when both are present.
        raw.update(legacy)

    input_features = raw.setdefault("input_features", {})
    input_features.setdefault("observation.state", {"type": "STATE"})["shape"] = [config.state_dim]
    input_features["observation.state"]["type"] = "STATE"
    _sync_visual_input_features(
        input_features,
        image_keys=tuple(config.image_keys),
        resize_height=config.resize_height,
        resize_width=config.resize_width,
    )
    output_features = raw.setdefault("output_features", {})
    output_features.setdefault("action", {"type": "ACTION"})["shape"] = [config.action_dim]
    output_features["action"]["type"] = "ACTION"

    destination.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as file:
        json.dump(raw, file, indent=4)
        file.write("\n")
    return config_path


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
