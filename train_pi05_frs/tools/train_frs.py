#!/usr/bin/env python
"""Validate configuration and train the multi-dataset tactile FRS decoder."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
from typing import Any
import urllib.parse
import zipfile

import yaml

from train_pi05_frs.train import train_decoder


TRAIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAIN_ROOT.parent
DEFAULT_CONFIG = TRAIN_ROOT / "configs" / "train_pi05_frs.yaml"

ROOT_KEYS = {
    "checkpoint",
    "allow_download",
    "datasets",
    "action_cache",
    "tactile_embedding_cache",
    "model",
    "norm_stats",
    "frs_training",
}
DATASET_KEYS = {"repo_id", "root", "revision", "action_key", "rename_map"}
ACTION_CACHE_KEYS = {
    "root",
    "model_sample_steps",
    "reverse_steps",
    "reverse_solver",
    "batch_size",
    "load_workers",
    "flush_every",
    "inference_seed",
    "split_seed",
    "val_fraction",
    "frame_stride",
    "drop_tail_action_chunks",
    "max_episodes",
    "max_samples",
}
TACTILE_CACHE_KEYS = {
    "enabled",
    "root",
    "dtype",
    "precompute_batch_size",
    "precompute_num_workers",
    "precompute_prefetch_factor",
    "precompute_video_backend",
    "precompute_flush_every",
}
MODEL_KEYS = {
    "use_tactile_encoder",
    "tactile_encoder_path",
    "freeze_tactile_encoder",
    "tactile_keys",
    "tactile_embedding_dim",
    "tactile_num_tokens",
    "tactile_image_size",
    "state_conditioning",
    "state_dropout_rate",
    "camera_map",
    "action_dim",
    "action_horizon",
    "paligemma_variant",
    "action_expert_variant",
}
NORM_STATS_KEYS = {"dir", "asset_id", "use_quantile_norm"}
TRAINING_KEYS = {
    "output",
    "tactile_window_divisor",
    "history_stride",
    "loss_mode",
    "gate_tau",
    "gate_temperature",
    "gate_lambda",
    "aux_decode_weight",
    "aux_decode_steps",
    "aux_decode_solver",
    "low_gate_safety_weight",
    "low_gate_safety_margin",
    "rank_low_gate_threshold",
    "rank_high_gate_threshold",
    "rank_weight",
    "rank_margin",
    "repair_weight",
    "repair_margin",
    "best_max_low_gate_unsafe_frac",
    "best_min_high_gate_gain",
    "best_min_high_gate_rank_satisfied_frac",
    "model_dim",
    "depth",
    "num_heads",
    "mlp_ratio",
    "learning_rate",
    "weight_decay",
    "grad_clip_norm",
    "warmup_epochs",
    "lr_reference_dim",
    "min_lr_ratio",
    "lr_schedule",
    "batch_size",
    "epochs",
    "validation_steps",
    "eval_every",
    "seed",
    "write_plots",
    "resume",
    "resume_from",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return value


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{field} must be a mapping")
    return value


def _reject_unknown_keys(
    value: Mapping[str, Any], allowed: set[str], *, prefix: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{prefix}.{unknown[0]} is an unknown configuration key")


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config.{field} must be a non-empty string")
    if any(character in value for character in ("\n", "\r", "\0")):
        raise ValueError(f"config.{field} contains an invalid path/string character")
    return value


def _boolean(section: Mapping[str, Any], key: str, *, prefix: str, default: bool) -> bool:
    value = section.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"config.{prefix}{key} must be a boolean")
    return value


def _integer(
    section: Mapping[str, Any],
    key: str,
    *,
    prefix: str,
    default: int,
    minimum: int = 1,
) -> int:
    value = section.get(key, default)
    if type(value) is not int or value < minimum:
        raise ValueError(f"config.{prefix}{key} must be an integer >= {minimum}")
    return value


def _number(section: Mapping[str, Any], key: str, *, prefix: str, default: float) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"config.{prefix}{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"config.{prefix}{key} must be finite")
    return result


def _string_mapping(value: object, field: str, *, nonempty: bool) -> Mapping[str, str]:
    mapping = _mapping(value, field)
    if nonempty and not mapping:
        raise ValueError(f"config.{field} must not be empty")
    for key, item in mapping.items():
        _nonempty_string(key, f"{field} key")
        _nonempty_string(item, f"{field}.{key}")
    return mapping  # type: ignore[return-value]


def _is_url(value: str) -> bool:
    return bool(urllib.parse.urlparse(value).scheme)


def resolve_local_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve(strict=False)


def resolve_url_or_local_path(value: str) -> str:
    return value if _is_url(value) else str(resolve_local_path(value))


def _local_path(value: object, field: str) -> Path:
    text = _nonempty_string(value, field)
    if _is_url(text):
        raise ValueError(f"config.{field} must be a local filesystem path")
    return resolve_local_path(text)


def _output_target(value: object, field: str) -> Path:
    path = _local_path(value, field)
    if path.resolve(strict=False) in (Path("/"), REPO_ROOT.resolve(), TRAIN_ROOT.resolve()):
        raise ValueError(f"config.{field} must not target a repository or filesystem root")
    return path


def _json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON mapping: {path}")
    return value


def _validate_encoder_checkpoint(
    path: Path, *, expected_embedding_dim: int, expected_image_size: int
) -> None:
    metadata = _json_mapping(path / "checkpoint.json", "tactile encoder checkpoint metadata")
    params_name = _nonempty_string(
        metadata.get("params_file", "params.npz"), "model.tactile_encoder params_file"
    )
    encoder_dir = path.resolve()
    relative_params_path = Path(params_name)
    if relative_params_path.is_absolute():
        raise ValueError(
            "tactile encoder params_file must remain within the checkpoint directory"
        )
    params_path = (encoder_dir / relative_params_path).resolve(strict=False)
    try:
        params_path.relative_to(encoder_dir)
    except ValueError as error:
        raise ValueError(
            "tactile encoder params_file must remain within the checkpoint directory"
        ) from error
    if not params_path.exists():
        raise FileNotFoundError(f"tactile encoder params_file is missing: {params_path}")
    if not params_path.is_file():
        raise ValueError(f"tactile encoder params_file must be a regular file: {params_path}")
    if not zipfile.is_zipfile(params_path):
        raise ValueError(f"tactile encoder params_file is not a valid npz file: {params_path}")
    parameter_paths = metadata.get("parameter_paths")
    if not isinstance(parameter_paths, list) or not parameter_paths:
        raise ValueError("tactile encoder checkpoint parameter_paths must be a non-empty list")
    if not any(str(value).startswith("tactile_resnet/") for value in parameter_paths):
        raise ValueError("tactile encoder checkpoint is missing tactile_resnet parameters")
    clip_config = metadata.get("tactile_clip_config")
    if not isinstance(clip_config, Mapping):
        raise ValueError("tactile encoder checkpoint is missing tactile_clip_config")
    embedding_dim = clip_config.get("embedding_dim")
    if type(embedding_dim) is not int or embedding_dim != expected_embedding_dim:
        raise ValueError(
            "tactile encoder embedding_dim does not match "
            f"config.model.tactile_embedding_dim: {embedding_dim!r} != {expected_embedding_dim}"
        )
    image_size = clip_config.get("tactile_image_size")
    if type(image_size) is not int or image_size != expected_image_size:
        raise ValueError(
            "tactile encoder tactile_image_size does not match "
            f"config.model.tactile_image_size: {image_size!r} != {expected_image_size}"
        )


def _validate_norm_stats(path: Path, *, use_quantile_norm: bool) -> None:
    payload = _json_mapping(path / "norm_stats.json", "norm stats")
    stats = payload.get("norm_stats")
    if not isinstance(stats, Mapping):
        raise ValueError("norm stats JSON must contain a norm_stats mapping")
    for name in ("state", "actions"):
        values = stats.get(name)
        if not isinstance(values, Mapping):
            raise ValueError(f"norm stats are missing {name}")
        required = {"mean", "std"}
        if use_quantile_norm:
            required.update(("q01", "q99"))
        missing = required - set(values)
        if missing:
            raise ValueError(f"norm stats {name} are missing {sorted(missing)}")


def source_cache_dir(cache_root: str | Path, repo_id: str) -> Path:
    parts = [part for part in repo_id.replace("\\", "/").split("/") if part not in ("", ".", "..")]
    if not parts:
        raise ValueError(f"invalid repo id: {repo_id!r}")
    return resolve_local_path(cache_root).joinpath(*parts)


def resolved_dataset_sources(
    datasets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {**source, "root": str(resolve_local_path(str(source["root"])))}
        for source in datasets
    ]


def validate_config(config: Mapping[str, Any], *, check_paths: bool) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("config root must be a mapping")
    _reject_unknown_keys(config, ROOT_KEYS, prefix="config")

    checkpoint = _nonempty_string(config.get("checkpoint"), "checkpoint")
    allow_download = _boolean(config, "allow_download", prefix="", default=True)
    datasets_value = config.get("datasets")
    if not isinstance(datasets_value, list) or not datasets_value:
        raise ValueError("config.datasets must be a non-empty list")
    datasets: list[Mapping[str, Any]] = []
    for index, source_value in enumerate(datasets_value):
        if not isinstance(source_value, Mapping):
            raise ValueError(f"config.datasets[{index}] must be a mapping")
        source = source_value
        for ignored_key in ("episodes", "weight"):
            if ignored_key in source:
                raise ValueError(
                    f"config.datasets[{index}].{ignored_key} is not supported by the "
                    "complete pipeline and would be ignored"
                )
        _reject_unknown_keys(source, DATASET_KEYS, prefix=f"config.datasets[{index}]")
        _nonempty_string(source.get("repo_id"), f"datasets[{index}].repo_id")
        _nonempty_string(source.get("root"), f"datasets[{index}].root")
        if source.get("revision") is not None:
            _nonempty_string(source["revision"], f"datasets[{index}].revision")
        if source.get("action_key") is not None:
            _nonempty_string(source["action_key"], f"datasets[{index}].action_key")
        _string_mapping(source.get("rename_map", {}), f"datasets[{index}].rename_map", nonempty=False)
        datasets.append(source)

    action_cache = _mapping(config.get("action_cache"), "action_cache")
    tactile_cache = _mapping(config.get("tactile_embedding_cache"), "tactile_embedding_cache")
    model = _mapping(config.get("model"), "model")
    norm_stats = _mapping(config.get("norm_stats"), "norm_stats")
    training = _mapping(config.get("frs_training"), "frs_training")
    for section, allowed, prefix in (
        (action_cache, ACTION_CACHE_KEYS, "config.action_cache"),
        (tactile_cache, TACTILE_CACHE_KEYS, "config.tactile_embedding_cache"),
        (model, MODEL_KEYS, "config.model"),
        (norm_stats, NORM_STATS_KEYS, "config.norm_stats"),
        (training, TRAINING_KEYS, "config.frs_training"),
    ):
        _reject_unknown_keys(section, allowed, prefix=prefix)

    sanitized_cache_dirs = [
        source_cache_dir(str(action_cache.get("root", "")), str(source["repo_id"]))
        for source in datasets
    ]
    if len(set(sanitized_cache_dirs)) != len(sanitized_cache_dirs):
        raise ValueError(
            "config.datasets entries resolve to the same sanitized action-cache directory"
        )

    for section, key in (
        (action_cache, "root"),
        (tactile_cache, "root"),
        (model, "tactile_encoder_path"),
        (norm_stats, "dir"),
        (norm_stats, "asset_id"),
        (training, "output"),
    ):
        _nonempty_string(section.get(key), key)

    tactile_enabled = _boolean(
        tactile_cache, "enabled", prefix="tactile_embedding_cache.", default=True
    )
    use_tactile_encoder = _boolean(
        model, "use_tactile_encoder", prefix="model.", default=True
    )
    freeze_tactile_encoder = _boolean(
        model, "freeze_tactile_encoder", prefix="model.", default=True
    )
    if not tactile_enabled or not use_tactile_encoder or not freeze_tactile_encoder:
        raise ValueError(
            "the pipeline requires tactile_embedding_cache.enabled, "
            "model.use_tactile_encoder, and model.freeze_tactile_encoder to be true"
        )
    _boolean(model, "state_conditioning", prefix="model.", default=False)
    _boolean(norm_stats, "use_quantile_norm", prefix="norm_stats.", default=True)
    _boolean(training, "write_plots", prefix="frs_training.", default=True)
    _boolean(training, "resume", prefix="frs_training.", default=False)

    tactile_keys = model.get("tactile_keys")
    if not isinstance(tactile_keys, list) or not tactile_keys:
        raise ValueError("config.model.tactile_keys must be a non-empty list")
    for index, key in enumerate(tactile_keys):
        _nonempty_string(key, f"model.tactile_keys[{index}]")
    camera_map = _string_mapping(model.get("camera_map"), "model.camera_map", nonempty=True)
    allowed_cameras = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    unknown_cameras = set(camera_map) - allowed_cameras
    if unknown_cameras:
        raise ValueError(
            f"config.model.camera_map has unknown keys: {sorted(unknown_cameras)}"
        )

    for key, default in (
        ("model_sample_steps", 10),
        ("reverse_steps", 50),
        ("batch_size", 16),
        ("load_workers", 4),
        ("flush_every", 8),
        ("frame_stride", 3),
    ):
        _integer(action_cache, key, prefix="action_cache.", default=default)
    _integer(action_cache, "inference_seed", prefix="action_cache.", default=0, minimum=0)
    _integer(action_cache, "split_seed", prefix="action_cache.", default=42, minimum=0)
    _integer(action_cache, "drop_tail_action_chunks", prefix="action_cache.", default=1, minimum=0)
    for key in ("max_episodes", "max_samples"):
        if action_cache.get(key) is not None:
            _integer(action_cache, key, prefix="action_cache.", default=1)
    reverse_solver = _nonempty_string(action_cache.get("reverse_solver", "fireflow"), "action_cache.reverse_solver")
    if reverse_solver not in ("euler", "fireflow", "slerpflow"):
        raise ValueError("config.action_cache.reverse_solver is invalid")
    val_fraction = _number(action_cache, "val_fraction", prefix="action_cache.", default=0.1)
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("config.action_cache.val_fraction must be in (0, 1)")

    for key, default, minimum in (
        ("precompute_batch_size", 128, 1),
        ("precompute_num_workers", 4, 0),
        ("precompute_prefetch_factor", 2, 1),
        ("precompute_flush_every", 20, 1),
    ):
        _integer(tactile_cache, key, prefix="tactile_embedding_cache.", default=default, minimum=minimum)
    if tactile_cache.get("dtype", "float16") not in ("float16", "float32"):
        raise ValueError("config.tactile_embedding_cache.dtype must be float16 or float32")
    if tactile_cache.get("precompute_video_backend") is not None:
        _nonempty_string(
            tactile_cache["precompute_video_backend"],
            "tactile_embedding_cache.precompute_video_backend",
        )

    model_integers = {
        key: _integer(model, key, prefix="model.", default=default)
        for key, default in (
        ("tactile_embedding_dim", 512),
        ("tactile_num_tokens", 4),
        ("tactile_image_size", 224),
        ("action_dim", 32),
        ("action_horizon", 50),
        )
    }
    if len(tactile_keys) != 4 or model_integers["tactile_num_tokens"] != 4:
        raise ValueError(
            "config.model.tactile_keys and tactile_num_tokens must each be exactly 4"
        )
    state_dropout_rate = _number(
        model, "state_dropout_rate", prefix="model.", default=0.0
    )
    if not 0.0 <= state_dropout_rate < 1.0:
        raise ValueError("config.model.state_dropout_rate must be in [0, 1)")
    allowed_variants = {
        "dummy",
        "gemma_300m",
        "gemma_2b",
        "gemma_300m_lora",
        "gemma_2b_lora",
    }
    for key, default in (
        ("paligemma_variant", "gemma_2b"),
        ("action_expert_variant", "gemma_300m"),
    ):
        variant = _nonempty_string(model.get(key, default), f"model.{key}")
        if variant not in allowed_variants:
            raise ValueError(f"config.model.{key} is not a supported Gemma variant")

    training_integers = {
        key: _integer(training, key, prefix="frs_training.", default=default)
        for key, default in (
        ("tactile_window_divisor", 1),
        ("history_stride", 3),
        ("aux_decode_steps", 10),
        ("model_dim", 256),
        ("depth", 6),
        ("num_heads", 4),
        ("mlp_ratio", 4),
        ("batch_size", 64),
        ("epochs", 300),
        ("validation_steps", 10),
        ("eval_every", 5),
        )
    }
    _integer(training, "warmup_epochs", prefix="frs_training.", default=5, minimum=0)
    _integer(training, "lr_reference_dim", prefix="frs_training.", default=256)
    _integer(training, "seed", prefix="frs_training.", default=42, minimum=0)
    if (
        model_integers["action_horizon"]
        % training_integers["tactile_window_divisor"]
        != 0
    ):
        raise ValueError(
            "config.model.action_horizon must be divisible by "
            "config.frs_training.tactile_window_divisor"
        )
    if training_integers["model_dim"] % training_integers["num_heads"] != 0:
        raise ValueError(
            "config.frs_training.model_dim must be divisible by "
            "config.frs_training.num_heads"
        )
    numeric_defaults = {
        "gate_tau": 0.5,
        "gate_temperature": 0.1,
        "gate_lambda": 1.0,
        "aux_decode_weight": 1.0,
        "low_gate_safety_weight": 0.0,
        "low_gate_safety_margin": 0.03,
        "rank_low_gate_threshold": 0.3,
        "rank_high_gate_threshold": 0.7,
        "rank_weight": 0.0,
        "rank_margin": 0.0,
        "repair_weight": 0.0,
        "repair_margin": 0.0,
        "best_max_low_gate_unsafe_frac": 0.1,
        "best_min_high_gate_gain": 0.0,
        "best_min_high_gate_rank_satisfied_frac": 0.8,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "min_lr_ratio": 0.1,
    }
    numbers = {
        key: _number(training, key, prefix="frs_training.", default=default)
        for key, default in numeric_defaults.items()
    }
    if not 0.0 <= numbers["gate_tau"] <= 1.0:
        raise ValueError("config.frs_training.gate_tau must be in [0, 1]")
    if numbers["gate_temperature"] <= 0.0:
        raise ValueError("config.frs_training.gate_temperature must be positive")
    if numbers["learning_rate"] <= 0.0 or numbers["grad_clip_norm"] <= 0.0:
        raise ValueError("config.frs_training learning_rate and grad_clip_norm must be positive")
    for key in (
        "gate_lambda",
        "aux_decode_weight",
        "low_gate_safety_weight",
        "low_gate_safety_margin",
        "rank_weight",
        "rank_margin",
        "repair_weight",
        "repair_margin",
        "weight_decay",
    ):
        if numbers[key] < 0.0:
            raise ValueError(f"config.frs_training.{key} must be non-negative")
    low = numbers["rank_low_gate_threshold"]
    high = numbers["rank_high_gate_threshold"]
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("config.frs_training gate thresholds must satisfy 0 <= low < high <= 1")
    for key in (
        "best_max_low_gate_unsafe_frac",
        "best_min_high_gate_rank_satisfied_frac",
        "min_lr_ratio",
    ):
        if not 0.0 <= numbers[key] <= 1.0:
            raise ValueError(f"config.frs_training.{key} must be in [0, 1]")
    if training.get("resume_from") not in (None, ""):
        _nonempty_string(training["resume_from"], "frs_training.resume_from")
    if training.get("loss_mode", "gated") not in ("gt", "predicted", "gated"):
        raise ValueError("config.frs_training.loss_mode is invalid")
    if training.get("aux_decode_solver", "fireflow") not in ("euler", "fireflow"):
        raise ValueError("config.frs_training.aux_decode_solver is invalid")
    if training.get("lr_schedule", "cosine") not in ("cosine", "constant"):
        raise ValueError("config.frs_training.lr_schedule is invalid")

    if _is_url(checkpoint) and not allow_download:
        raise ValueError("config.allow_download must be true for a URL checkpoint")

    for value, field in (
        (action_cache["root"], "action_cache.root"),
        (tactile_cache["root"], "tactile_embedding_cache.root"),
        (training["output"], "frs_training.output"),
    ):
        _output_target(value, field)
    _local_path(model["tactile_encoder_path"], "model.tactile_encoder_path")
    for index, source in enumerate(datasets):
        _local_path(source["root"], f"datasets[{index}].root")

    if not check_paths:
        return config

    if _is_url(checkpoint):
        pass
    else:
        checkpoint_path = resolve_local_path(checkpoint)
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
        if not (checkpoint_path / "params").is_dir():
            raise FileNotFoundError(f"checkpoint is missing params/: {checkpoint_path}")

    encoder = _local_path(model["tactile_encoder_path"], "model.tactile_encoder_path")
    if not encoder.is_dir():
        raise FileNotFoundError(f"tactile encoder does not exist: {encoder}")
    _validate_encoder_checkpoint(
        encoder,
        expected_embedding_dim=model_integers["tactile_embedding_dim"],
        expected_image_size=model_integers["tactile_image_size"],
    )
    for index, source in enumerate(datasets):
        dataset_root = _local_path(source["root"], f"datasets[{index}].root")
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"dataset does not exist: {dataset_root}")
        info_path = dataset_root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"dataset is missing meta/info.json: {dataset_root}")
        info = _json_mapping(info_path, f"dataset {index} metadata")
        if info.get("codebase_version") != "v3.0":
            raise ValueError(f"dataset must be LeRobot v3.0: {dataset_root}")
        features = info.get("features")
        if not isinstance(features, Mapping):
            raise ValueError(f"dataset features must be a mapping: {dataset_root}")
        visual_keys = {
            str(key)
            for key, feature in features.items()
            if isinstance(feature, Mapping) and feature.get("dtype") in ("image", "video")
        }
        rename_map = source.get("rename_map", {})
        post_rename_visual_keys = {
            str(rename_map.get(key, key)) for key in visual_keys
        }
        required_visual_keys = set(camera_map.values()) | set(tactile_keys)
        missing_visual_keys = required_visual_keys - post_rename_visual_keys
        if missing_visual_keys:
            raise ValueError(
                f"config.model.camera_map/tactile_keys reference missing dataset features "
                f"after rename for {source['repo_id']}: {sorted(missing_visual_keys)}"
            )
        action_key = source.get("action_key")
        if action_key is not None and action_key not in features:
            raise ValueError(
                f"config.datasets[{index}].action_key is missing from dataset features: "
                f"{action_key}"
            )
    norm_dir = str(norm_stats["dir"])
    if not _is_url(norm_dir):
        asset_dir = resolve_local_path(norm_dir) / str(norm_stats["asset_id"])
        if not asset_dir.is_dir():
            raise FileNotFoundError(f"norm stats asset does not exist: {asset_dir}")
        _validate_norm_stats(
            asset_dir,
            use_quantile_norm=bool(norm_stats["use_quantile_norm"]),
        )
    resume_from = training.get("resume_from")
    if resume_from not in (None, "") and not resolve_local_path(str(resume_from)).is_dir():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_from}")
    if training.get("resume", False) and resume_from in (None, ""):
        last = _output_target(training["output"], "frs_training.output") / "last"
        if not last.is_dir():
            raise FileNotFoundError(f"resume checkpoint does not exist: {last}")
    for value, field in (
        (action_cache["root"], "action_cache.root"),
        (tactile_cache["root"], "tactile_embedding_cache.root"),
        (training["output"], "frs_training.output"),
    ):
        target = _output_target(value, field)
        ancestor = target
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
            raise PermissionError(f"config.{field} has no writable parent: {ancestor}")
    return config


def _positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    return _integer(config, key, prefix="frs_training.", default=default)


def train_from_config(config: Mapping[str, Any]) -> None:
    validate_config(config, check_paths=True)
    datasets = config["datasets"]
    action_cache = config["action_cache"]
    tactile_cache = config["tactile_embedding_cache"]
    model = config["model"]
    training = config["frs_training"]
    dataset_sources = resolved_dataset_sources(datasets)
    encoder_dir = resolve_local_path(str(model["tactile_encoder_path"]))
    if not encoder_dir.is_dir():
        raise FileNotFoundError(f"tactile encoder does not exist: {encoder_dir}")
    cache_dirs = [source_cache_dir(action_cache["root"], str(source["repo_id"])) for source in datasets]
    missing = [path for path in cache_dirs if not (path / "manifest.json").is_file()]
    if missing:
        raise FileNotFoundError(f"action caches are missing: {missing}")

    train_decoder(
        cache_dir=None,
        tactile_encoder_dir=encoder_dir,
        output_dir=resolve_local_path(str(training["output"])),
        dataset_repo_id=None,
        dataset_root=None,
        tactile_window_divisor=_positive_int(training, "tactile_window_divisor", 1),
        history_stride=_positive_int(training, "history_stride", 3),
        loss_mode=str(training.get("loss_mode", "gated")),
        gate_tau=float(training.get("gate_tau", 0.5)),
        gate_temperature=float(training.get("gate_temperature", 0.1)),
        gate_lambda=float(training.get("gate_lambda", 1.0)),
        aux_decode_weight=float(training.get("aux_decode_weight", 1.0)),
        aux_decode_steps=_positive_int(training, "aux_decode_steps", 10),
        aux_decode_solver=str(training.get("aux_decode_solver", "fireflow")),
        low_gate_safety_weight=float(training.get("low_gate_safety_weight", 0.0)),
        low_gate_safety_margin=float(training.get("low_gate_safety_margin", 0.03)),
        rank_weight=float(training.get("rank_weight", 0.0)),
        rank_margin=float(training.get("rank_margin", 0.0)),
        repair_weight=float(training.get("repair_weight", 0.0)),
        repair_margin=float(training.get("repair_margin", 0.0)),
        low_gate_threshold=float(training.get("rank_low_gate_threshold", 0.3)),
        high_gate_threshold=float(training.get("rank_high_gate_threshold", 0.7)),
        state_conditioning=model.get("state_conditioning", False),
        state_dropout_rate=float(model.get("state_dropout_rate", 0.0)),
        best_max_low_gate_unsafe_frac=float(training.get("best_max_low_gate_unsafe_frac", 0.1)),
        best_min_high_gate_gain=float(training.get("best_min_high_gate_gain", 0.0)),
        best_min_high_gate_rank_satisfied_frac=float(training.get("best_min_high_gate_rank_satisfied_frac", 0.8)),
        model_dim=_positive_int(training, "model_dim", 256),
        depth=_positive_int(training, "depth", 6),
        num_heads=_positive_int(training, "num_heads", 4),
        mlp_ratio=_positive_int(training, "mlp_ratio", 4),
        learning_rate=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
        warmup_epochs=int(training.get("warmup_epochs", 5)),
        lr_reference_dim=int(training.get("lr_reference_dim", 256)),
        min_learning_rate_ratio=float(training.get("min_lr_ratio", 0.1)),
        cosine_decay=training.get("lr_schedule", "cosine") == "cosine",
        batch_size=_positive_int(training, "batch_size", 64),
        epochs=_positive_int(training, "epochs", 300),
        validation_steps=_positive_int(training, "validation_steps", 10),
        eval_every=_positive_int(training, "eval_every", 5),
        seed=int(training.get("seed", 42)),
        write_plots=training.get("write_plots", True),
        num_workers=0,
        prefetch_batches=1,
        load_threads=1,
        pipeline_prefetch=1,
        image_cache_size=0,
        encode_batch_size=1,
        resume=training.get("resume", False),
        resume_from=(None if training.get("resume_from") in (None, "") else resolve_local_path(str(training["resume_from"]))),
        cache_dirs=cache_dirs,
        dataset_sources=dataset_sources,
        tactile_embedding_cache_root=resolve_local_path(str(tactile_cache["root"])),
        tactile_keys=tuple(str(key) for key in model["tactile_keys"]),
        tactile_embedding_dim=int(model.get("tactile_embedding_dim", 512)),
        tactile_image_size=int(model.get("tactile_image_size", 224)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-output", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.check:
        validate_config(config, check_paths=True)
        if args.print_output:
            print(resolve_local_path(str(config["frs_training"]["output"])))
        else:
            print(f"configuration and input paths are valid: {args.config}")
        return
    if args.print_output:
        parser.error("--print-output requires --check")
    train_from_config(config)


if __name__ == "__main__":
    main()
