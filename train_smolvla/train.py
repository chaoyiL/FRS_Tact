#!/usr/bin/env python
"""Fine-tune JAX SmolVLA directly from a LeRobotDataset."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import yaml

from lerobot.datasets.transforms import build_image_transforms
from train_smolvla import JaxSmolVLA, JaxSmolVLAConfig
from train_smolvla.atomic_checkpoint import assemble_checkpoint_atomically
from train_smolvla.checkpoint import (
    count_expert_layers,
    count_vlm_layers,
    extend_vlm_layers,
    load_params,
    resolve_checkpoint,
)
from train_smolvla.data import (
    DatasetSource,
    LeRobotJaxDataLoader,
    parse_dataset_sources,
    split_sources_train_val,
)
from train_smolvla.lora import resolve_module_modes
from train_smolvla.training import JaxSmolVLATrainer
from train_smolvla.validation import contract_from_config, validate_checkpoint

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train.yaml"
DATA_SPLIT_FILENAME = "data_split.json"
DATA_SPLIT_VERSION = 1
ALLOWED_TOP_LEVEL_KEYS = frozenset(
    {
        "allow_download",
        "allow_tokenizer_download",
        "batch_size",
        "checkpoint",
        "data_parallel",
        "datasets",
        "eval_freq",
        "full_vlm_checkpoint",
        "image_transforms",
        "log_freq",
        "modality_dropout",
        "model",
        "num_workers",
        "output",
        "prefetch_factor",
        "resume",
        "return_uint8",
        "revision",
        "save_freq",
        "seed",
        "steps",
        "launcher",
        "validation",
        "video_backend",
        "wandb",
    }
)


def _no_extra_loader_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    del cfg
    return {}


def _identity_prepare_params(
    params: Mapping[str, Any],
    config: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    del config, seed
    return dict(params)


@dataclass(frozen=True)
class TrainingComponents:
    """Concrete package components used by the shared training orchestration."""

    config_type: type[Any]
    model_type: type[Any]
    loader_type: type[Any]
    trainer_type: type[Any]
    resolve_checkpoint: Callable[..., Path]
    load_params: Callable[[str | Path], dict[str, Any]]
    count_vlm_layers: Callable[[Mapping[str, Any]], int]
    count_expert_layers: Callable[[Mapping[str, Any]], int]
    extend_vlm_layers: Callable[..., dict[str, Any]]
    resolve_module_modes: Callable[[Any], Mapping[str, Any]]
    contract_from_config: Callable[[Any], Any]
    validate_checkpoint: Callable[..., Any]
    prepare_params: Callable[..., dict[str, Any]] = _identity_prepare_params
    extra_loader_kwargs: Callable[[Mapping[str, Any]], dict[str, Any]] = _no_extra_loader_kwargs
    allowed_top_level_keys: frozenset[str] = ALLOWED_TOP_LEVEL_KEYS


VISUAL_COMPONENTS = TrainingComponents(
    config_type=JaxSmolVLAConfig,
    model_type=JaxSmolVLA,
    loader_type=LeRobotJaxDataLoader,
    trainer_type=JaxSmolVLATrainer,
    resolve_checkpoint=resolve_checkpoint,
    load_params=load_params,
    count_vlm_layers=count_vlm_layers,
    count_expert_layers=count_expert_layers,
    extend_vlm_layers=extend_vlm_layers,
    resolve_module_modes=resolve_module_modes,
    contract_from_config=contract_from_config,
    validate_checkpoint=validate_checkpoint,
)


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    default_config: Path = DEFAULT_CONFIG,
    description: str = __doc__ or "SmolVLA training",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"YAML config path (default: {default_config})",
    )
    return parser.parse_args(argv)


def load_yaml_config(
    path: Path,
    *,
    allowed_top_level_keys: frozenset[str] = ALLOWED_TOP_LEVEL_KEYS,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    unknown = sorted(set(data) - allowed_top_level_keys)
    if unknown:
        raise ValueError(f"unknown top-level config keys in {path}: {unknown}")
    return data


def require(cfg: dict[str, Any], key: str) -> Any:
    if key not in cfg or cfg[key] in (None, ""):
        raise ValueError(f"missing required config field: {key}")
    return cfg[key]


def apply_model_overrides(config: JaxSmolVLAConfig, overrides: dict[str, Any] | None) -> JaxSmolVLAConfig:
    return config.with_overrides(overrides)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def _save_training_checkpoint_atomically(
    final_path: str | Path,
    *,
    trainer: JaxSmolVLATrainer,
    preprocessor: Any,
    source_dir: str | Path,
    data_split_path: str | Path | None,
    components: TrainingComponents | None = None,
) -> Path:
    contract_factory = (
        contract_from_config if components is None else components.contract_from_config
    )
    checkpoint_validator = (
        validate_checkpoint if components is None else components.validate_checkpoint
    )
    expected = contract_factory(trainer.config)

    def writer(staging: Path) -> None:
        trainer.save(staging, source_dir=source_dir)
        preprocessor.save_normalization_assets(staging)
        if data_split_path is not None:
            shutil.copy2(data_split_path, staging / DATA_SPLIT_FILENAME)

    def validator(staging: Path) -> None:
        checkpoint_validator(
            staging,
            expected=expected,
            base_sidecars=source_dir,
        ).require_valid()

    return assemble_checkpoint_atomically(final_path, writer, validator)


def _split_manifest(
    sources: list[DatasetSource],
    train_sources: list[DatasetSource],
    val_sources: list[DatasetSource],
    *,
    val_fraction: float,
    split_seed: int,
    eval_seed: int,
    sample_seed: int,
) -> dict[str, Any]:
    if len({source.repo_id for source in sources}) != len(sources):
        raise ValueError("data split persistence requires unique dataset repo_id values")
    train_by_repo = {source.repo_id: source for source in train_sources}
    val_by_repo = {source.repo_id: source for source in val_sources}
    datasets = []
    for source in sources:
        train = train_by_repo[source.repo_id]
        val = val_by_repo.get(source.repo_id)
        datasets.append(
            {
                "repo_id": source.repo_id,
                "revision": source.revision,
                "train_episodes": list(train.episodes or []),
                "val_episodes": list(val.episodes or []) if val is not None else [],
            }
        )
    return {
        "version": DATA_SPLIT_VERSION,
        "val_fraction": float(val_fraction),
        "split_seed": int(split_seed),
        "eval_seed": int(eval_seed),
        "sample_seed": int(sample_seed),
        "validation_sample_indices": [],
        "datasets": datasets,
    }


def _sources_from_split_manifest(
    sources: list[DatasetSource],
    manifest: dict[str, Any],
    *,
    val_fraction: float,
) -> tuple[list[DatasetSource], list[DatasetSource]]:
    if int(manifest.get("version", -1)) != DATA_SPLIT_VERSION:
        raise ValueError(f"unsupported data split version: {manifest.get('version')}")
    saved_fraction = float(manifest.get("val_fraction", -1.0))
    if not abs(saved_fraction - val_fraction) < 1e-12:
        raise ValueError(
            f"validation fraction changed since split creation: {saved_fraction} != {val_fraction}"
        )
    entries = manifest.get("datasets")
    if not isinstance(entries, list) or len(entries) != len(sources):
        raise ValueError("data split datasets do not match the configured dataset count")

    train_sources: list[DatasetSource] = []
    val_sources: list[DatasetSource] = []
    for source, entry in zip(sources, entries, strict=True):
        if not isinstance(entry, dict) or entry.get("repo_id") != source.repo_id:
            raise ValueError(
                f"data split dataset order mismatch: expected {source.repo_id!r}, "
                f"got {entry.get('repo_id') if isinstance(entry, dict) else entry!r}"
            )
        if entry.get("revision") != source.revision:
            raise ValueError(
                f"data split revision mismatch for {source.repo_id}: "
                f"{entry.get('revision')!r} != {source.revision!r}"
            )
        train_ids = [int(value) for value in entry.get("train_episodes", [])]
        val_ids = [int(value) for value in entry.get("val_episodes", [])]
        if not train_ids or set(train_ids) & set(val_ids):
            raise ValueError(f"invalid persisted episode split for {source.repo_id}")
        if source.episodes is not None and set(train_ids) | set(val_ids) != set(source.episodes):
            raise ValueError(f"persisted split is incompatible with explicit episodes for {source.repo_id}")
        train_sources.append(replace(source, episodes=train_ids))
        if val_ids:
            val_sources.append(replace(source, episodes=val_ids))
    return train_sources, val_sources


def _load_split_manifest(paths: list[Path]) -> tuple[dict[str, Any] | None, Path | None]:
    for path in paths:
        if path.is_file():
            with path.open(encoding="utf-8") as file:
                manifest = json.load(file)
            if not isinstance(manifest, dict):
                raise ValueError(f"data split manifest must be a mapping: {path}")
            return manifest, path
    return None, None


def init_wandb(
    cfg: dict[str, Any],
    *,
    config_path: Path,
    checkpoint: Path,
    model: JaxSmolVLAConfig,
    data_split: dict[str, Any] | None,
):
    wandb_cfg = cfg.get("wandb") or {}
    if not bool(wandb_cfg.get("enabled", False)):
        return None

    import wandb

    mode = str(wandb_cfg.get("mode", "online"))
    run = wandb.init(
        project=wandb_cfg.get("project", "smolvla-jax"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name"),
        group=wandb_cfg.get("group"),
        tags=list(wandb_cfg.get("tags") or []),
        notes=wandb_cfg.get("notes"),
        dir=str(Path(require(cfg, "output"))),
        mode=mode,
        config={
            "config_path": str(config_path.resolve()),
            "checkpoint": str(checkpoint),
            "datasets": cfg.get("datasets"),
            "batch_size": cfg.get("batch_size"),
            "steps": cfg.get("steps"),
            "seed": cfg.get("seed"),
            "data_parallel": cfg.get("data_parallel"),
            "validation": cfg.get("validation"),
            "resolved_data_split": data_split,
            "image_transforms": cfg.get("image_transforms"),
            "modality_dropout": cfg.get("modality_dropout"),
            "model": model.to_dict(),
            "wandb": {k: v for k, v in wandb_cfg.items() if k != "api_key"},
        },
    )
    print(f"wandb={run.url if run is not None else mode}")
    return run


def run_validation(
    trainer: JaxSmolVLATrainer,
    val_data: LeRobotJaxDataLoader,
    *,
    step: int,
    eval_count: int,
    seed: int,
    val_cfg: dict[str, Any],
    wandb_run,
) -> int:
    max_batches = val_cfg.get("max_batches")
    rollout = bool(val_cfg.get("rollout", True))
    rollout_steps = val_cfg.get("rollout_steps")
    metrics = trainer.evaluate(
        val_data.batches(),
        seed=seed,
        max_batches=None if max_batches in (None, 0) else int(max_batches),
        rollout=rollout,
        rollout_steps=None if rollout_steps in (None, 0) else int(rollout_steps),
    )
    eval_count += 1
    mse_text = f" action_mse={metrics['action_mse']:.6f}" if rollout else ""
    print(
        f"val step={step} loss={metrics['loss']:.6f}{mse_text} "
        f"samples={int(metrics['n_samples'])} eval_count={eval_count}"
    )
    if wandb_run is not None:
        import wandb

        payload = {"val/loss": float(metrics["loss"])}
        if rollout:
            payload["val/action_mse"] = float(metrics["action_mse"])
        wandb.log(payload, step=step)
    return eval_count


def run_training(
    config_path: Path,
    *,
    components: TrainingComponents = VISUAL_COMPONENTS,
) -> None:
    cfg = load_yaml_config(
        config_path,
        allowed_top_level_keys=components.allowed_top_level_keys,
    )

    checkpoint = components.resolve_checkpoint(
        require(cfg, "checkpoint"),
        revision=cfg.get("revision"),
        local_files_only=not bool(cfg.get("allow_download", False)),
    )
    print(f"config={config_path.resolve()}")
    print(f"checkpoint={checkpoint}")

    config = apply_model_overrides(
        components.config_type.from_pretrained(checkpoint),
        cfg.get("model"),
    )
    print(
        f"model overrides: action_dim={config.action_dim} state_dim={config.state_dim} "
        f"image_keys={list(config.image_keys)} "
        f"num_vlm_layers={config.num_vlm_layers} num_expert_layers={config.num_expert_layers} "
        f"expert_width_multiplier={config.expert_width_multiplier} "
        f"text_hidden_size={config.text_hidden_size} expert_hidden_size={config.expert_hidden_size}"
    )

    allow_download = bool(cfg.get("allow_download", False))
    params = components.load_params(checkpoint)
    checkpoint_vlm_layers = components.count_vlm_layers(params)
    checkpoint_expert_layers = components.count_expert_layers(params)
    if config.num_expert_layers > checkpoint_expert_layers:
        raise ValueError(
            f"requested {config.num_expert_layers} expert layers, but checkpoint only has "
            f"{checkpoint_expert_layers}; use num_expert_layers: -1 (auto) or "
            f"num_expert_layers: {checkpoint_expert_layers}"
        )
    if config.num_vlm_layers > checkpoint_vlm_layers:
        full_vlm_checkpoint = cfg.get("full_vlm_checkpoint") or config.tokenizer_name
        print(
            f"extending VLM: checkpoint_layers={checkpoint_vlm_layers} "
            f"requested_layers={config.num_vlm_layers} source={full_vlm_checkpoint}"
        )
        params = components.extend_vlm_layers(
            params,
            config.num_vlm_layers,
            source=full_vlm_checkpoint,
            local_files_only=not allow_download,
        )
        print(f"extended VLM parameters to {components.count_vlm_layers(params)} layers")

    params = components.prepare_params(
        params,
        config,
        seed=int(cfg.get("seed", 0)),
    )
    model = components.model_type(config)
    modality_dropout_cfg = cfg.get("modality_dropout")
    trainer = components.trainer_type(
        model,
        params,
        seed=int(cfg.get("seed", 0)),
        total_steps=int(require(cfg, "steps")),
        modality_dropout=modality_dropout_cfg,
    )
    md = trainer.modality_dropout
    print(
        "modality_dropout="
        + (
            f"enabled every_n={md.every_n_steps} prob={md.prob:g} "
            f"drop_language={md.drop_language} drop_state={md.drop_state} "
            f"camera_indices={md.camera_indices}"
            if md.enable
            else "disabled"
        )
    )
    trainable_count = sum(int(value.size) for value in trainer.state.params.values())
    frozen_count = sum(int(value.size) for value in trainer.frozen_params.values())
    print(f"module_modes={components.resolve_module_modes(config)}")
    print(
        f"parameters: trainable={trainable_count:,} frozen={frozen_count:,} "
        f"trainable_ratio={trainable_count / max(trainable_count + frozen_count, 1):.4%}"
    )
    resume = cfg.get("resume")
    if resume:
        trainer.restore(Path(resume))
    data_parallel = bool(cfg.get("data_parallel", False))
    if data_parallel:
        trainer.enable_data_parallel()

    allow_tokenizer_download = bool(cfg.get("allow_tokenizer_download", False))
    output = Path(require(cfg, "output"))
    output.mkdir(parents=True, exist_ok=True)
    sources = parse_dataset_sources(cfg)
    val_cfg = dict(cfg.get("validation") or {})
    val_enabled = bool(val_cfg.get("enabled", True))
    val_fraction = float(val_cfg.get("fraction", 0.1))
    split_seed = int(val_cfg.get("split_seed", cfg.get("seed", 0)))
    eval_seed = int(val_cfg.get("seed", 0))
    sample_seed = int(val_cfg.get("sample_seed", eval_seed + 1))
    train_sources = sources
    val_sources = []
    split_manifest = None
    split_path = output / DATA_SPLIT_FILENAME
    if val_enabled:
        split_candidates = []
        if resume:
            split_candidates.append(Path(resume) / DATA_SPLIT_FILENAME)
        split_candidates.append(split_path)
        split_manifest, loaded_split_path = _load_split_manifest(split_candidates)
        if split_manifest is None:
            train_sources, val_sources = split_sources_train_val(
                sources,
                val_fraction=val_fraction,
                seed=split_seed,
            )
            split_manifest = _split_manifest(
                sources,
                train_sources,
                val_sources,
                val_fraction=val_fraction,
                split_seed=split_seed,
                eval_seed=eval_seed,
                sample_seed=sample_seed,
            )
            print(f"created data split: {split_path}")
        else:
            train_sources, val_sources = _sources_from_split_manifest(
                sources,
                split_manifest,
                val_fraction=val_fraction,
            )
            eval_seed = int(split_manifest.get("eval_seed", eval_seed))
            sample_seed = int(split_manifest.get("sample_seed", sample_seed))
            print(f"loaded data split: {loaded_split_path}")
        _write_json(split_path, split_manifest)
        if not val_sources:
            print("warning: validation enabled but no held-out episodes; disabling val")
            val_enabled = False
            train_sources = sources

    common_loader_kwargs = {
        "batch_size": int(cfg.get("batch_size", 8)),
        "num_workers": int(cfg.get("num_workers", 4)),
        "prefetch_factor": int(cfg.get("prefetch_factor", 2)),
        "video_backend": cfg.get("video_backend"),
        "return_uint8": bool(cfg.get("return_uint8", True)),
        "seed": trainer.seed,
        "local_files_only": not (allow_tokenizer_download or allow_download),
    }
    common_loader_kwargs.update(components.extra_loader_kwargs(cfg))
    if data_parallel and common_loader_kwargs["batch_size"] % jax.device_count():
        raise ValueError(
            f"batch_size={common_loader_kwargs['batch_size']} must be divisible by "
            f"the {jax.device_count()} data-parallel devices"
        )
    train_image_transforms = build_image_transforms(cfg.get("image_transforms"))
    print(
        f"data_loader: video_backend={cfg.get('video_backend') or 'auto'} "
        f"return_uint8={common_loader_kwargs['return_uint8']} "
        f"num_workers={common_loader_kwargs['num_workers']} "
        f"prefetch_factor={common_loader_kwargs['prefetch_factor']}"
    )
    print(
        "image_transforms="
        + (
            f"enabled max_num_transforms={train_image_transforms._cfg.max_num_transforms} "
            f"tfs={list(train_image_transforms.transforms)}"
            if train_image_transforms is not None
            else "disabled"
        )
    )
    data = components.loader_type(
        checkpoint,
        config,
        sources=train_sources,
        image_transforms=train_image_transforms,
        **common_loader_kwargs,
    )
    batches = data.batches(start_batch=trainer.step_count)
    for summary in data.dataset_summaries:
        print(
            f"train_dataset={summary['repo_id']} frames={summary['frames']} "
            f"episodes={summary['episodes']} fps={summary['fps']} "
            f"action_key={summary['action_key']!r} weight={summary['weight']} "
            f"visual_keys={summary.get('visual_keys')}"
        )
    print(f"train_frames={len(data.dataset)}")

    val_data = None
    if val_enabled:
        max_batches = val_cfg.get("max_batches")
        validation_protocol = {
            "batch_size": int(common_loader_kwargs["batch_size"]),
            "max_batches": None if max_batches in (None, 0) else int(max_batches),
            "rollout": bool(val_cfg.get("rollout", True)),
            "rollout_steps": val_cfg.get("rollout_steps"),
        }
        saved_protocol = split_manifest.get("validation_protocol")
        if saved_protocol is not None and saved_protocol != validation_protocol:
            raise ValueError(
                "validation protocol changed since data_split.json was created: "
                f"{saved_protocol} != {validation_protocol}"
            )
        split_manifest["validation_protocol"] = validation_protocol
        fixed_subset_size = (
            None if max_batches in (None, 0) else int(max_batches) * int(common_loader_kwargs["batch_size"])
        )
        persisted_indices = tuple(split_manifest.get("validation_sample_indices", []))
        # Validation must stay unaugmented for stable metrics.
        val_data = components.loader_type(
            checkpoint,
            config,
            sources=val_sources,
            preprocessor=data.preprocessor,
            shuffle=False,
            infinite=False,
            drop_last=True,
            image_transforms=None,
            fixed_subset_size=fixed_subset_size if not persisted_indices else None,
            fixed_subset_seed=sample_seed,
            subset_indices=persisted_indices or None,
            **common_loader_kwargs,
        )
        validation_dataset_frames = {
            summary["repo_id"]: int(summary["frames"]) for summary in val_data.dataset_summaries
        }
        saved_dataset_frames = split_manifest.get("validation_dataset_frames")
        if saved_dataset_frames is not None and saved_dataset_frames != validation_dataset_frames:
            raise ValueError(
                "validation dataset frame counts changed since data_split.json was created: "
                f"{saved_dataset_frames} != {validation_dataset_frames}"
            )
        split_manifest["validation_dataset_frames"] = validation_dataset_frames
        split_manifest["validation_sample_indices"] = list(val_data.subset_indices)
        split_manifest["eval_seed"] = eval_seed
        split_manifest["sample_seed"] = sample_seed
        _write_json(split_path, split_manifest)
        for summary in val_data.dataset_summaries:
            print(
                f"val_dataset={summary['repo_id']} frames={summary['frames']} "
                f"episodes={summary['episodes']} fps={summary['fps']} "
                f"action_key={summary['action_key']!r} weight={summary['weight']}"
            )
        print(
            f"val_frames={val_data.full_dataset_size} sampled_frames={len(val_data.dataset)} "
            f"fraction={val_fraction:g} eval_seed={eval_seed} sample_seed={sample_seed} "
            f"rollout={bool(val_cfg.get('rollout', True))}"
        )

    steps = int(require(cfg, "steps"))
    log_freq = int(cfg.get("log_freq", 10))
    save_freq = int(cfg.get("save_freq", 1000))
    eval_freq = int(cfg.get("eval_freq", val_cfg.get("eval_freq", save_freq)))
    if eval_freq <= 0:
        raise ValueError(f"eval_freq must be positive, got {eval_freq}")
    wandb_cfg = cfg.get("wandb") or {}
    wandb_run = init_wandb(
        cfg,
        config_path=config_path,
        checkpoint=checkpoint,
        model=config,
        data_split=split_manifest,
    )
    log_checkpoints = bool(wandb_cfg.get("log_checkpoints", False))
    eval_count = 0

    start = time.perf_counter()
    last_log_time = start
    last_log_step = trainer.step_count
    try:
        while trainer.step_count < steps:
            metrics = trainer.step(next(batches))
            step = trainer.step_count
            drop_info = trainer.last_dropout_info
            drop_applied = bool(drop_info["applied"])
            drop_name = str(drop_info["modality"])
            if drop_applied:
                print(f"step={step} drop={drop_name}")
            if step == 1 or step % log_freq == 0:
                metrics = jax.device_get(metrics)
                now = time.perf_counter()
                elapsed = now - start
                window_steps = step - last_log_step
                window_seconds = max(now - last_log_time, 1e-9)
                steps_per_s = window_steps / window_seconds
                samples_per_s = steps_per_s * int(cfg.get("batch_size", 8))
                last_log_time = now
                last_log_step = step
                loss = float(metrics["loss"])
                grad_norm = float(metrics["grad_norm"])
                lr = float(metrics["learning_rate"])
                print(
                    f"step={step} loss={loss:.6f} "
                    f"grad_norm={grad_norm:.4f} "
                    f"lr={lr:.3e} steps_per_s={steps_per_s:.3f} "
                    f"samples_per_s={samples_per_s:.1f} elapsed={elapsed:.1f}s"
                    + (f" drop={drop_name}" if drop_applied else "")
                )
                if wandb_run is not None:
                    import wandb

                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/grad_norm": grad_norm,
                            "train/learning_rate": lr,
                            "train/steps_per_s": steps_per_s,
                            "train/samples_per_s": samples_per_s,
                            "train/elapsed_s": elapsed,
                        },
                        step=step,
                    )
            if val_data is not None and (step % eval_freq == 0 or step == steps):
                eval_count = run_validation(
                    trainer,
                    val_data,
                    step=step,
                    eval_count=eval_count,
                    seed=eval_seed,
                    val_cfg=val_cfg,
                    wandb_run=wandb_run,
                )
            if step % save_freq == 0 or step == steps:
                path = output / f"checkpoint-{step:08d}"
                _save_training_checkpoint_atomically(
                    path,
                    trainer=trainer,
                    preprocessor=data.preprocessor,
                    source_dir=checkpoint,
                    data_split_path=split_path if split_manifest is not None else None,
                    components=components,
                )
                print(f"saved checkpoint: {path}")
                if wandb_run is not None:
                    import wandb

                    wandb.log({"train/checkpoint_step": step}, step=step)
                    if log_checkpoints:
                        wandb.save(str(path / "*"), base_path=str(output))
    finally:
        if wandb_run is not None:
            import wandb

            wandb.finish()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_training(args.config)


if __name__ == "__main__":
    main()
