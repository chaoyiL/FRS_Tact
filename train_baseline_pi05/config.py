"""Dependency-light, strict configuration loading for direct Pi0.5 training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any, Mapping

import yaml


TACTILE_KEYS = (
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)
RIGHT_TACTILE_KEYS = (
    # The suffix identifies the arm/camera; left/right identifies a jaw face.
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _required(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"{section}.{key} is required.") from exc


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _path(value: object, name: str) -> Path:
    return Path(_string(value, name))


def _nullable_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _string_mapping(value: object, name: str) -> dict[str, str]:
    raw = _mapping(value, name)
    return {_string(key, f"{name} key"): _string(item, f"{name}[{key!r}]") for key, item in raw.items()}


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result

def _nonnegative_integer(value: object, name: str) -> int:
    result = _integer(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    return float(value)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, not a quoted boolean.")
    return value


def _tactile_keys(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a sequence of strings.")
    keys = tuple(value)
    if keys not in (TACTILE_KEYS, RIGHT_TACTILE_KEYS):
        raise ValueError(f"{name} must contain the approved tactile keys in canonical order.")
    return keys


@dataclass(frozen=True)
class DatasetConfig:
    repo_id: str
    root: Path
    revision: str | None
    action_key: str
    rename_map: dict[str, str]
    camera_map: dict[str, str]
    frame_stride: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    split_seed: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DatasetConfig":
        return cls(
            repo_id=_string(_required(raw, "repo_id", "dataset"), "dataset.repo_id"),
            root=_path(_required(raw, "root", "dataset"), "dataset.root"),
            revision=_nullable_string(raw.get("revision"), "dataset.revision"),
            action_key=_string(_required(raw, "action_key", "dataset"), "dataset.action_key"),
            rename_map=_string_mapping(raw.get("rename_map", {}), "dataset.rename_map"),
            camera_map=_string_mapping(raw.get("camera_map", {}), "dataset.camera_map"),
            frame_stride=_positive_integer(raw.get("frame_stride", 1), "dataset.frame_stride"),
            train_fraction=_number(_required(raw, "train_fraction", "dataset"), "dataset.train_fraction"),
            validation_fraction=_number(
                _required(raw, "validation_fraction", "dataset"), "dataset.validation_fraction"
            ),
            test_fraction=_number(_required(raw, "test_fraction", "dataset"), "dataset.test_fraction"),
            split_seed=_integer(_required(raw, "split_seed", "dataset"), "dataset.split_seed"),
        )


@dataclass(frozen=True)
class SourcePolicyConfig:
    checkpoint: Path
    norm_stats_dir: Path
    norm_stats_asset_id: str
    seed: int
    sample_steps: int
    action_horizon: int
    model_action_dim: int
    paligemma_variant: str
    action_expert_variant: str
    use_quantile_norm: bool
    allow_download: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SourcePolicyConfig":
        return cls(
            checkpoint=_path(_required(raw, "checkpoint", "source"), "source.checkpoint"),
            norm_stats_dir=_path(_required(raw, "norm_stats_dir", "source"), "source.norm_stats_dir"),
            norm_stats_asset_id=_string(
                _required(raw, "norm_stats_asset_id", "source"), "source.norm_stats_asset_id"
            ),
            seed=_integer(_required(raw, "seed", "source"), "source.seed"),
            sample_steps=_positive_integer(_required(raw, "sample_steps", "source"), "source.sample_steps"),
            action_horizon=_positive_integer(
                _required(raw, "action_horizon", "source"), "source.action_horizon"
            ),
            model_action_dim=_positive_integer(
                _required(raw, "model_action_dim", "source"), "source.model_action_dim"
            ),
            paligemma_variant=_string(
                _required(raw, "paligemma_variant", "source"), "source.paligemma_variant"
            ),
            action_expert_variant=_string(
                _required(raw, "action_expert_variant", "source"), "source.action_expert_variant"
            ),
            use_quantile_norm=_boolean(
                _required(raw, "use_quantile_norm", "source"), "source.use_quantile_norm"
            ),
            allow_download=_boolean(_required(raw, "allow_download", "source"), "source.allow_download"),
        )


@dataclass(frozen=True)
class TactileConfig:
    encoder_checkpoint: Path
    embedding_dim: int
    freeze_encoder: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TactileConfig":
        return cls(
            encoder_checkpoint=_path(
                _required(raw, "encoder_checkpoint", "tactile"), "tactile.encoder_checkpoint"
            ),
            embedding_dim=_positive_integer(
                _required(raw, "embedding_dim", "tactile"), "tactile.embedding_dim"
            ),
            freeze_encoder=_boolean(_required(raw, "freeze_encoder", "tactile"), "tactile.freeze_encoder"),
        )


@dataclass(frozen=True)
class CacheConfig:
    action_root: Path
    tactile_root: Path
    action_batch_size: int = 64
    tactile_batch_size: int = 32
    action_prefetch: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CacheConfig":
        return cls(
            action_root=_path(_required(raw, "action_root", "cache"), "cache.action_root"),
            tactile_root=_path(_required(raw, "tactile_root", "cache"), "cache.tactile_root"),
            action_batch_size=_positive_integer(raw.get("action_batch_size", 64), "cache.action_batch_size"),
            tactile_batch_size=_positive_integer(raw.get("tactile_batch_size", 32), "cache.tactile_batch_size"),
            action_prefetch=_boolean(raw.get("action_prefetch", False), "cache.action_prefetch"),
        )


@dataclass(frozen=True)
class DecoderTrainConfig:
    output: Path
    action_horizon: int = 50
    action_dim: int = 20
    tactile_dim: int = 512
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    tactile_keys: tuple[str, ...] = TACTILE_KEYS
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0
    workers: int = 0
    pin_memory: bool = False
    device: str = "cuda"
    resume: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DecoderTrainConfig":
        return cls(
            output=_path(_required(raw, "output", "decoder"), "decoder.output"),
            action_horizon=_positive_integer(
                raw.get("action_horizon", 50), "decoder.action_horizon"
            ),
            action_dim=_positive_integer(raw.get("action_dim", 20), "decoder.action_dim"),
            tactile_dim=_positive_integer(raw.get("tactile_dim", 512), "decoder.tactile_dim"),
            d_model=_positive_integer(raw.get("d_model", 128), "decoder.d_model"),
            nhead=_positive_integer(raw.get("nhead", 4), "decoder.nhead"),
            num_layers=_positive_integer(raw.get("num_layers", 2), "decoder.num_layers"),
            dim_feedforward=_positive_integer(
                raw.get("dim_feedforward", 256), "decoder.dim_feedforward"
            ),
            dropout=_number(raw.get("dropout", 0.1), "decoder.dropout"),
            tactile_keys=_tactile_keys(raw.get("tactile_keys", TACTILE_KEYS), "decoder.tactile_keys"),
            batch_size=_positive_integer(raw.get("batch_size", 256), "decoder.batch_size"),
            epochs=_positive_integer(raw.get("epochs", 50), "decoder.epochs"),
            learning_rate=_number(raw.get("learning_rate", 3e-4), "decoder.learning_rate"),
            weight_decay=_number(raw.get("weight_decay", 1e-4), "decoder.weight_decay"),
            seed=_integer(raw.get("seed", 0), "decoder.seed"),
            workers=_nonnegative_integer(raw.get("workers", 0), "decoder.workers"),
            pin_memory=_boolean(raw.get("pin_memory", False), "decoder.pin_memory"),
            device=_string(raw.get("device", "cuda"), "decoder.device"),
            resume=_boolean(raw.get("resume", False), "decoder.resume"),
        )


@dataclass(frozen=True)
class EvaluationConfig:
    split: str = "test"
    batch_size: int = 256
    shuffle_tactile: bool = True
    output: Path | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, default_batch_size: int) -> "EvaluationConfig":
        split = _string(raw.get("split", "test"), "evaluation.split")
        if split not in {"validation", "test"}:
            raise ValueError("evaluation.split must be validation or test.")
        output = raw.get("output")
        return cls(
            split=split,
            batch_size=_positive_integer(raw.get("batch_size", default_batch_size), "evaluation.batch_size"),
            shuffle_tactile=_boolean(raw.get("shuffle_tactile", True), "evaluation.shuffle_tactile"),
            output=None if output is None else _path(output, "evaluation.output"),
        )


@dataclass(frozen=True)
class BaselineTrainConfig:
    dataset: DatasetConfig
    source: SourcePolicyConfig
    tactile: TactileConfig
    cache: CacheConfig
    decoder: DecoderTrainConfig
    config_path: Path
    evaluation: EvaluationConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, config_path: Path) -> "BaselineTrainConfig":
        raw = _mapping(raw, "configuration")
        decoder = DecoderTrainConfig.from_mapping(_mapping(_required(raw, "decoder", "configuration"), "decoder"))
        return cls(
            dataset=DatasetConfig.from_mapping(_mapping(_required(raw, "dataset", "configuration"), "dataset")),
            source=SourcePolicyConfig.from_mapping(_mapping(_required(raw, "source", "configuration"), "source")),
            tactile=TactileConfig.from_mapping(_mapping(_required(raw, "tactile", "configuration"), "tactile")),
            cache=CacheConfig.from_mapping(_mapping(_required(raw, "cache", "configuration"), "cache")),
            decoder=decoder,
            evaluation=EvaluationConfig.from_mapping(_mapping(raw.get("evaluation", {}), "evaluation"), default_batch_size=decoder.batch_size),
            config_path=config_path,
        )

    def validate_contract(self) -> None:
        if self.source.action_horizon != 50:
            raise ValueError("source.action_horizon must be 50 for the direct decoder contract.")
        decoder_contract = {
            "action_horizon": 50,
            "tactile_dim": 512,
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
        }
        for field, expected in decoder_contract.items():
            if getattr(self.decoder, field) != expected:
                raise ValueError(f"decoder.{field} must be {expected} for the direct decoder contract.")
        if _integer(self.decoder.action_dim, "decoder.action_dim") not in (10, 20):
            raise ValueError("decoder.action_dim must be 10 or 20 for the direct decoder contract.")
        _tactile_keys(self.decoder.tactile_keys, "decoder.tactile_keys")
        if self.source.model_action_dim < self.decoder.action_dim:
            raise ValueError("source.model_action_dim must be at least decoder.action_dim.")
        if self.source.seed != 0:
            raise ValueError("source.seed must be 0 for the fixed Pi0.5 source policy.")
        if self.source.sample_steps != 10:
            raise ValueError("source.sample_steps must be 10 for the fixed Pi0.5 source policy.")
        if self.tactile.embedding_dim != 512:
            raise ValueError("tactile.embedding_dim must be 512.")
        if self.decoder.action_horizon != self.source.action_horizon:
            raise ValueError("decoder.action_horizon must match source.action_horizon.")
        if self.decoder.tactile_dim != self.tactile.embedding_dim:
            raise ValueError("decoder.tactile_dim must match tactile.embedding_dim.")
        if self.decoder.learning_rate <= 0 or self.decoder.weight_decay < 0:
            raise ValueError("decoder learning_rate must be positive and weight_decay non-negative.")
        fractions = (
            self.dataset.train_fraction,
            self.dataset.validation_fraction,
            self.dataset.test_fraction,
        )
        if any(fraction < 0.0 or fraction > 1.0 for fraction in fractions) or not isclose(
            sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("dataset split fractions must be non-negative and sum to 1.")
        validate_paths(self)


def _resolved(path: Path, config_path: Path) -> Path:
    return (path if path.is_absolute() else config_path.parent / path).resolve()


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def validate_paths(config: BaselineTrainConfig) -> None:
    """Reject writable locations that overlap source datasets or reference assets."""
    input_roots = tuple(
        _resolved(path, config.config_path)
        for path in (
            config.dataset.root,
            config.source.checkpoint,
            config.source.norm_stats_dir,
            config.tactile.encoder_checkpoint,
        )
    )
    outputs = tuple(
        _resolved(path, config.config_path)
        for path in (config.cache.action_root, config.cache.tactile_root, config.decoder.output)
    )
    for output in outputs:
        if any(_overlap(output, input_root) for input_root in input_roots):
            raise ValueError(f"writable output {output} overlaps an input asset root.")
    for index, output in enumerate(outputs):
        if any(_overlap(output, other) for other in outputs[index + 1 :]):
            raise ValueError("cache and decoder output roots must be distinct and non-overlapping.")


def load_config(path: Path) -> BaselineTrainConfig:
    """Load and validate a standalone direct decoder training configuration."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = BaselineTrainConfig.from_mapping(raw, config_path=path.resolve())
    config.validate_contract()
    return config
