"""Dependency-light state-only structured-noise Flow configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any, Mapping

import yaml


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
    reverse_steps: int = 50
    reverse_solver: str = "slerpflow"

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
            reverse_steps=_positive_integer(raw.get("reverse_steps", 50), "source.reverse_steps"),
            reverse_solver=_string(raw.get("reverse_solver", "slerpflow"), "source.reverse_solver"),
        )


@dataclass(frozen=True)
class CacheConfig:
    action_root: Path
    action_batch_size: int = 4
    action_prefetch: bool = False

    @classmethod
    def from_mapping(cls, raw):
        return cls(
            _path(_required(raw, "action_root", "cache"), "cache.action_root"),
            _positive_integer(raw.get("action_batch_size", 4), "cache.action_batch_size"),
            _boolean(raw.get("action_prefetch", False), "cache.action_prefetch"),
        )


@dataclass(frozen=True)
class DecoderTrainConfig:
    output: Path
    action_horizon: int = 50
    action_dim: int = 20
    state_dim: int | None = None
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.0
    decode_steps: int = 10
    decode_solver: str = "fireflow"
    state_dropout_rate: float = 0.1
    aux_decode_weight: float = 4.0
    grad_clip_norm: float = 1.0
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    workers: int = 0
    pin_memory: bool = False
    device: str = "cuda"
    resume: bool = False

    @classmethod
    def from_mapping(cls, raw):
        import dataclasses

        values = {"output": _path(_required(raw, "output", "decoder"), "decoder.output")}
        for field in dataclasses.fields(cls):
            if field.name == "output":
                continue
            value = raw.get(field.name, field.default)
            label = "decoder." + field.name
            if field.name == "state_dim":
                value = _positive_integer(
                    value if value is not None else (7 if raw.get("action_dim", 20) == 10 else 20), label
                )
            elif type(field.default) is bool:
                value = _boolean(value, label)
            elif type(field.default) is int:
                value = _integer(value, label)
            elif type(field.default) is float:
                value = _number(value, label)
            else:
                value = _string(value, label)
            values[field.name] = value
        return cls(**values)


@dataclass(frozen=True)
class EvaluationConfig:
    split: str = "test"
    batch_size: int = 128
    output: Path | None = None


@dataclass(frozen=True)
class BaselineTrainConfig:
    dataset: DatasetConfig
    source: SourcePolicyConfig
    cache: CacheConfig
    decoder: DecoderTrainConfig
    config_path: Path
    evaluation: EvaluationConfig

    def validate_contract(self):
        d = self.decoder
        for name, expected in dict(
            action_horizon=50,
            d_model=256,
            nhead=4,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.0,
            decode_steps=10,
            decode_solver="fireflow",
        ).items():
            if getattr(d, name) != expected:
                raise ValueError(f"decoder.{name} must be {expected}")
        if d.action_dim not in (10, 20) or d.state_dim != (7 if d.action_dim == 10 else 20):
            raise ValueError("decoder action/state profile must be 10/7 or 20/20")
        if self.source.action_horizon != 50 or self.source.model_action_dim < d.action_dim:
            raise ValueError("source dimensions do not match decoder")
        if self.source.seed != 0 or self.source.sample_steps != 10:
            raise ValueError("source seed/sample_steps must be 0/10")
        if self.source.reverse_solver not in ("euler", "fireflow", "slerpflow"):
            raise ValueError("unsupported source.reverse_solver")
        if (
            not 0 <= d.state_dropout_rate < 1
            or d.learning_rate <= 0
            or d.weight_decay < 0
            or d.grad_clip_norm <= 0
            or d.aux_decode_weight < 0
        ):
            raise ValueError("invalid decoder optimization parameters")
        if min(d.batch_size, d.epochs) <= 0 or d.workers < 0:
            raise ValueError("invalid batch_size/epochs/workers")
        fractions = (
            self.dataset.train_fraction,
            self.dataset.validation_fraction,
            self.dataset.test_fraction,
        )
        if any(not 0 <= value <= 1 for value in fractions) or not isclose(sum(fractions), 1.0, abs_tol=1e-9):
            raise ValueError("split fractions must sum to one")
        if self.evaluation.split not in ("validation", "test") or self.evaluation.batch_size <= 0:
            raise ValueError("invalid evaluation split or batch size")

        def resolved(path):
            return (path if path.is_absolute() else self.config_path.parent / path).resolve()

        inputs = [
            resolved(path) for path in (self.dataset.root, self.source.checkpoint, self.source.norm_stats_dir)
        ]
        outputs = [resolved(self.cache.action_root), resolved(d.output)]

        def overlap(left, right):
            return left == right or left.is_relative_to(right) or right.is_relative_to(left)

        if overlap(*outputs) or any(overlap(out, source) for out in outputs for source in inputs):
            raise ValueError("cache/output roots must be separate and cannot overlap input assets")


def load_config(path: Path):
    path = Path(path).resolve()
    raw = _mapping(yaml.safe_load(path.read_text()), "configuration")
    if any(key in raw for key in ("tactile", "tactile_embedding_cache")) or any(
        "tactile" in key for key in raw.get("decoder", {})
    ):
        raise ValueError("state-only configuration must not contain tactile settings")
    decoder = DecoderTrainConfig.from_mapping(raw["decoder"])
    evaluation = raw.get("evaluation", {})
    config = BaselineTrainConfig(
        DatasetConfig.from_mapping(raw["dataset"]),
        SourcePolicyConfig.from_mapping(raw["source"]),
        CacheConfig.from_mapping(raw["cache"]),
        decoder,
        path,
        EvaluationConfig(
            _string(evaluation.get("split", "test"), "evaluation.split"),
            _positive_integer(evaluation.get("batch_size", decoder.batch_size), "evaluation.batch_size"),
            None if evaluation.get("output") is None else _path(evaluation["output"], "evaluation.output"),
        ),
    )
    config.validate_contract()
    return config
