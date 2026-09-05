"""Dependency-light configuration for direct Pi0.5 deployment."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml


TACTILE_KEYS = (
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)
# Suffix 1 is the right-hand camera; left/right are its two tactile faces.
RIGHT_TACTILE_KEYS = (
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)
_CAMERA_MAP = {
    "left_wrist_0_rgb": "observation.images.camera0",
    "right_wrist_0_rgb": "observation.images.camera1",
}


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _required(raw: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in raw:
        raise ValueError(f"{section}.{key} is required.")
    return raw[key]


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _path(value: object, name: str) -> Path:
    return Path(_string(value, name))


def _resolved_path(value: Path, base: Path) -> Path:
    return (value if value.is_absolute() else base / value).expanduser().resolve()


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _positive(value: object, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_number(value: object, name: str) -> float:
    result = _number(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


@dataclass(frozen=True)
class SourceConfig:
    checkpoint: Path
    seed: int
    sample_steps: int
    action_horizon: int
    action_dim: int
    state_dim: int
    model_action_dim: int
    paligemma_variant: str
    action_expert_variant: str
    camera_map: dict[str, str]


@dataclass(frozen=True)
class NormStatsConfig:
    directory: Path
    asset_id: str
    use_quantile_norm: bool


@dataclass(frozen=True)
class TactileEncoderConfig:
    checkpoint: Path
    embedding_dim: int
    tactile_keys: tuple[str, ...]
    key_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DirectDecoderDeploymentConfig:
    checkpoint: Path
    device: str
    action_horizon: int
    action_dim: int
    tactile_dim: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    tactile_keys: tuple[str, ...]


@dataclass(frozen=True)
class ConnectionConfig:
    address: str
    port: int
    retry_interval_s: float
    ping_interval_s: float
    ping_timeout_s: float
    observation_timeout_s: float
    action_ack_timeout_s: float
    token: str | None
    token_env: str
    require_token: bool


@dataclass(frozen=True)
class ObservationConfig:
    data_type: str
    language_prompt: str
    single_arm_mode: bool
    no_state_obs_mode: bool


@dataclass(frozen=True)
class ControlConfig:
    control_frequency: float
    controller_frequency: float
    action_horizon: int
    steps_per_inference: int
    max_normalized_action_abs: float
    max_normalized_delta_rms: float


@dataclass(frozen=True)
class RuntimeConfig:
    auto_start: bool
    warmup_runs: int
    max_iterations: int


@dataclass(frozen=True)
class LoggingConfig:
    save_observations: bool
    output_dir: Path
    save_every: int
    queue_size: int


@dataclass(frozen=True)
class DeploymentConfig:
    source: SourceConfig
    norm_stats: NormStatsConfig
    tactile_encoder: TactileEncoderConfig
    direct_decoder: DirectDecoderDeploymentConfig
    connection: ConnectionConfig
    observation: ObservationConfig
    control: ControlConfig
    runtime: RuntimeConfig
    logging: LoggingConfig
    config_path: Path

    @property
    def model(self) -> SourceConfig:
        """Compatibility name for the fixed Pi0.5 source model contract."""
        return self.source


def _source(raw: Mapping[str, Any]) -> SourceConfig:
    camera_map = _mapping(_required(raw, "camera_map", "source"), "source.camera_map")
    result = SourceConfig(
        checkpoint=_path(_required(raw, "checkpoint", "source"), "source.checkpoint"),
        seed=_integer(_required(raw, "seed", "source"), "source.seed"),
        sample_steps=_positive(_required(raw, "sample_steps", "source"), "source.sample_steps"),
        action_horizon=_positive(_required(raw, "action_horizon", "source"), "source.action_horizon"),
        action_dim=_positive(_required(raw, "action_dim", "source"), "source.action_dim"),
        state_dim=_positive(_required(raw, "state_dim", "source"), "source.state_dim"),
        model_action_dim=_positive(_required(raw, "model_action_dim", "source"), "source.model_action_dim"),
        paligemma_variant=_string(_required(raw, "paligemma_variant", "source"), "source.paligemma_variant"),
        action_expert_variant=_string(_required(raw, "action_expert_variant", "source"), "source.action_expert_variant"),
        camera_map={_string(k, "source.camera_map key"): _string(v, "source.camera_map value") for k, v in camera_map.items()},
    )
    if result.action_horizon != 50 or (result.action_dim, result.state_dim) not in ((10, 7), (20, 20)):
        raise ValueError("source must use horizon=50 and action/state dimensions 10/7 or 20/20.")
    if result.model_action_dim < result.action_dim:
        raise ValueError("source.model_action_dim cannot be narrower than source.action_dim.")
    if result.seed != 0 or result.sample_steps != 10:
        raise ValueError("source seed/sample_steps must be 0/10.")
    if not result.camera_map or set(result.camera_map) - {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}:
        raise ValueError("source.camera_map must use valid Pi0.5 visual camera slots.")
    return result


def _keys(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a sequence of strings.")
    keys = tuple(value)
    if keys not in (TACTILE_KEYS, RIGHT_TACTILE_KEYS):
        raise ValueError(f"{name} must use canonical tactile order.")
    return keys


def _from_mapping(raw: Mapping[str, Any], config_path: Path) -> DeploymentConfig:
    config_path = config_path.expanduser().resolve()
    base = config_path.parent
    source = _source(_mapping(_required(raw, "source", "configuration"), "source"))
    norm_raw = _mapping(_required(raw, "norm_stats", "configuration"), "norm_stats")
    tactile_raw = _mapping(_required(raw, "tactile_encoder", "configuration"), "tactile_encoder")
    decoder_raw = _mapping(_required(raw, "direct_decoder", "configuration"), "direct_decoder")
    connection_raw = _mapping(_required(raw, "connection", "configuration"), "connection")
    observation_raw = _mapping(_required(raw, "observation", "configuration"), "observation")
    control_raw = _mapping(_required(raw, "control", "configuration"), "control")
    runtime_raw = _mapping(_required(raw, "runtime", "configuration"), "runtime")
    logging_raw = _mapping(_required(raw, "logging", "configuration"), "logging")
    norm_stats = NormStatsConfig(_path(_required(norm_raw, "dir", "norm_stats"), "norm_stats.dir"), _string(_required(norm_raw, "asset_id", "norm_stats"), "norm_stats.asset_id"), _boolean(_required(norm_raw, "use_quantile_norm", "norm_stats"), "norm_stats.use_quantile_norm"))
    tactile = TactileEncoderConfig(_path(_required(tactile_raw, "checkpoint", "tactile_encoder"), "tactile_encoder.checkpoint"), _positive(_required(tactile_raw, "embedding_dim", "tactile_encoder"), "tactile_encoder.embedding_dim"), _keys(_required(tactile_raw, "tactile_keys", "tactile_encoder"), "tactile_encoder.tactile_keys"))
    key_map = dict(_mapping(tactile_raw.get("key_map", {key: key for key in tactile.tactile_keys}), "tactile_encoder.key_map"))
    if set(key_map) != set(tactile.tactile_keys) or any(not isinstance(value, str) or value not in TACTILE_KEYS for value in key_map.values()) or len(set(key_map.values())) != len(key_map):
        raise ValueError("tactile_encoder.key_map must map each model tactile key to a unique server tactile key.")
    tactile = replace(tactile, key_map=key_map)
    decoder = DirectDecoderDeploymentConfig(_path(_required(decoder_raw, "checkpoint", "direct_decoder"), "direct_decoder.checkpoint"), _string(decoder_raw.get("device", "cpu"), "direct_decoder.device"), _positive(_required(decoder_raw, "action_horizon", "direct_decoder"), "direct_decoder.action_horizon"), _positive(_required(decoder_raw, "action_dim", "direct_decoder"), "direct_decoder.action_dim"), _positive(_required(decoder_raw, "tactile_dim", "direct_decoder"), "direct_decoder.tactile_dim"), _positive(_required(decoder_raw, "d_model", "direct_decoder"), "direct_decoder.d_model"), _positive(_required(decoder_raw, "nhead", "direct_decoder"), "direct_decoder.nhead"), _positive(_required(decoder_raw, "num_layers", "direct_decoder"), "direct_decoder.num_layers"), _positive(_required(decoder_raw, "dim_feedforward", "direct_decoder"), "direct_decoder.dim_feedforward"), _number(_required(decoder_raw, "dropout", "direct_decoder"), "direct_decoder.dropout"), _keys(_required(decoder_raw, "tactile_keys", "direct_decoder"), "direct_decoder.tactile_keys"))
    connection = ConnectionConfig(_string(_required(connection_raw, "address", "connection"), "connection.address"), _positive(_required(connection_raw, "port", "connection"), "connection.port"), _positive_number(connection_raw.get("retry_interval_s", 1.0), "connection.retry_interval_s"), _positive_number(connection_raw.get("ping_interval_s", 20.0), "connection.ping_interval_s"), _positive_number(connection_raw.get("ping_timeout_s", 20.0), "connection.ping_timeout_s"), _positive_number(connection_raw.get("observation_timeout_s", 30.0), "connection.observation_timeout_s"), _positive_number(_required(connection_raw, "action_ack_timeout_s", "connection"), "connection.action_ack_timeout_s"), None if connection_raw.get("token") is None else _string(connection_raw["token"], "connection.token"), _string(connection_raw.get("token_env", "VB_ROBOT_TOKEN"), "connection.token_env"), _boolean(connection_raw.get("require_token", False), "connection.require_token"))
    observation = ObservationConfig(_string(_required(observation_raw, "data_type", "observation"), "observation.data_type"), _string(_required(observation_raw, "language_prompt", "observation"), "observation.language_prompt"), _boolean(_required(observation_raw, "single_arm_mode", "observation"), "observation.single_arm_mode"), _boolean(_required(observation_raw, "no_state_obs_mode", "observation"), "observation.no_state_obs_mode"))
    control = ControlConfig(_positive_number(_required(control_raw, "control_frequency", "control"), "control.control_frequency"), _positive_number(_required(control_raw, "controller_frequency", "control"), "control.controller_frequency"), _positive(_required(control_raw, "action_horizon", "control"), "control.action_horizon"), _positive(_required(control_raw, "steps_per_inference", "control"), "control.steps_per_inference"), _positive_number(_required(control_raw, "max_normalized_action_abs", "control"), "control.max_normalized_action_abs"), _positive_number(_required(control_raw, "max_normalized_delta_rms", "control"), "control.max_normalized_delta_rms"))
    runtime = RuntimeConfig(_boolean(runtime_raw.get("auto_start", False), "runtime.auto_start"), _positive(runtime_raw.get("warmup_runs", 1), "runtime.warmup_runs"), _integer(runtime_raw.get("max_iterations", 0), "runtime.max_iterations"))
    logging = LoggingConfig(_boolean(logging_raw.get("save_observations", False), "logging.save_observations"), _path(logging_raw.get("output_dir", "outputs/pi05_direct"), "logging.output_dir"), _positive(logging_raw.get("save_every", 1), "logging.save_every"), _positive(logging_raw.get("queue_size", 32), "logging.queue_size"))
    if tactile.embedding_dim != 512:
        raise ValueError("tactile_encoder.embedding_dim must be 512.")
    if (decoder.action_horizon, decoder.tactile_dim, decoder.d_model, decoder.nhead, decoder.num_layers, decoder.dim_feedforward, decoder.dropout) != (50, 512, 128, 4, 2, 256, 0.1) or decoder.action_dim != source.action_dim or decoder.tactile_keys != tactile.tactile_keys:
        raise ValueError("direct_decoder dimensions and tactile keys must match the source/encoder contract.")
    if control.action_horizon != source.action_horizon or not 1 <= control.steps_per_inference <= 50:
        raise ValueError("control action horizon/steps must match the fixed source contract.")
    if observation.data_type != "vitac" or observation.no_state_obs_mode:
        raise ValueError("observation must be vitac with state enabled.")
    if observation.single_arm_mode != (source.action_dim == 10):
        raise ValueError("observation.single_arm_mode must be true for right-arm 10D and false for bimanual 20D.")
    if runtime.max_iterations < 0:
        raise ValueError("runtime.max_iterations must be non-negative.")
    source = replace(source, checkpoint=_resolved_path(source.checkpoint, base))
    norm_stats = replace(norm_stats, directory=_resolved_path(norm_stats.directory, base))
    tactile = replace(tactile, checkpoint=_resolved_path(tactile.checkpoint, base))
    decoder = replace(decoder, checkpoint=_resolved_path(decoder.checkpoint, base))
    return DeploymentConfig(source, norm_stats, tactile, decoder, connection, observation, control, runtime, logging, config_path)


def load_deployment_config(path: Path) -> DeploymentConfig:
    """Parse a deployment YAML without importing JAX or PyTorch."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    return _from_mapping(_mapping(yaml.safe_load(path.read_text(encoding="utf-8")) or {}, "configuration"), path)


def make_server_config(config: DeploymentConfig) -> dict[str, Any]:
    """Build the exact robot-server config for direct tactile scheduling."""
    return {
        "data_type": config.observation.data_type,
        "language_prompt": config.observation.language_prompt,
        "control_frequency": config.control.control_frequency,
        "controller_frequency": config.control.controller_frequency,
        # The hardware wire stays bimanual; right-arm projection is client-side.
        "single_arm_mode": False,
        "no_state_obs_mode": config.observation.no_state_obs_mode,
        "steps_per_inference": config.control.steps_per_inference,
        "action_horizon": config.control.action_horizon,
        "observation_profile": "pi05_vitac_224",
        "execution_protocol": "frs_steering_v1",
        "steering_protection_interval_s": None,
        "frs_tactile_keys": [getattr(config.tactile_encoder, "key_map", {}).get(key, key) for key in config.tactile_encoder.tactile_keys],
    }


def _readable_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory is missing: {path}")
    try:
        with os.scandir(path) as entries:
            next(entries, None)
    except OSError as error:
        raise OSError(f"{label} directory is not readable: {path}") from error


def _readable_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        with path.open("rb") as file:
            file.read(1)
    except OSError as error:
        raise OSError(f"{label} is not readable: {path}") from error


def preflight_deployment_assets(config: DeploymentConfig) -> None:
    """Check deployment asset presence/readability without JAX, Torch, or network access."""
    _readable_directory(config.source.checkpoint, "source checkpoint")
    _readable_directory(config.source.checkpoint / "params", "source checkpoint params")
    _readable_file(
        config.norm_stats.directory / config.norm_stats.asset_id / "norm_stats.json",
        "norm stats norm_stats.json",
    )

    encoder_directory = config.tactile_encoder.checkpoint
    metadata_path = encoder_directory / "checkpoint.json"
    _readable_file(metadata_path, "tactile encoder checkpoint.json")
    try:
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"tactile encoder checkpoint.json is invalid: {metadata_path}") from error
    if not isinstance(metadata, Mapping):
        raise ValueError(f"tactile encoder checkpoint.json must contain a mapping: {metadata_path}")
    params_file = metadata.get("params_file", "params.npz")
    if not isinstance(params_file, str) or not params_file.strip() or "\x00" in params_file:
        raise ValueError("tactile encoder params_file must be a non-empty relative path")
    relative_params = Path(params_file)
    if relative_params.is_absolute():
        raise ValueError("tactile encoder params_file must remain within the checkpoint directory")
    params_path = (encoder_directory / relative_params).resolve()
    try:
        params_path.relative_to(encoder_directory)
    except ValueError as error:
        raise ValueError(
            "tactile encoder params_file must remain within the checkpoint directory"
        ) from error
    _readable_file(params_path, f"tactile encoder parameter file {params_file}")
    _readable_file(config.direct_decoder.checkpoint, "direct decoder best.pt")


def expected_source_contract(config: DeploymentConfig) -> dict[str, object]:
    """Return the resolved source identity required from a training checkpoint."""
    return {
        "pi": {"checkpoint": str(config.source.checkpoint), "norm_stats_dir": str(config.norm_stats.directory), "norm_stats_asset_id": config.norm_stats.asset_id, "variant": {"paligemma": config.source.paligemma_variant, "action_expert": config.source.action_expert_variant}, "model_action_width": config.source.model_action_dim, "sample_steps": config.source.sample_steps},
        "encoder": {"checkpoint": str(config.tactile_encoder.checkpoint), "key_order": list(config.tactile_encoder.tactile_keys)},
    }
