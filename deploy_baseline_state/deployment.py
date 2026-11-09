"""Dependency-light configuration for direct Pi0.5 deployment."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml


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
    reverse_steps: int = 50
    reverse_solver: str = "slerpflow"


@dataclass(frozen=True)
class NormStatsConfig:
    directory: Path
    asset_id: str
    use_quantile_norm: bool


@dataclass(frozen=True)
class StateDecoderDeploymentConfig:
    checkpoint: Path
    device: str
    action_horizon: int
    action_dim: int
    state_dim: int
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.0
    decode_steps: int = 10
    decode_solver: str = "fireflow"


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
    decoder: StateDecoderDeploymentConfig
    connection: ConnectionConfig
    observation: ObservationConfig
    control: ControlConfig
    runtime: RuntimeConfig
    logging: LoggingConfig
    config_path: Path

    @property
    def direct_decoder(self):
        return self.decoder

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
        action_expert_variant=_string(
            _required(raw, "action_expert_variant", "source"), "source.action_expert_variant"
        ),
        camera_map={
            _string(k, "source.camera_map key"): _string(v, "source.camera_map value")
            for k, v in camera_map.items()
        },
        reverse_steps=_positive(raw.get("reverse_steps", 50), "source.reverse_steps"),
        reverse_solver=_string(raw.get("reverse_solver", "slerpflow"), "source.reverse_solver"),
    )
    if result.action_horizon != 50 or (result.action_dim, result.state_dim) not in ((10, 7), (20, 20)):
        raise ValueError("source must use horizon=50 and action/state dimensions 10/7 or 20/20.")
    if result.model_action_dim < result.action_dim:
        raise ValueError("source.model_action_dim cannot be narrower than source.action_dim.")
    if result.seed != 0 or result.sample_steps != 10:
        raise ValueError("source seed/sample_steps must be 0/10.")
    if not result.camera_map or set(result.camera_map) - {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }:
        raise ValueError("source.camera_map must use valid Pi0.5 visual camera slots.")
    if result.reverse_solver not in ("euler", "fireflow", "slerpflow"):
        raise ValueError("unsupported source.reverse_solver")
    return result


def _from_mapping(raw, config_path):
    config_path = config_path.resolve()
    base = config_path.parent
    if "tactile_encoder" in raw:
        raise ValueError("state-only deployment has no tactile encoder")
    source = _source(_mapping(raw["source"], "source"))
    source = replace(source, checkpoint=_resolved_path(source.checkpoint, base))
    norm_raw = raw["norm_stats"]
    norm = NormStatsConfig(
        _resolved_path(_path(norm_raw["dir"], "norm_stats.dir"), base),
        _string(norm_raw["asset_id"], "norm_stats.asset_id"),
        _boolean(norm_raw["use_quantile_norm"], "norm_stats.use_quantile_norm"),
    )
    d = _mapping(raw["decoder"], "decoder")
    if any("tactile" in key for key in d):
        raise ValueError("state-only decoder cannot configure tactile inputs")
    decoder = StateDecoderDeploymentConfig(
        _resolved_path(_path(d["checkpoint"], "decoder.checkpoint"), base),
        _string(d.get("device", "cuda"), "decoder.device"),
        _positive(d.get("action_horizon", 50), "decoder.action_horizon"),
        _positive(d.get("action_dim", source.action_dim), "decoder.action_dim"),
        _positive(d.get("state_dim", source.state_dim), "decoder.state_dim"),
        _positive(d.get("d_model", 256), "decoder.d_model"),
        _positive(d.get("nhead", 4), "decoder.nhead"),
        _positive(d.get("num_layers", 6), "decoder.num_layers"),
        _positive(d.get("dim_feedforward", 1024), "decoder.dim_feedforward"),
        _number(d.get("dropout", 0.0), "decoder.dropout"),
        _positive(d.get("decode_steps", 10), "decoder.decode_steps"),
        _string(d.get("decode_solver", "fireflow"), "decoder.decode_solver"),
    )
    if (decoder.action_horizon, decoder.action_dim, decoder.state_dim) != (
        50,
        source.action_dim,
        source.state_dim,
    ):
        raise ValueError("decoder dimensions must match source profile")
    if (
        decoder.d_model,
        decoder.nhead,
        decoder.num_layers,
        decoder.dim_feedforward,
        decoder.dropout,
        decoder.decode_steps,
        decoder.decode_solver,
    ) != (256, 4, 6, 1024, 0.0, 10, "fireflow"):
        raise ValueError("decoder must match the state-only TacFRS backbone")
    c = raw["connection"]
    connection = ConnectionConfig(
        _string(c["address"], "connection.address"),
        _positive(c["port"], "connection.port"),
        *[
            _positive_number(c.get(key, default), "connection." + key)
            for key, default in [
                ("retry_interval_s", 1.0),
                ("ping_interval_s", 20.0),
                ("ping_timeout_s", 20.0),
                ("observation_timeout_s", 30.0),
                ("action_ack_timeout_s", 30.0),
            ]
        ],
        c.get("token"),
        _string(c.get("token_env", "VB_ROBOT_TOKEN"), "connection.token_env"),
        _boolean(c.get("require_token", True), "connection.require_token"),
    )
    o = raw["observation"]
    observation = ObservationConfig(
        _string(o.get("data_type", "vision"), "observation.data_type"),
        _string(o["language_prompt"], "observation.language_prompt"),
        _boolean(o["single_arm_mode"], "observation.single_arm_mode"),
        _boolean(o.get("no_state_obs_mode", False), "observation.no_state_obs_mode"),
    )
    if observation.no_state_obs_mode or observation.data_type not in ("vision", "vitac"):
        raise ValueError("state-only deployment requires state and visual VLA observations")
    if observation.single_arm_mode != (source.action_dim == 10):
        raise ValueError("single_arm_mode does not match source profile")
    c = raw["control"]
    control = ControlConfig(
        *[
            _positive_number(c[key], "control." + key)
            for key in ("control_frequency", "controller_frequency")
        ],
        _positive(c["action_horizon"], "control.action_horizon"),
        _positive(c["steps_per_inference"], "control.steps_per_inference"),
        _positive_number(c["max_normalized_action_abs"], "control.max_normalized_action_abs"),
        _positive_number(c["max_normalized_delta_rms"], "control.max_normalized_delta_rms"),
    )
    if control.action_horizon != 50 or not 1 <= control.steps_per_inference <= 50:
        raise ValueError("invalid control horizon")
    r = raw.get("runtime", {})
    runtime = RuntimeConfig(
        _boolean(r.get("auto_start", False), "runtime.auto_start"),
        _positive(r.get("warmup_runs", 1), "runtime.warmup_runs"),
        _integer(r.get("max_iterations", 0), "runtime.max_iterations"),
    )
    if runtime.max_iterations < 0:
        raise ValueError("runtime.max_iterations must be nonnegative")
    log = raw.get("logging", {})
    logging = LoggingConfig(
        _boolean(log.get("save_observations", False), "logging.save_observations"),
        _resolved_path(_path(log.get("output_dir", "outputs/baseline_state"), "logging.output_dir"), base),
        _positive(log.get("save_every", 1), "logging.save_every"),
        _positive(log.get("queue_size", 32), "logging.queue_size"),
    )
    return DeploymentConfig(
        source, norm, decoder, connection, observation, control, runtime, logging, config_path
    )


def load_deployment_config(path):
    path = Path(path)
    return _from_mapping(_mapping(yaml.safe_load(path.read_text()), "configuration"), path)


def make_server_config(config):
    return {
        "data_type": config.observation.data_type,
        "language_prompt": config.observation.language_prompt,
        "control_frequency": config.control.control_frequency,
        "controller_frequency": config.control.controller_frequency,
        "single_arm_mode": False,
        "no_state_obs_mode": False,
        "steps_per_inference": config.control.steps_per_inference,
        "action_horizon": 50,
        "observation_profile": "pi05_vitac_224",
        "execution_protocol": "frs_steering_v1",
        "steering_protection_interval_s": None,
        "frs_tactile_keys": [],
    }


def preflight_deployment_assets(config):
    for path, label in [
        (config.source.checkpoint, "source checkpoint"),
        (config.source.checkpoint / "params", "source params"),
    ]:
        if not path.is_dir() or not os.access(path, os.R_OK):
            raise FileNotFoundError(f"{label} directory is missing: {path}")
    for path, label in [
        (config.norm_stats.directory / config.norm_stats.asset_id / "norm_stats.json", "norm stats"),
        (config.decoder.checkpoint, "state decoder best.pt"),
    ]:
        if not path.is_file() or not os.access(path, os.R_OK):
            raise FileNotFoundError(f"{label} is missing: {path}")


def expected_source_contract(config):
    return {
        "pi": {
            "checkpoint": str(config.source.checkpoint),
            "norm_stats_dir": str(config.norm_stats.directory),
            "norm_stats_asset_id": config.norm_stats.asset_id,
            "variant": {
                "paligemma": config.source.paligemma_variant,
                "action_expert": config.source.action_expert_variant,
            },
            "model_action_width": config.source.model_action_dim,
            "sample_steps": config.source.sample_steps,
            "use_quantile_norm": config.norm_stats.use_quantile_norm,
        },
        "flow": {
            "origin": "structured_noise",
            "conditioning": "state_only",
            "reverse_steps": config.source.reverse_steps,
            "reverse_solver": config.source.reverse_solver,
        },
    }
