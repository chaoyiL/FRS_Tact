"""Focused compatibility test for the legacy FRS client configuration hook."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from deploy_pi05 import deployment


ROOT = Path(__file__).resolve().parents[2]


def _import_remote_client(monkeypatch):
    """Import the config hook without the optional robot/JAX runtime stack."""
    bridge_module = types.ModuleType("deploy_pi05.bridge_client")
    bridge_module.RobotBridgeClient = object
    protocol_module = types.ModuleType("deploy_pi05.frs_protocol")
    for name in ("FRSChunkEnd", "FRSChunkStart", "FRSSteerAck", "FRSSteerRequest"):
        setattr(protocol_module, name, type(name, (), {}))
    runtime_module = types.ModuleType("deploy_pi05.frs_runtime")
    runtime_module.FRSChunkReady = object
    runtime_module.FRSRuntime = object
    runtime_module.FRSSteerResult = object
    runtime_module.validate_frs_config_section = lambda config: None
    policy_module = types.ModuleType("deploy_pi05.policy")
    policy_module.Pi05DeploymentConfig = object
    policy_module.Pi05RemotePolicy = object

    monkeypatch.setitem(sys.modules, "deploy_pi05.bridge_client", bridge_module)
    monkeypatch.setitem(sys.modules, "deploy_pi05.frs_protocol", protocol_module)
    monkeypatch.setitem(sys.modules, "deploy_pi05.frs_runtime", runtime_module)
    monkeypatch.setitem(sys.modules, "deploy_pi05.policy", policy_module)
    monkeypatch.delitem(sys.modules, "deploy_pi05.remote_client", raising=False)
    return importlib.import_module("deploy_pi05.remote_client")


def test_compat_load_config_selects_frs_profile(monkeypatch) -> None:
    remote_client = _import_remote_client(monkeypatch)

    config = remote_client.load_config(ROOT / "configs" / "deploy_pi05_frs.yaml")

    assert config["observation"]["data_type"] == "vitac"
    assert config["logging"]["output_dir"] == "outputs/pi05_frs_observations"


def test_compat_load_config_delegates_to_frs_profile_loader(monkeypatch, tmp_path: Path) -> None:
    remote_client = _import_remote_client(monkeypatch)
    config_path = tmp_path / "deploy.yaml"
    expected = {"observation": {"data_type": "vitac"}}
    calls: list[tuple[Path, str]] = []

    def load_deployment_config(path: Path, mode: str) -> dict:
        calls.append((path, mode))
        return expected

    monkeypatch.setattr(remote_client, "load_deployment_config", load_deployment_config)

    assert remote_client.load_config(config_path) == expected
    assert calls == [(config_path, "frs")]


def test_default_config_uses_shared_frs_profile(monkeypatch) -> None:
    remote_client = _import_remote_client(monkeypatch)

    expected_path = ROOT / "configs" / "deploy_pi05_frs.yaml"

    assert remote_client.DEFAULT_CONFIG == expected_path
    assert remote_client.DEFAULT_CONFIG.is_file()
    config = remote_client.load_config(remote_client.DEFAULT_CONFIG)
    assert config["observation"]["data_type"] == "vitac"


def test_remote_client_uses_shared_deployment_helpers(monkeypatch) -> None:
    remote_client = _import_remote_client(monkeypatch)

    assert remote_client.ObservationSaver is deployment.ObservationSaver
    assert remote_client.make_policy_config is deployment.make_policy_config
    assert remote_client.resolve_token is deployment.resolve_token
    assert remote_client.optional_bool is deployment.optional_bool
    assert remote_client.prepare_observation is deployment.prepare_observation
    assert remote_client.make_server_config is deployment.make_server_config
    assert remote_client.section is deployment.section
