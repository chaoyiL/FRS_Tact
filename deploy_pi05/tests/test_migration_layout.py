"""Contract tests for the self-contained Pi0.5 deployment entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest
import yaml

from deploy_pi05.frs_config import validate_frs_config_section


DEPLOY_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (DEPLOY_ROOT / relative_path).read_text(encoding="utf-8")


def _load_config(name: str) -> dict[str, object]:
    payload = yaml.safe_load(_read(f"configs/{name}"))
    assert isinstance(payload, dict)
    return payload


def test_mode_specific_wrappers_and_configs_exist() -> None:
    assert (DEPLOY_ROOT / "scripts/start_pi05.sh").is_file()
    assert (DEPLOY_ROOT / "scripts/start_pi05_frs.sh").is_file()
    assert (DEPLOY_ROOT / "scripts/start_remote_client.sh").is_file()
    assert (DEPLOY_ROOT / "configs/deploy_pi05.yaml").is_file()
    assert (DEPLOY_ROOT / "configs/deploy_pi05_frs.yaml").is_file()


def test_mode_configs_are_standalone_and_select_the_required_observations() -> None:
    plain = _load_config("deploy_pi05.yaml")
    frs = _load_config("deploy_pi05_frs.yaml")

    assert "profiles" not in plain
    assert "profiles" not in frs
    assert plain["observation"]["data_type"] == "vision"
    assert frs["observation"]["data_type"] == "vitac"
    assert "frs" not in plain
    assert frs["frs"]["enabled"] is True


def test_wrappers_choose_their_mode_specific_config_and_mode() -> None:
    plain = _read("scripts/start_pi05.sh")
    frs = _read("scripts/start_pi05_frs.sh")

    assert "${DEPLOY_ROOT}/configs/deploy_pi05.yaml" in plain
    assert "--mode pi05" in plain
    assert "${DEPLOY_ROOT}/configs/deploy_pi05_frs.yaml" in frs
    assert "--mode frs" in frs


def test_shared_launcher_uses_renamed_entrypoints_and_private_environment() -> None:
    launcher = _read("scripts/start_remote_client.sh")

    assert "deploy_pi05.pi05_client" in launcher
    assert "deploy_pi05.remote_client" in launcher
    assert "deploy_pi05_frs" not in launcher
    assert '${DEPLOY_ROOT}/.venv/bin/python' in launcher
    assert 'PYTHONPATH="${DEPLOY_ROOT}/src:${DEPLOY_ROOT}:${ROOT}' in launcher
    assert launcher.index('cd "${DEPLOY_ROOT}"') < launcher.index('exec "${PYTHON_BIN}"')


def test_frs_wrapper_preserves_pi05_deploy_config_precedence() -> None:
    environment = os.environ | {
        "VB_ROBOT_TOKEN": "layout-check",
        "PI05_DEPLOY_CONFIG": str(DEPLOY_ROOT / "configs/deploy_pi05.yaml"),
        "PI05_FRS_DEPLOY_CONFIG": str(DEPLOY_ROOT / "configs/deploy_pi05_frs.yaml"),
    }
    result = subprocess.run(
        ["bash", str(DEPLOY_ROOT / "scripts/start_pi05_frs.sh"), "--check"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert f"config={DEPLOY_ROOT / 'configs/deploy_pi05.yaml'}" in result.stdout


def test_frs_config_error_names_the_migrated_package() -> None:
    with pytest.raises(ValueError, match="deploy_pi05 requires frs.enabled=true"):
        validate_frs_config_section({"frs": {"enabled": False}})
