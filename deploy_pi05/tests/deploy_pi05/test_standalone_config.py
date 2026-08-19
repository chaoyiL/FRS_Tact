"""Standalone Pi0.5 deployment configuration contract."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from deploy_pi05.deployment import load_deployment_config


ROOT = Path(__file__).resolve().parents[2]
PLAIN_CONFIG = ROOT / "configs" / "deploy_pi05.yaml"
FRS_CONFIG = ROOT / "configs" / "deploy_pi05_frs.yaml"


def _raw(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_plain_config_is_standalone_vision_without_frs_or_profiles() -> None:
    config = _raw(PLAIN_CONFIG)

    assert config["observation"]["data_type"] == "vision"
    assert "frs" not in config
    assert "profiles" not in config


def test_frs_config_is_standalone_vitac_with_frs_and_without_profiles() -> None:
    config = _raw(FRS_CONFIG)

    assert config["observation"]["data_type"] == "vitac"
    assert "frs" in config
    assert "profiles" not in config


@pytest.mark.parametrize(
    ("path", "mode", "data_type", "output_dir"),
    [
        (PLAIN_CONFIG, "pi05", "vision", "outputs/pi05_observations"),
        (FRS_CONFIG, "frs", "vitac", "outputs/pi05_frs_observations"),
    ],
)
def test_matching_standalone_config_loads_without_value_injection(
    path: Path, mode: str, data_type: str, output_dir: str
) -> None:
    loaded = load_deployment_config(path, mode)

    assert loaded["observation"]["data_type"] == data_type
    assert loaded["logging"]["output_dir"] == output_dir
    assert loaded["checkpoint"] == "/home/typhon/ManiSkill-vitac/checkpoints/6000/"
    assert loaded["model"]["action_horizon"] == 50


@pytest.mark.parametrize(
    ("path", "mode", "expected_data_type"),
    [
        (PLAIN_CONFIG, "frs", "vitac"),
        (FRS_CONFIG, "pi05", "vision"),
    ],
)
def test_standalone_config_rejects_the_wrong_mode(
    path: Path, mode: str, expected_data_type: str
) -> None:
    with pytest.raises(ValueError, match=rf"observation\.data_type must be {expected_data_type!r}"):
        load_deployment_config(path, mode)
