"""Black-box tests for the pi05 deployment shell entrypoints."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = "deploy_pi05_frs/scripts"
CONFIG = "deploy_pi05_frs/configs/deploy_pi05.yaml"


def _env(**overrides: str) -> dict[str, str]:
    return {**os.environ, "VB_ROBOT_TOKEN": "redacted", **overrides}


def _check(script: str, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["bash", str(ROOT / script), "--check", *args],
        cwd=ROOT,
        env=env or _env(),
        text=True,
        capture_output=True,
        check=True,
    )
    assert "redacted" not in result.stdout
    assert "redacted" not in result.stderr
    return result.stdout


def test_plain_wrapper_selects_plain_entrypoint() -> None:
    output = _check(f"{SCRIPTS}/start_pi05.sh")

    assert "mode=pi05" in output
    assert "entrypoint=deploy_pi05_frs.pi05_client" in output
    assert "configs/deploy_pi05.yaml" in output


def test_frs_wrapper_selects_frs_entrypoint() -> None:
    output = _check(f"{SCRIPTS}/start_pi05_frs.sh")

    assert "mode=frs" in output
    assert "entrypoint=deploy_pi05_frs.remote_client" in output


def test_direct_launcher_defaults_to_frs_mode() -> None:
    output = _check(f"{SCRIPTS}/start_remote_client.sh", "--config", CONFIG)

    assert "mode=frs" in output
    assert "entrypoint=deploy_pi05_frs.remote_client" in output


def test_shared_config_override_has_precedence_for_both_wrappers(tmp_path: Path) -> None:
    config = tmp_path / "shared.yaml"
    config.write_text("profiles: {}\n", encoding="utf-8")
    env = _env(
        PI05_DEPLOY_CONFIG=str(config),
        PI05_FRS_DEPLOY_CONFIG=str(tmp_path / "legacy.yaml"),
    )

    assert f"config={config}" in _check(f"{SCRIPTS}/start_pi05.sh", env=env)
    assert f"config={config}" in _check(f"{SCRIPTS}/start_pi05_frs.sh", env=env)


def test_frs_wrapper_uses_legacy_config_override_as_fallback(tmp_path: Path) -> None:
    config = tmp_path / "legacy.yaml"
    config.write_text("profiles: {}\n", encoding="utf-8")

    output = _check(
        f"{SCRIPTS}/start_pi05_frs.sh",
        env=_env(PI05_FRS_DEPLOY_CONFIG=str(config)),
    )

    assert f"config={config}" in output


@pytest.mark.parametrize(
    ("mode", "expected_python"),
    [("pi05", "plain-python"), ("frs", "frs-python")],
)
def test_mode_specific_python_override_has_precedence(mode: str, expected_python: str) -> None:
    output = _check(
        f"{SCRIPTS}/start_remote_client.sh",
        "--mode",
        mode,
        "--config",
        CONFIG,
        env=_env(
            PI05_PYTHON="plain-python",
            PI05_FRS_PYTHON="frs-python",
            VB3_PYTHON="fallback-python",
        ),
    )

    assert f"python={expected_python}" in output


def test_launcher_rejects_invalid_mode() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / SCRIPTS / "start_remote_client.sh"), "--mode", "bad", "--config", CONFIG],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "Unsupported mode: bad" in result.stderr


def test_launcher_rejects_missing_max_iterations_value() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / SCRIPTS / "start_remote_client.sh"), "--config", CONFIG, "--max-iterations"],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "--max-iterations requires a value" in result.stderr


def test_wrapper_forwards_max_iterations_to_selected_module(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" >"${PI05_ARGS_FILE}"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    args_file = tmp_path / "args.txt"
    env = _env(PI05_PYTHON=str(fake_python), PI05_ARGS_FILE=str(args_file))

    subprocess.run(
        ["bash", str(ROOT / SCRIPTS / "start_pi05.sh"), "--max-iterations", "2"],
        cwd=ROOT,
        env=env,
        check=True,
    )

    assert args_file.read_text(encoding="utf-8").splitlines()[-2:] == ["--max-iterations", "2"]
