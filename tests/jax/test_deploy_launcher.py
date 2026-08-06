from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "deploy_smolvla" / "start_vtsmolvla.sh"
DEFAULT_CONFIG = ROOT / "configs" / "deploy_smolvla_jax.yaml"


def _run_check(*, token_file: Path, token: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["VB3_TOKEN_FILE"] = str(token_file)
    if token is None:
        env.pop("VB_ROBOT_TOKEN", None)
    else:
        env["VB_ROBOT_TOKEN"] = token
    return subprocess.run(
        ["bash", str(LAUNCHER), "--check"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_launcher_reads_token_file_without_printing_secret(tmp_path: Path) -> None:
    token_file = tmp_path / "token_list.txt"
    token_file.write_text("one-click-secret\n", encoding="utf-8")

    result = _run_check(token_file=token_file)

    assert result.returncode == 0, result.stderr
    assert f"config={DEFAULT_CONFIG}" in result.stdout
    assert f"token_source=file:{token_file}" in result.stdout
    assert "one-click-secret" not in result.stdout + result.stderr


def test_launcher_accepts_environment_token_without_token_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing-token-list.txt"

    result = _run_check(token_file=missing, token="environment-secret")

    assert result.returncode == 0, result.stderr
    assert "token_source=environment:VB_ROBOT_TOKEN" in result.stdout
    assert "environment-secret" not in result.stdout + result.stderr


def test_launcher_fails_before_start_when_token_is_unavailable(tmp_path: Path) -> None:
    missing = tmp_path / "missing-token-list.txt"

    result = _run_check(token_file=missing)

    assert result.returncode == 2
    assert "VB_ROBOT_TOKEN" in result.stderr
    assert str(missing) in result.stderr
