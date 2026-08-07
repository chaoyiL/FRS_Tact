from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "deploy_smolvla" / "start_vtsmolvla.sh"
CLIENT = ROOT / "deploy_smolvla" / "scripts" / "run_client.sh"
DEFAULT_CONFIG = ROOT / "deploy_smolvla" / "configs" / "deploy_smolvla_jax.yaml"
DEFAULT_MODEL_CACHE = ROOT / "checkpoints" / "model"


def _copy_deploy_entry_points(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    deploy_dir = project / "deploy_smolvla"
    scripts_dir = deploy_dir / "scripts"
    config_dir = deploy_dir / "configs"
    scripts_dir.mkdir(parents=True)
    config_dir.mkdir()
    shutil.copy2(LAUNCHER, deploy_dir / LAUNCHER.name)
    shutil.copy2(CLIENT, scripts_dir / CLIENT.name)
    shutil.copy2(DEFAULT_CONFIG, config_dir / DEFAULT_CONFIG.name)
    return project


def _fake_python(tmp_path: Path) -> Path:
    python = tmp_path / "fake-python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ -n "${FAKE_PYTHON_MARKER:-}" ]]; then\n'
        '    : >"${FAKE_PYTHON_MARKER}"\n'
        "fi\n"
        "printf '%s\\n' \"${HF_HUB_CACHE}\"\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    return python


def _run_client(
    project: Path,
    python: Path,
    *,
    hub_cache: Path | None = None,
    marker: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["FRS_PYTHON"] = str(python)
    env.pop("HF_HUB_CACHE", None)
    env.pop("HUGGINGFACE_HUB_CACHE", None)
    if hub_cache is not None:
        env["HF_HUB_CACHE"] = str(hub_cache)
    if marker is not None:
        env["FAKE_PYTHON_MARKER"] = str(marker)
    return subprocess.run(
        ["bash", str(project / "deploy_smolvla" / "scripts" / CLIENT.name)],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_check(
    *, token_file: Path, token: str | None = None, hub_cache: Path | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["VB3_TOKEN_FILE"] = str(token_file)
    env.pop("HF_HUB_CACHE", None)
    env.pop("HUGGINGFACE_HUB_CACHE", None)
    if token is None:
        env.pop("VB_ROBOT_TOKEN", None)
    else:
        env["VB_ROBOT_TOKEN"] = token
    if hub_cache is not None:
        env["HF_HUB_CACHE"] = str(hub_cache)
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


def test_launcher_uses_project_model_cache_by_default(tmp_path: Path) -> None:
    result = _run_check(token_file=tmp_path / "missing", token="secret")

    assert result.returncode == 0, result.stderr
    assert f"model_cache={DEFAULT_MODEL_CACHE}" in result.stdout


def test_launcher_preserves_explicit_hf_hub_cache(tmp_path: Path) -> None:
    cache = tmp_path / "hub"

    result = _run_check(token_file=tmp_path / "missing", token="secret", hub_cache=cache)

    assert result.returncode == 0, result.stderr
    assert f"model_cache={cache}" in result.stdout
    assert cache.is_dir()


def test_run_client_uses_project_model_cache_by_default(tmp_path: Path) -> None:
    project = _copy_deploy_entry_points(tmp_path)
    result = _run_client(project, _fake_python(tmp_path))

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{project / 'checkpoints' / 'model'}\n"
    assert (project / "checkpoints" / "model").is_dir()
    assert (project / "checkpoints" / "encoder").is_dir()


def test_launcher_creates_checkpoint_directories_in_fresh_project(tmp_path: Path) -> None:
    project = _copy_deploy_entry_points(tmp_path)
    env = os.environ.copy()
    env["VB_ROBOT_TOKEN"] = "secret"
    env.pop("HF_HUB_CACHE", None)
    env.pop("HUGGINGFACE_HUB_CACHE", None)

    result = subprocess.run(
        ["bash", str(project / "deploy_smolvla" / LAUNCHER.name), "--check"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (project / "checkpoints" / "model").is_dir()
    assert (project / "checkpoints" / "encoder").is_dir()


def test_launcher_keeps_existing_checkpoint_files_unchanged(tmp_path: Path) -> None:
    project = _copy_deploy_entry_points(tmp_path)
    model_sentinel = project / "checkpoints" / "model" / "model.sentinel"
    encoder_sentinel = project / "checkpoints" / "encoder" / "encoder.sentinel"
    model_sentinel.parent.mkdir(parents=True)
    encoder_sentinel.parent.mkdir(parents=True)
    model_sentinel.write_text("existing model cache\n", encoding="utf-8")
    encoder_sentinel.write_text("existing encoder checkpoint\n", encoding="utf-8")
    env = os.environ.copy()
    env["VB_ROBOT_TOKEN"] = "secret"
    env.pop("HF_HUB_CACHE", None)
    env.pop("HUGGINGFACE_HUB_CACHE", None)

    result = subprocess.run(
        ["bash", str(project / "deploy_smolvla" / LAUNCHER.name), "--check"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert model_sentinel.read_text(encoding="utf-8") == "existing model cache\n"
    assert encoder_sentinel.read_text(encoding="utf-8") == "existing encoder checkpoint\n"


def test_run_client_preserves_explicit_hf_hub_cache(tmp_path: Path) -> None:
    project = _copy_deploy_entry_points(tmp_path)
    cache = tmp_path / "custom-cache" / "hub"

    result = _run_client(project, _fake_python(tmp_path), hub_cache=cache)

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{cache}\n"
    assert cache.is_dir()


def test_run_client_fails_before_python_when_hf_hub_cache_is_a_file(
    tmp_path: Path,
) -> None:
    project = _copy_deploy_entry_points(tmp_path)
    cache = tmp_path / "hub-cache-file"
    marker = tmp_path / "python-started"
    cache.write_text("not a directory\n", encoding="utf-8")

    result = _run_client(
        project,
        _fake_python(tmp_path),
        hub_cache=cache,
        marker=marker,
    )

    assert result.returncode != 0
    assert not marker.exists()


def test_launcher_help_documents_project_local_hf_hub_cache(tmp_path: Path) -> None:
    project = _copy_deploy_entry_points(tmp_path)
    env = os.environ.copy()
    env.pop("HF_HUB_CACHE", None)
    env.pop("HUGGINGFACE_HUB_CACHE", None)

    result = subprocess.run(
        ["bash", str(project / "deploy_smolvla" / LAUNCHER.name), "--help"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "HF_HUB_CACHE" in result.stdout
    assert "<project>/checkpoints/model" in result.stdout


def test_checkpoint_gitignore_is_root_anchored() -> None:
    for path in ("checkpoints/model", "checkpoints/encoder"):
        result = subprocess.run(
            ["git", "check-ignore", "--verbose", "--no-index", "--", path],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, path
        assert "/checkpoints/" in result.stdout

    nested = subprocess.run(
        [
            "git",
            "check-ignore",
            "--verbose",
            "--no-index",
            "--",
            "somewhere/checkpoints/model",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert nested.returncode == 1
    assert nested.stdout == ""
