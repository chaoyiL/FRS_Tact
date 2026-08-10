from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SHARED_LAUNCHER = ROOT / "deploy_smolvla" / "scripts" / "start_remote_client.sh"
FRS_LAUNCHER = ROOT / "deploy_smolvla" / "scripts" / "start_frs.sh"
VT_LAUNCHER = ROOT / "deploy_smolvla" / "scripts" / "start_vtsmolvla.sh"
FRS_CONFIG = ROOT / "deploy_smolvla" / "configs" / "deploy_frs.yaml"
DEFAULT_CONFIG = ROOT / "deploy_smolvla" / "configs" / "deploy_smolvla_jax.yaml"
DEFAULT_MODEL_CACHE = ROOT / "checkpoints" / "model"


def test_policy_loader_selects_vt_policy_for_a_tactile_contract(monkeypatch, tmp_path) -> None:
    from deploy_smolvla import remote_client
    from train_vtsmolvla.validation import CheckpointContract

    selected = []
    monkeypatch.setattr(remote_client, "resolve_checkpoint", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(
        remote_client,
        "validate_checkpoint",
        lambda *args, **kwargs: type("Report", (), {"require_valid": lambda self: self})(),
    )
    monkeypatch.setattr(
        remote_client.VTJaxSmolVLAPolicy,
        "from_pretrained",
        lambda *args, **kwargs: selected.append("vt") or object(),
    )
    monkeypatch.setattr(
        remote_client.JaxSmolVLAPolicy,
        "from_pretrained",
        lambda *args, **kwargs: selected.append("visual") or object(),
    )
    expected = CheckpointContract(
        state_dim=20,
        action_dim=20,
        chunk_size=20,
        image_keys=("rgb",),
        tactile_keys=("touch",),
        tactile_num_tokens=1,
    )

    remote_client._load_validated_policy(
        "checkpoint",
        revision=None,
        allow_download=False,
        expected=expected,
        rename_map=None,
    )

    assert selected == ["vt"]


def test_visual_contract_loads_visual_policy_with_frozen_tactile_projection(
    monkeypatch, tmp_path
) -> None:
    from deploy_smolvla import remote_client

    config = {
        "checkpoint_contract": {
            "state_dim": 20,
            "action_dim": 20,
            "chunk_size": 20,
            "image_keys": ["rgb"],
            "tactile_keys": [],
            "tactile_embedding_dim": 512,
            "tactile_num_tokens": 0,
            "lora_rank": 0,
            "vlm_lora_target_modules": [],
        }
    }
    expected = remote_client._checkpoint_contract(config, {"action_horizon": 20})
    selected = []
    validated = []
    monkeypatch.setattr(remote_client, "resolve_checkpoint", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(
        remote_client,
        "validate_checkpoint",
        lambda *args, **kwargs: validated.append(kwargs["expected"])
        or type("Report", (), {"require_valid": lambda self: self})(),
    )
    monkeypatch.setattr(
        remote_client.VTJaxSmolVLAPolicy,
        "from_pretrained",
        lambda *args, **kwargs: selected.append("vt") or object(),
    )
    monkeypatch.setattr(
        remote_client.JaxSmolVLAPolicy,
        "from_pretrained",
        lambda *args, **kwargs: selected.append("visual") or object(),
    )

    remote_client._load_validated_policy(
        "checkpoint",
        revision=None,
        allow_download=False,
        expected=expected,
        rename_map=None,
    )

    assert expected.tactile_proj_mode == "frozen"
    assert validated == [expected]
    assert selected == ["visual"]


def _copy_deploy_entry_points(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    deploy_dir = project / "deploy_smolvla"
    scripts_dir = deploy_dir / "scripts"
    config_dir = deploy_dir / "configs"
    scripts_dir.mkdir(parents=True)
    config_dir.mkdir()
    shutil.copy2(SHARED_LAUNCHER, scripts_dir / SHARED_LAUNCHER.name)
    (scripts_dir / SHARED_LAUNCHER.name).chmod(0o644)
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


def _run_launcher(
    project: Path,
    python: Path,
    *,
    hub_cache: Path | None = None,
    marker: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["FRS_PYTHON"] = str(python)
    env["VB_ROBOT_TOKEN"] = "test-token"
    env.pop("HF_HUB_CACHE", None)
    env.pop("HUGGINGFACE_HUB_CACHE", None)
    if hub_cache is not None:
        env["HF_HUB_CACHE"] = str(hub_cache)
    if marker is not None:
        env["FAKE_PYTHON_MARKER"] = str(marker)
    return subprocess.run(
        [
            "bash",
            str(project / "deploy_smolvla" / "scripts" / SHARED_LAUNCHER.name),
            "--config",
            str(project / "deploy_smolvla" / "configs" / DEFAULT_CONFIG.name),
        ],
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
        ["bash", str(SHARED_LAUNCHER), "--config", str(DEFAULT_CONFIG), "--check"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_wrapper_check(
    wrapper: Path,
    *,
    extra_args: tuple[str, ...] = (),
    config_override: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["VB_ROBOT_TOKEN"] = "test-token"
    env["HF_HUB_CACHE"] = str(ROOT / "checkpoints" / "model")
    if config_override is not None:
        env["FRS_DEPLOY_CONFIG"] = str(config_override)
    return subprocess.run(
        ["bash", str(wrapper), "--check", *extra_args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_frs_wrapper_uses_frs_config_without_executable_nested_script() -> None:
    assert SHARED_LAUNCHER.stat().st_mode & 0o111 == 0

    result = _run_wrapper_check(FRS_LAUNCHER)

    assert result.returncode == 0, result.stderr
    assert f"config={FRS_CONFIG}" in result.stdout


def test_vt_wrapper_uses_vt_config_without_executable_nested_script() -> None:
    assert SHARED_LAUNCHER.stat().st_mode & 0o111 == 0

    result = _run_wrapper_check(VT_LAUNCHER)

    assert result.returncode == 0, result.stderr
    assert f"config={DEFAULT_CONFIG}" in result.stdout


@pytest.mark.parametrize(
    ("wrapper", "explicit_config"),
    (
        (FRS_LAUNCHER, DEFAULT_CONFIG),
        (VT_LAUNCHER, FRS_CONFIG),
    ),
)
def test_wrapper_allows_later_explicit_config_override(
    wrapper: Path, explicit_config: Path
) -> None:
    result = _run_wrapper_check(
        wrapper,
        extra_args=("--config", str(explicit_config)),
    )

    assert result.returncode == 0, result.stderr
    assert f"config={explicit_config}" in result.stdout


@pytest.mark.parametrize(
    ("wrapper", "config_override"),
    (
        (FRS_LAUNCHER, DEFAULT_CONFIG),
        (VT_LAUNCHER, FRS_CONFIG),
    ),
)
def test_public_wrapper_preserves_frs_deploy_config_override(
    wrapper: Path, config_override: Path
) -> None:
    result = _run_wrapper_check(wrapper, config_override=config_override)

    assert result.returncode == 0, result.stderr
    assert f"config={config_override}" in result.stdout


def test_shared_launcher_requires_explicit_config(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["VB_ROBOT_TOKEN"] = "test-token"
    env["HF_HUB_CACHE"] = str(tmp_path / "hub")

    result = subprocess.run(
        ["bash", str(SHARED_LAUNCHER), "--check"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--config is required" in result.stderr


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


def test_launcher_exec_uses_project_model_cache_by_default(tmp_path: Path) -> None:
    project = _copy_deploy_entry_points(tmp_path)
    result = _run_launcher(project, _fake_python(tmp_path))

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
        [
            "bash",
            str(project / "deploy_smolvla" / "scripts" / SHARED_LAUNCHER.name),
            "--config",
            str(project / "deploy_smolvla" / "configs" / DEFAULT_CONFIG.name),
            "--check",
        ],
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
        [
            "bash",
            str(project / "deploy_smolvla" / "scripts" / SHARED_LAUNCHER.name),
            "--config",
            str(project / "deploy_smolvla" / "configs" / DEFAULT_CONFIG.name),
            "--check",
        ],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert model_sentinel.read_text(encoding="utf-8") == "existing model cache\n"
    assert encoder_sentinel.read_text(encoding="utf-8") == "existing encoder checkpoint\n"


def test_launcher_exec_preserves_explicit_hf_hub_cache(tmp_path: Path) -> None:
    project = _copy_deploy_entry_points(tmp_path)
    cache = tmp_path / "custom-cache" / "hub"

    result = _run_launcher(project, _fake_python(tmp_path), hub_cache=cache)

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{cache}\n"
    assert cache.is_dir()


def test_launcher_fails_before_python_when_hf_hub_cache_is_a_file(
    tmp_path: Path,
) -> None:
    project = _copy_deploy_entry_points(tmp_path)
    cache = tmp_path / "hub-cache-file"
    marker = tmp_path / "python-started"
    cache.write_text("not a directory\n", encoding="utf-8")

    result = _run_launcher(
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
        [
            "bash",
            str(project / "deploy_smolvla" / "scripts" / SHARED_LAUNCHER.name),
            "--help",
        ],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "HF_HUB_CACHE" in result.stdout
    assert "<project>/checkpoints/model" in result.stdout
    assert "FRS_DEPLOY_CONFIG" not in result.stdout


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
