from datetime import datetime
import os
from pathlib import Path
import subprocess

import pytest

from train_vtsmolvla import launcher
from train_vtsmolvla.launcher import load_launcher_settings, run_pipeline

ROOT = Path(__file__).resolve().parents[2]


def test_vt_launcher_reads_yaml_owned_settings():
    settings, config = load_launcher_settings(
        ROOT / "train_vtsmolvla/configs/train.yaml",
        ROOT,
    )
    assert settings.tmux_session == "vtsmolvla_train"
    assert settings.foreground is False
    assert settings.logs_dir == ROOT / "train_vtsmolvla/outputs/logs"
    assert config["tactile_embedding_cache"]["enabled"] is True


def test_vt_preflight_rejects_missing_encoder_before_gpu_or_subprocess(monkeypatch, tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "checkpoint: org/model\n"
        "datasets: []\n"
        "output: output\n"
        "resume: null\n"
        "launcher: {tmux_session: vt, foreground: true, logs_dir: logs}\n"
        "tactile_embedding_cache: {enabled: true, root: cache}\n"
        "model:\n"
        "  use_tactile_encoder: true\n"
        "  tactile_encoder_path: missing-encoder\n"
        "  freeze_tactile_encoder: true\n"
        "  tactile_keys: [left]\n"
        "  tactile_embedding_dim: 512\n"
        "  tactile_num_tokens: 1\n",
        encoding="utf-8",
    )
    settings, config = load_launcher_settings(config_path, tmp_path)
    monkeypatch.setattr(
        launcher,
        "shared_preflight",
        lambda *args, **kwargs: pytest.fail("shared preflight must not run for a missing encoder"),
    )

    with pytest.raises(FileNotFoundError, match="missing-encoder"):
        launcher.preflight(settings, config)


def test_vt_preflight_delegates_with_the_vt_checkpoint_resolver(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "checkpoint: org/model\n"
        "datasets: [{root: dataset}]\n"
        "output: output\n"
        "resume: null\n"
        "launcher: {tmux_session: vt, foreground: true, logs_dir: logs}\n"
        "tactile_embedding_cache: {enabled: false, root: cache}\n"
        "model:\n"
        "  use_tactile_encoder: true\n"
        "  tactile_encoder_path: encoder\n"
        "  freeze_tactile_encoder: true\n"
        "  tactile_keys: [left]\n"
        "  tactile_embedding_dim: 512\n"
        "  tactile_num_tokens: 1\n",
        encoding="utf-8",
    )
    settings, config = load_launcher_settings(config_path, tmp_path)
    calls = []
    monkeypatch.setattr(
        launcher,
        "resolve_checkpoint",
        lambda checkpoint, **kwargs: calls.append((checkpoint, kwargs)) or tmp_path / "checkpoint",
    )
    monkeypatch.setattr(
        launcher.shared_launcher,
        "resolve_checkpoint",
        lambda *args, **kwargs: pytest.fail("visual checkpoint resolver must not be used"),
    )
    monkeypatch.setattr(
        launcher.shared_launcher.jax,
        "devices",
        lambda: [type("Device", (), {"platform": "gpu"})()],
    )

    assert launcher.preflight(settings, config) == tmp_path / "checkpoint"
    assert calls == [("org/model", {"revision": None, "local_files_only": True})]


def test_cache_precompute_succeeds_before_training_and_uses_distinct_logs(monkeypatch, tmp_path):
    settings = launcher.LauncherSettings(
        project_root=tmp_path,
        config_path=tmp_path / "train.yaml",
        output=tmp_path / "output",
        resume=None,
        tmux_session="vtsmolvla_train",
        foreground=True,
        logs_dir=tmp_path / "logs",
    )
    config = {"tactile_embedding_cache": {"enabled": True, "root": "cache"}}
    calls = []
    monkeypatch.setattr(
        launcher,
        "stream_command",
        lambda command, *, cwd, log_path: calls.append((command, cwd, log_path.name)) or 0,
    )

    status = run_pipeline(
        settings,
        config,
        uv_bin="/usr/bin/uv",
        now=datetime(2026, 8, 8, 14, 30, 0),
    )

    assert status == 0
    assert [name for _, _, name in calls] == [
        "precompute_20260808_143000.log",
        "train_20260808_143000.log",
    ]
    assert calls[0][0] == [
        "/usr/bin/uv", "run", "--no-sync", "python",
        "-m", "train_vtsmolvla.precompute",
        "--config", str(settings.config_path),
    ]
    assert calls[1][0] == [
        "/usr/bin/uv", "run", "--no-sync", "python", "-m",
        "train_vtsmolvla.train", "--config", str(settings.config_path),
    ]


def test_precompute_failure_prevents_training(monkeypatch, tmp_path):
    settings = launcher.LauncherSettings(
        project_root=tmp_path,
        config_path=tmp_path / "train.yaml",
        output=tmp_path / "output",
        resume=None,
        tmux_session="vtsmolvla_train",
        foreground=True,
        logs_dir=tmp_path / "logs",
    )
    calls = []
    monkeypatch.setattr(
        launcher,
        "stream_command",
        lambda command, **kwargs: calls.append(command) or 7,
    )

    assert run_pipeline(
        settings,
        {"tactile_embedding_cache": {"enabled": True, "root": "cache"}},
        uv_bin="uv",
        now=datetime(2026, 8, 8, 14, 30, 0),
    ) == 7
    assert len(calls) == 1


def test_vt_launcher_uses_the_shared_tmux_handoff(monkeypatch):
    settings, _ = load_launcher_settings(
        ROOT / "train_vtsmolvla/configs/train.yaml",
        ROOT,
    )
    calls = []
    monkeypatch.setattr(
        launcher.shared_launcher,
        "maybe_launch_tmux",
        lambda actual, **kwargs: calls.append((actual, kwargs)) or True,
    )

    assert launcher.launch(settings, {"tactile_embedding_cache": {"enabled": False}}) == 0
    assert calls == [(
        settings,
        {
            "foreground_env": "VTSMOLVLA_FOREGROUND",
            "launcher_module": "train_vtsmolvla.launcher",
            "log_prefix": "vtsmolvla",
        },
    )]


def test_shell_is_thin_and_discovers_root_from_foreign_cwd(tmp_path):
    shell = (ROOT / "train_vtsmolvla/scripts/train.sh").read_text()
    assert "python -m train_vtsmolvla.launcher" in shell
    for forbidden in ("batch_size", "optimizer_lr", "save_freq", "precompute_batch_size"):
        assert forbidden not in shell

    fake_uv = tmp_path / "uv"
    fake_uv.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@"\n', encoding="utf-8")
    fake_uv.chmod(0o755)
    foreign_cwd = tmp_path / "foreign"
    foreign_cwd.mkdir()
    result = subprocess.run(
        ["bash", str(ROOT / "train_vtsmolvla/scripts/train.sh")],
        cwd=foreign_cwd,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "run",
        "--no-sync",
        "python",
        "-m",
        "train_vtsmolvla.launcher",
        "--config",
        str(ROOT / "train_vtsmolvla/configs/train.yaml"),
    ]


def test_legacy_entrypoints_are_replaced_by_package_readmes_and_setup_commands():
    assert not (ROOT / "scripts/start_vtsmolvla_train.sh").exists()
    assert not (ROOT / "train_for_agent.md").exists()
    for package in ("train_smolvla", "train_vtsmolvla"):
        readme = (ROOT / package / "README.md").read_text(encoding="utf-8")
        assert f"bash {package}/scripts/train.sh" in readme
        assert "resume" in readme
        assert "tmux attach" in readme
        assert "logs" in readme
    setup = (ROOT / "scripts/setup_env.sh").read_text(encoding="utf-8")
    assert "train_smolvla/scripts/train.sh" in setup
    assert "train_vtsmolvla/scripts/train.sh" in setup
    assert "scripts/start_vtsmolvla_train.sh" not in setup
