from datetime import datetime
import os
from pathlib import Path
import subprocess

import pytest

from train_vtsmolvla import launcher


ROOT = Path(__file__).resolve().parents[2]


class FixedDatetime:
    @classmethod
    def now(cls):
        return datetime(2026, 8, 8, 14, 30, 0)


def _settings(tmp_path, **overrides):
    values = {
        "project_root": tmp_path,
        "config_path": tmp_path / "configs/train.yaml",
        "output": tmp_path / "output",
        "resume": None,
        "tmux_session": "vtsmolvla_train",
        "foreground": False,
        "logs_dir": tmp_path / "logs",
        "precompute": True,
    }
    values.update(overrides)
    return launcher.LauncherSettings(**values)


def test_default_yaml_controls_vt_launcher_and_timestamped_logs(monkeypatch):
    settings = launcher.load_launcher_settings(
        ROOT / "train_vtsmolvla/configs/train.yaml", ROOT
    )
    assert settings.tmux_session == "vtsmolvla_train"
    assert settings.foreground is False
    assert settings.logs_dir == ROOT / "train_vtsmolvla/outputs/logs"
    assert settings.precompute is True
    monkeypatch.setattr(launcher, "datetime", FixedDatetime)
    precompute_log, train_log = launcher.timestamped_log_paths(settings)
    assert precompute_log.name == "precompute_20260808_143000.log"
    assert train_log.name == "train_20260808_143000.log"


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


def test_preflight_checks_dataset_encoder_checkpoint_gpu_resume_and_output(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    settings = _settings(tmp_path)
    calls = []
    monkeypatch.setattr(
        launcher,
        "resolve_checkpoint",
        lambda checkpoint, **kwargs: calls.append((checkpoint, kwargs)) or tmp_path / "checkpoint",
    )
    monkeypatch.setattr(launcher.jax, "devices", lambda: [type("Device", (), {"platform": "gpu"})()])
    config = {
        "checkpoint": "owner/model",
        "datasets": [{"root": "dataset"}],
        "model": {"tactile_encoder_path": "encoder"},
    }

    assert launcher.preflight(settings, config) == tmp_path / "checkpoint"
    assert calls == [("owner/model", {"revision": None, "local_files_only": True})]

    with pytest.raises(FileNotFoundError, match="missing-dataset"):
        launcher.preflight(settings, {**config, "datasets": [{"root": "missing-dataset"}]})
    with pytest.raises(FileNotFoundError, match="missing-encoder"):
        launcher.preflight(
            settings,
            {**config, "model": {"tactile_encoder_path": "missing-encoder"}},
        )

    monkeypatch.setattr(launcher.jax, "devices", lambda: [type("Device", (), {"platform": "cpu"})()])
    with pytest.raises(RuntimeError, match="GPU"):
        launcher.preflight(settings, config)
    monkeypatch.setattr(launcher.jax, "devices", lambda: [type("Device", (), {"platform": "gpu"})()])

    (settings.output / "checkpoint-1").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="resume"):
        launcher.preflight(settings, config)
    with pytest.raises(FileNotFoundError, match="missing-resume"):
        launcher.preflight(
            _settings(tmp_path, output=tmp_path / "fresh", resume=tmp_path / "missing-resume"),
            config,
        )


def test_tmux_uses_yaml_session_and_reenters_vt_launcher(monkeypatch, tmp_path):
    settings = _settings(tmp_path, tmux_session="from-yaml", precompute=False)
    recorded = []
    monkeypatch.delenv("VTSMOLVLA_FOREGROUND", raising=False)
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/tmux")
    monkeypatch.setattr(launcher, "tmux_session_exists", lambda name: False)
    monkeypatch.setattr(launcher, "find_uv", lambda: "/usr/bin/uv")
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, check: recorded.append(command))

    assert launcher.launch(settings) == 0
    command = recorded[0]
    assert command[command.index("-s") + 1] == "from-yaml"
    assert "VTSMOLVLA_FOREGROUND=1" in command
    assert command[command.index("-m") + 1] == "train_vtsmolvla.launcher"


def test_foreground_runs_precompute_before_training_with_yaml_logs(monkeypatch, tmp_path):
    settings = _settings(tmp_path, foreground=True, logs_dir=tmp_path / "yaml-logs")
    streamed = []
    monkeypatch.setattr(launcher, "find_uv", lambda: "/usr/bin/uv")
    monkeypatch.setattr(
        launcher,
        "stream_command",
        lambda command, **kwargs: streamed.append((command, kwargs["log_path"])) or 0,
    )
    monkeypatch.setattr(launcher, "datetime", FixedDatetime)

    assert launcher.launch(settings) == 0
    assert [command[command.index("-m") + 1] for command, _ in streamed] == [
        "tools.precompute_tactile_embeddings",
        "train_vtsmolvla.train",
    ]
    assert [path.name for _, path in streamed] == [
        "precompute_20260808_143000.log",
        "train_20260808_143000.log",
    ]
    assert all(path.parent == tmp_path / "yaml-logs" for _, path in streamed)


def test_precompute_failure_stops_before_training(monkeypatch, tmp_path):
    settings = _settings(tmp_path, foreground=True)
    calls = []
    monkeypatch.setattr(launcher, "find_uv", lambda: "/usr/bin/uv")
    monkeypatch.setattr(
        launcher,
        "stream_command",
        lambda command, **kwargs: calls.append(command) or 23,
    )

    assert launcher.launch(settings) == 23
    assert len(calls) == 1
    assert calls[0][calls[0].index("-m") + 1] == "tools.precompute_tactile_embeddings"


def test_existing_tmux_environment_does_not_create_a_nested_session(monkeypatch, tmp_path):
    settings = _settings(tmp_path, precompute=False)
    streamed = []
    monkeypatch.delenv("VTSMOLVLA_FOREGROUND", raising=False)
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    monkeypatch.setattr(launcher.shutil, "which", lambda name: pytest.fail("tmux must not be queried"))
    monkeypatch.setattr(launcher, "find_uv", lambda: "/usr/bin/uv")
    monkeypatch.setattr(launcher, "stream_command", lambda command, **kwargs: streamed.append(command) or 0)

    assert launcher.launch(settings) == 0
    assert streamed[0][streamed[0].index("-m") + 1] == "train_vtsmolvla.train"


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
