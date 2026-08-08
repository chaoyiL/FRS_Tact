from datetime import datetime
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from train_smolvla import launcher
from train_smolvla.launcher import (
    LauncherSettings,
    load_launcher_settings,
    preflight,
    stream_command,
    timestamped_log_path,
)

ROOT = Path(__file__).resolve().parents[2]


class FixedDatetime:
    @classmethod
    def now(cls):
        return datetime(2026, 8, 8, 14, 30, 0)


def test_launcher_defaults_to_tmux_and_timestamped_ignored_log(monkeypatch):
    settings = load_launcher_settings(ROOT / "train_smolvla/configs/train.yaml", ROOT)
    assert settings.tmux_session == "smolvla_train"
    assert settings.foreground is False
    assert settings.logs_dir == ROOT / "train_smolvla/outputs/logs"
    monkeypatch.setattr(launcher, "datetime", FixedDatetime)
    assert timestamped_log_path(settings).name == "train_20260808_143000.log"


def test_launcher_yaml_controls_session_foreground_and_logs(tmp_path):
    project_root = tmp_path / "project"
    config_path = project_root / "configs" / "custom.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "checkpoint": "owner/model",
                "output": "runs/custom",
                "resume": None,
                "launcher": {
                    "tmux_session": "custom-visual",
                    "foreground": True,
                    "logs_dir": "runtime/custom-logs",
                },
            }
        ),
        encoding="utf-8",
    )

    settings = load_launcher_settings(Path("configs/custom.yaml"), project_root)

    assert settings.tmux_session == "custom-visual"
    assert settings.foreground is True
    assert settings.logs_dir == project_root / "runtime/custom-logs"


def test_existing_tmux_session_is_rejected(monkeypatch):
    settings = load_launcher_settings(ROOT / "train_smolvla/configs/train.yaml", ROOT)
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/tmux")
    monkeypatch.setattr(launcher, "tmux_session_exists", lambda name: True)
    with pytest.raises(RuntimeError, match="tmux attach -t smolvla_train"):
        launcher.launch(settings)


def test_shell_contains_no_training_constants():
    shell = (ROOT / "train_smolvla/scripts/train.sh").read_text()
    assert "python -m train_smolvla.launcher" in shell
    for forbidden in ("batch_size", "optimizer_lr", "save_freq", "steps:"):
        assert forbidden not in shell


def test_shell_discovers_root_from_a_foreign_cwd_without_starting_training(tmp_path):
    fake_uv = tmp_path / "uv"
    fake_uv.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@"\n', encoding="utf-8")
    fake_uv.chmod(0o755)
    foreign_cwd = tmp_path / "foreign"
    foreign_cwd.mkdir()

    result = subprocess.run(
        ["bash", str(ROOT / "train_smolvla/scripts/train.sh")],
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
        "train_smolvla.launcher",
        "--config",
        str(ROOT / "train_smolvla/configs/train.yaml"),
    ]


def test_preflight_resolves_checkpoint_checks_local_data_gpu_and_existing_output(
    monkeypatch, tmp_path
):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    output = tmp_path / "output"
    settings = LauncherSettings(
        project_root=tmp_path,
        config_path=tmp_path / "train.yaml",
        output=output,
        resume=None,
        tmux_session="smolvla_train",
        foreground=False,
        logs_dir=tmp_path / "logs",
    )
    calls = []
    monkeypatch.setattr(
        launcher,
        "resolve_checkpoint",
        lambda checkpoint, **kwargs: calls.append((checkpoint, kwargs)) or tmp_path / "checkpoint",
    )
    monkeypatch.setattr(launcher.jax, "devices", lambda: [type("Device", (), {"platform": "gpu"})()])

    preflight(
        settings,
        {"checkpoint": "local-checkpoint", "allow_download": False, "datasets": [{"root": "dataset"}]},
        checkpoint_resolver=launcher.resolve_checkpoint,
    )

    assert calls == [("local-checkpoint", {"revision": None, "local_files_only": True})]


def test_preflight_rebases_an_existing_relative_checkpoint_from_a_non_repo_cwd(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    checkpoint = project_root / "checkpoints" / "local"
    checkpoint.mkdir(parents=True)
    foreign_cwd = tmp_path / "elsewhere"
    foreign_cwd.mkdir()
    settings = LauncherSettings(
        project_root=project_root,
        config_path=project_root / "train.yaml",
        output=project_root / "output",
        resume=None,
        tmux_session="smolvla_train",
        foreground=False,
        logs_dir=project_root / "logs",
    )
    calls = []
    monkeypatch.chdir(foreign_cwd)
    monkeypatch.setattr(
        launcher,
        "resolve_checkpoint",
        lambda value, **kwargs: calls.append((value, kwargs)) or checkpoint,
    )
    monkeypatch.setattr(launcher.jax, "devices", lambda: [type("Device", (), {"platform": "gpu"})()])

    preflight(
        settings,
        {"checkpoint": "checkpoints/local", "datasets": []},
        checkpoint_resolver=launcher.resolve_checkpoint,
    )

    assert calls == [(checkpoint, {"revision": None, "local_files_only": True})]


def test_preflight_leaves_huggingface_repo_ids_unmodified(monkeypatch, tmp_path):
    settings = LauncherSettings(
        project_root=tmp_path,
        config_path=tmp_path / "train.yaml",
        output=tmp_path / "output",
        resume=None,
        tmux_session="smolvla_train",
        foreground=False,
        logs_dir=tmp_path / "logs",
    )
    calls = []
    monkeypatch.setattr(
        launcher,
        "resolve_checkpoint",
        lambda value, **kwargs: calls.append((value, kwargs)) or tmp_path / "checkpoint",
    )
    monkeypatch.setattr(launcher.jax, "devices", lambda: [type("Device", (), {"platform": "gpu"})()])

    preflight(
        settings,
        {"checkpoint": "org/model", "datasets": []},
        checkpoint_resolver=launcher.resolve_checkpoint,
    )

    assert calls == [("org/model", {"revision": None, "local_files_only": True})]


def test_preflight_reports_an_absolute_path_for_a_missing_local_checkpoint(monkeypatch, tmp_path):
    settings = LauncherSettings(
        project_root=tmp_path,
        config_path=tmp_path / "train.yaml",
        output=tmp_path / "output",
        resume=None,
        tmux_session="smolvla_train",
        foreground=False,
        logs_dir=tmp_path / "logs",
    )
    missing = tmp_path / "checkpoints" / "missing"
    monkeypatch.setattr(launcher, "resolve_checkpoint", lambda *args, **kwargs: pytest.fail("unexpected resolver call"))

    with pytest.raises(FileNotFoundError, match=str(missing)) as error:
        preflight(
            settings,
            {"checkpoint": "./checkpoints/missing", "datasets": []},
            checkpoint_resolver=launcher.resolve_checkpoint,
        )

    assert "set checkpoint" in str(error.value)


def test_preflight_rejects_missing_local_data_resume_and_checkpoint_overwrite(monkeypatch, tmp_path):
    settings = LauncherSettings(
        project_root=tmp_path,
        config_path=tmp_path / "train.yaml",
        output=tmp_path / "output",
        resume=None,
        tmux_session="smolvla_train",
        foreground=False,
        logs_dir=tmp_path / "logs",
    )
    monkeypatch.setattr(launcher, "resolve_checkpoint", lambda *args, **kwargs: tmp_path / "checkpoint")
    monkeypatch.setattr(launcher.jax, "devices", lambda: [type("Device", (), {"platform": "gpu"})()])

    with pytest.raises(FileNotFoundError, match=str(tmp_path / "missing-data")):
        preflight(
            settings,
            {"checkpoint": "checkpoint", "datasets": [{"root": "missing-data"}]},
            checkpoint_resolver=launcher.resolve_checkpoint,
        )

    dataset_file = tmp_path / "dataset-file"
    dataset_file.write_text("not a dataset directory", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=str(dataset_file)):
        preflight(
            settings,
            {"checkpoint": "checkpoint", "datasets": [{"root": "dataset-file"}]},
            checkpoint_resolver=launcher.resolve_checkpoint,
        )

    (tmp_path / "output" / "checkpoint-1").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="resume"):
        preflight(
            settings,
            {"checkpoint": "checkpoint", "datasets": []},
            checkpoint_resolver=launcher.resolve_checkpoint,
        )

    resume_settings = LauncherSettings(
        project_root=tmp_path,
        config_path=tmp_path / "train.yaml",
        output=tmp_path / "fresh-output",
        resume=tmp_path / "missing-resume",
        tmux_session="smolvla_train",
        foreground=False,
        logs_dir=tmp_path / "logs",
    )
    with pytest.raises(FileNotFoundError, match=str(tmp_path / "missing-resume")):
        preflight(
            resume_settings,
            {"checkpoint": "checkpoint", "datasets": []},
            checkpoint_resolver=launcher.resolve_checkpoint,
        )


def test_preflight_rejects_cpu_only_jax(monkeypatch, tmp_path):
    settings = LauncherSettings(
        project_root=tmp_path,
        config_path=tmp_path / "train.yaml",
        output=tmp_path / "output",
        resume=None,
        tmux_session="smolvla_train",
        foreground=False,
        logs_dir=tmp_path / "logs",
    )
    monkeypatch.setattr(launcher, "resolve_checkpoint", lambda *args, **kwargs: tmp_path / "checkpoint")
    monkeypatch.setattr(launcher.jax, "devices", lambda: [type("Device", (), {"platform": "cpu"})()])

    with pytest.raises(RuntimeError, match="GPU"):
        preflight(
            settings,
            {"checkpoint": "owner/model", "datasets": []},
            checkpoint_resolver=launcher.resolve_checkpoint,
        )


def test_stream_command_tees_output_and_returns_child_exit_code(tmp_path, capsys):
    log_path = tmp_path / "logs" / "train.log"

    status = stream_command(
        [sys.executable, "-c", "print('training output'); raise SystemExit(7)"],
        cwd=tmp_path,
        log_path=log_path,
    )

    assert status == 7
    assert "training output" in capsys.readouterr().out
    assert log_path.read_text() == "training output\n"


def test_launch_creates_a_tmux_session_that_reenters_the_launcher(monkeypatch, tmp_path):
    settings = LauncherSettings(
        project_root=tmp_path,
        config_path=tmp_path / "configs/train.yaml",
        output=tmp_path / "output",
        resume=None,
        tmux_session="smolvla_train",
        foreground=False,
        logs_dir=tmp_path / "logs",
    )
    recorded = []
    monkeypatch.delenv("SMOLVLA_FOREGROUND", raising=False)
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/tmux" if name == "tmux" else None)
    monkeypatch.setattr(launcher, "tmux_session_exists", lambda name: False)
    monkeypatch.setattr(launcher, "find_uv", lambda: "/usr/bin/uv")
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, check: recorded.append((command, check)))

    assert launcher.launch(settings) == 0
    assert recorded == [(
        [
            "/usr/bin/tmux", "new-session", "-d", "-s", "smolvla_train", "-c", str(tmp_path),
            "env", "SMOLVLA_FOREGROUND=1", "/usr/bin/uv", "run", "--no-sync", "python", "-m",
            "train_smolvla.launcher", "--config", str(tmp_path / "configs/train.yaml"),
        ],
        True,
    )]


def test_foreground_environment_bypasses_tmux_and_streams_training(monkeypatch, tmp_path):
    settings = LauncherSettings(
        project_root=tmp_path,
        config_path=tmp_path / "configs/train.yaml",
        output=tmp_path / "output",
        resume=None,
        tmux_session="smolvla_train",
        foreground=False,
        logs_dir=tmp_path / "logs",
    )
    streamed = []
    monkeypatch.setenv("SMOLVLA_FOREGROUND", "1")
    monkeypatch.setattr(launcher, "find_uv", lambda: "/usr/bin/uv")
    monkeypatch.setattr(launcher, "stream_command", lambda command, **kwargs: streamed.append((command, kwargs)) or 9)

    assert launcher.launch(settings) == 9
    assert streamed[0][0] == [
        "/usr/bin/uv", "run", "--no-sync", "python", "-m", "train_smolvla.train", "--config",
        str(tmp_path / "configs/train.yaml"),
    ]


def test_foreground_yaml_bypasses_tmux_and_uses_yaml_log_directory(monkeypatch, tmp_path):
    settings = LauncherSettings(
        project_root=tmp_path,
        config_path=tmp_path / "configs/train.yaml",
        output=tmp_path / "output",
        resume=None,
        tmux_session="yaml-session",
        foreground=True,
        logs_dir=tmp_path / "yaml-logs",
    )
    streamed = []
    monkeypatch.delenv("SMOLVLA_FOREGROUND", raising=False)
    monkeypatch.setattr(launcher.shutil, "which", lambda name: pytest.fail("tmux must not be queried"))
    monkeypatch.setattr(launcher, "find_uv", lambda: "/usr/bin/uv")
    monkeypatch.setattr(launcher, "stream_command", lambda command, **kwargs: streamed.append((command, kwargs)) or 0)

    assert launcher.launch(settings) == 0
    assert streamed[0][1]["log_path"].parent == tmp_path / "yaml-logs"


def test_existing_tmux_environment_does_not_create_a_nested_session(monkeypatch, tmp_path):
    settings = LauncherSettings(
        project_root=tmp_path,
        config_path=tmp_path / "configs/train.yaml",
        output=tmp_path / "output",
        resume=None,
        tmux_session="smolvla_train",
        foreground=False,
        logs_dir=tmp_path / "logs",
    )
    streamed = []
    monkeypatch.delenv("SMOLVLA_FOREGROUND", raising=False)
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    monkeypatch.setattr(launcher.shutil, "which", lambda name: pytest.fail("tmux must not be queried"))
    monkeypatch.setattr(launcher, "find_uv", lambda: "/usr/bin/uv")
    monkeypatch.setattr(launcher, "stream_command", lambda command, **kwargs: streamed.append(command) or 0)

    assert launcher.launch(settings) == 0
    assert streamed[0][streamed[0].index("-m") + 1] == "train_smolvla.train"
