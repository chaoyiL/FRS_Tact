"""Launch visual SmolVLA training from its YAML configuration."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax

from train_smolvla.checkpoint import resolve_checkpoint
from train_smolvla.train import DEFAULT_CONFIG, load_yaml_config


@dataclass(frozen=True)
class LauncherSettings:
    project_root: Path
    config_path: Path
    output: Path
    resume: Path | None
    tmux_session: str
    foreground: bool
    logs_dir: Path


def _project_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _required(mapping: dict[str, Any], key: str, *, context: str = "config") -> Any:
    value = mapping.get(key)
    if value in (None, ""):
        raise ValueError(f"missing required {context} field: {key}")
    return value


def load_launcher_settings(config_path: Path, project_root: Path) -> LauncherSettings:
    """Load launcher-only settings and resolve all filesystem paths to the project."""

    root = project_root.expanduser().resolve()
    resolved_config = _project_path(root, config_path).resolve()
    config = load_yaml_config(resolved_config)
    launcher = config.get("launcher")
    if not isinstance(launcher, dict):
        raise ValueError("missing required config mapping: launcher")
    resume_value = config.get("resume")
    return LauncherSettings(
        project_root=root,
        config_path=resolved_config,
        output=_project_path(root, _required(config, "output")).resolve(),
        resume=None if resume_value in (None, "") else _project_path(root, resume_value).resolve(),
        tmux_session=str(_required(launcher, "tmux_session", context="launcher")),
        foreground=bool(launcher.get("foreground", False)),
        logs_dir=_project_path(root, _required(launcher, "logs_dir", context="launcher")).resolve(),
    )


def find_uv() -> str:
    executable = shutil.which("uv")
    if executable is None:
        raise RuntimeError("找不到 uv，请先运行 scripts/setup_env.sh")
    return executable


def timestamped_log_path(settings: LauncherSettings, now: datetime | None = None) -> Path:
    instant = now or datetime.now()
    return settings.logs_dir / f"train_{instant:%Y%m%d_%H%M%S}.log"


def tmux_session_exists(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _configured_dataset_roots(config: dict[str, Any], project_root: Path) -> list[Path]:
    datasets = config.get("datasets") or []
    if not isinstance(datasets, list):
        raise ValueError("datasets must be a list")
    roots: list[Path] = []
    for dataset in datasets:
        if isinstance(dataset, dict) and dataset.get("root") not in (None, ""):
            roots.append(_project_path(project_root, dataset["root"]).resolve())
    return roots


def _local_checkpoint_path(project_root: Path, value: str | Path) -> Path | None:
    """Return a project-relative local checkpoint path without reclassifying ``org/model`` IDs."""

    raw_path = Path(value).expanduser()
    candidate = _project_path(project_root, raw_path).resolve()
    text = str(value)
    is_explicit_relative = text.startswith(("./", "../"))
    is_nested_relative_path = not raw_path.is_absolute() and len(raw_path.parts) > 2
    if raw_path.is_absolute() or is_explicit_relative or candidate.exists() or is_nested_relative_path:
        return candidate
    return None


def _checkpoint_target(project_root: Path, value: str | Path) -> str | Path:
    local_path = _local_checkpoint_path(project_root, value)
    if local_path is None:
        return value
    if not local_path.exists():
        raise FileNotFoundError(
            f"checkpoint path does not exist: {local_path}; set checkpoint to an existing local path "
            "or use a Hugging Face repo ID"
        )
    return local_path


def preflight(settings: LauncherSettings, config: dict[str, Any]) -> Path:
    """Reject missing local resources and unsafe launches before creating tmux."""

    for dataset_root in _configured_dataset_roots(config, settings.project_root):
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"dataset root does not exist: {dataset_root}; update datasets[].root or prepare the dataset"
            )

    checkpoint = resolve_checkpoint(
        _checkpoint_target(settings.project_root, _required(config, "checkpoint")),
        revision=config.get("revision"),
        local_files_only=not bool(config.get("allow_download", False)),
    )
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("no GPU device is available; run this launcher on a GPU-enabled JAX environment")

    if settings.resume is None and settings.output.exists() and any(settings.output.glob("checkpoint-*")):
        raise FileExistsError(
            f"output already contains checkpoints: {settings.output}; set resume in the YAML or choose a new output"
        )
    if settings.resume is not None and not settings.resume.is_dir():
        raise FileNotFoundError(
            f"resume directory does not exist: {settings.resume}; set resume to an existing checkpoint directory"
        )
    return checkpoint


def stream_command(command: Sequence[str], *, cwd: Path, log_path: Path) -> int:
    """Stream a foreground child process to both stdout and one run-specific log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert process.stdout is not None
    with log_path.open("a", encoding="utf-8") as log:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
    return process.wait()


def _foreground_command(settings: LauncherSettings, uv_bin: str) -> list[str]:
    return [
        uv_bin,
        "run",
        "--no-sync",
        "python",
        "-m",
        "train_smolvla.train",
        "--config",
        str(settings.config_path),
    ]


def launch(settings: LauncherSettings) -> int:
    """Start training in tmux when possible, otherwise stream it in this process."""

    in_foreground = (
        settings.foreground
        or os.environ.get("SMOLVLA_FOREGROUND") == "1"
        or bool(os.environ.get("TMUX"))
    )
    tmux_bin = None if in_foreground else shutil.which("tmux")
    if not in_foreground and tmux_bin is not None:
        if tmux_session_exists(settings.tmux_session):
            raise RuntimeError(
                f"tmux session {settings.tmux_session!r} already exists; attach with "
                f"tmux attach -t {settings.tmux_session}"
            )
        uv_bin = find_uv()
        command = [
            tmux_bin,
            "new-session",
            "-d",
            "-s",
            settings.tmux_session,
            "-c",
            str(settings.project_root),
            "env",
            "SMOLVLA_FOREGROUND=1",
            uv_bin,
            "run",
            "--no-sync",
            "python",
            "-m",
            "train_smolvla.launcher",
            "--config",
            str(settings.config_path),
        ]
        subprocess.run(command, check=True)
        print(f"[smolvla] started tmux session: {settings.tmux_session}")
        return 0

    if not in_foreground and tmux_bin is None:
        print("[smolvla] warning: tmux is unavailable; running in the foreground", file=sys.stderr)
    log_path = timestamped_log_path(settings).resolve()
    print(f"[smolvla] log: {log_path}")
    return stream_command(_foreground_command(settings, find_uv()), cwd=settings.project_root, log_path=log_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG})",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    settings = load_launcher_settings(args.config, project_root)
    config = load_yaml_config(settings.config_path)
    preflight(settings, config)
    return launch(settings)


if __name__ == "__main__":
    raise SystemExit(main())
