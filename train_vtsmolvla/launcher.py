"""Launch vision-tactile SmolVLA training from its YAML configuration."""

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

from train_smolvla.launcher import (
    _checkpoint_target,
    _configured_dataset_roots,
    _project_path,
    _required,
    find_uv,
    stream_command,
    tmux_session_exists,
)
from train_smolvla.train import load_yaml_config
from train_vtsmolvla.checkpoint import resolve_checkpoint
from train_vtsmolvla.train import DEFAULT_CONFIG, VT_ALLOWED_TOP_LEVEL_KEYS, _validate_vt_config


@dataclass(frozen=True)
class LauncherSettings:
    project_root: Path
    config_path: Path
    output: Path
    resume: Path | None
    tmux_session: str
    foreground: bool
    logs_dir: Path
    precompute: bool


def load_launcher_settings(config_path: Path, project_root: Path) -> LauncherSettings:
    """Load VT launcher values and rebase relative paths to the repository root."""

    root = project_root.expanduser().resolve()
    resolved_config = _project_path(root, config_path).resolve()
    config = load_yaml_config(
        resolved_config,
        allowed_top_level_keys=VT_ALLOWED_TOP_LEVEL_KEYS,
    )
    launcher = config.get("launcher")
    if not isinstance(launcher, dict):
        raise ValueError("missing required config mapping: launcher")
    cache = config.get("tactile_embedding_cache") or {}
    if not isinstance(cache, dict):
        raise ValueError("tactile_embedding_cache must be a mapping")
    resume_value = config.get("resume")
    return LauncherSettings(
        project_root=root,
        config_path=resolved_config,
        output=_project_path(root, _required(config, "output")).resolve(),
        resume=None if resume_value in (None, "") else _project_path(root, resume_value).resolve(),
        tmux_session=str(_required(launcher, "tmux_session", context="launcher")),
        foreground=bool(launcher.get("foreground", False)),
        logs_dir=_project_path(
            root,
            _required(launcher, "logs_dir", context="launcher"),
        ).resolve(),
        precompute=bool(cache.get("enabled", False)),
    )


def timestamped_log_paths(
    settings: LauncherSettings,
    now: datetime | None = None,
) -> tuple[Path, Path]:
    instant = now or datetime.now()
    suffix = f"{instant:%Y%m%d_%H%M%S}.log"
    return (
        settings.logs_dir / f"precompute_{suffix}",
        settings.logs_dir / f"train_{suffix}",
    )


def _tactile_encoder_path(config: dict[str, Any], project_root: Path) -> Path:
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("missing required config mapping: model")
    return _project_path(
        project_root,
        _required(model, "tactile_encoder_path", context="model"),
    ).resolve()


def preflight(settings: LauncherSettings, config: dict[str, Any]) -> Path:
    """Reject missing VT resources and unsafe output state before launching."""

    for dataset_root in _configured_dataset_roots(config, settings.project_root):
        if not dataset_root.is_dir():
            raise FileNotFoundError(
                f"dataset root does not exist: {dataset_root}; update datasets[].root or prepare the dataset"
            )

    encoder = _tactile_encoder_path(config, settings.project_root)
    if not encoder.is_dir():
        raise FileNotFoundError(
            f"tactile encoder does not exist: {encoder}; update model.tactile_encoder_path"
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


def _python_module_command(uv_bin: str, module: str, config_path: Path) -> list[str]:
    return [
        uv_bin,
        "run",
        "--no-sync",
        "python",
        "-m",
        module,
        "--config",
        str(config_path),
    ]


def launch(settings: LauncherSettings) -> int:
    """Start VT training in tmux or run cache preparation and training inline."""

    in_foreground = (
        settings.foreground
        or os.environ.get("VTSMOLVLA_FOREGROUND") == "1"
        or bool(os.environ.get("TMUX"))
    )
    tmux_bin = None if in_foreground else shutil.which("tmux")
    if tmux_bin is not None:
        if tmux_session_exists(settings.tmux_session):
            raise RuntimeError(
                f"tmux session {settings.tmux_session!r} already exists; attach with "
                f"tmux attach -t {settings.tmux_session}"
            )
        command = [
            tmux_bin,
            "new-session",
            "-d",
            "-s",
            settings.tmux_session,
            "-c",
            str(settings.project_root),
            "env",
            "VTSMOLVLA_FOREGROUND=1",
            find_uv(),
            "run",
            "--no-sync",
            "python",
            "-m",
            "train_vtsmolvla.launcher",
            "--config",
            str(settings.config_path),
        ]
        subprocess.run(command, check=True)
        print(f"[vtsmolvla] started tmux session: {settings.tmux_session}")
        return 0

    if not in_foreground:
        print("[vtsmolvla] warning: tmux is unavailable; running in the foreground", file=sys.stderr)

    uv_bin = find_uv()
    precompute_log, train_log = timestamped_log_paths(settings)
    if settings.precompute:
        print(f"[vtsmolvla] precompute log: {precompute_log.resolve()}")
        status = stream_command(
            _python_module_command(
                uv_bin,
                "tools.precompute_tactile_embeddings",
                settings.config_path,
            ),
            cwd=settings.project_root,
            log_path=precompute_log.resolve(),
        )
        if status != 0:
            return status

    print(f"[vtsmolvla] train log: {train_log.resolve()}")
    return stream_command(
        _python_module_command(uv_bin, "train_vtsmolvla.train", settings.config_path),
        cwd=settings.project_root,
        log_path=train_log.resolve(),
    )


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
    config = load_yaml_config(
        settings.config_path,
        allowed_top_level_keys=VT_ALLOWED_TOP_LEVEL_KEYS,
    )
    _validate_vt_config(settings.config_path)
    preflight(settings, config)
    return launch(settings)


if __name__ == "__main__":
    raise SystemExit(main())
