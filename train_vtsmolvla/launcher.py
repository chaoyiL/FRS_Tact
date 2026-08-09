"""Launch VT-SmolVLA training and its tactile cache stage from one YAML."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from train_smolvla import launcher as shared_launcher
from train_smolvla.launcher import (
    LauncherSettings,
    find_uv,
    launcher_settings_from_config,
    preflight as shared_preflight,
    stream_command,
    timestamped_log_path,
)
from train_smolvla.train import load_yaml_config
from train_vtsmolvla.checkpoint import resolve_checkpoint
from train_vtsmolvla.train import (
    DEFAULT_CONFIG,
    VT_ALLOWED_TOP_LEVEL_KEYS,
    _validate_vt_config,
)


def load_launcher_settings(
    config_path: Path,
    project_root: Path,
) -> tuple[LauncherSettings, dict[str, Any]]:
    """Load VT YAML and construct the common launcher settings."""

    root = project_root.expanduser().resolve()
    resolved = config_path if config_path.is_absolute() else root / config_path
    resolved = resolved.expanduser().resolve()
    config = load_yaml_config(
        resolved,
        allowed_top_level_keys=VT_ALLOWED_TOP_LEVEL_KEYS,
    )
    return launcher_settings_from_config(resolved, root, config), config


def _project_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def preflight(settings: LauncherSettings, config: dict[str, Any]) -> Path:
    """Validate VT configuration and encoder before shared launch checks."""

    _validate_vt_config(settings.config_path)
    model = config["model"]
    encoder = _project_path(settings.project_root, model["tactile_encoder_path"]).resolve()
    if not encoder.is_dir():
        raise FileNotFoundError(
            f"tactile encoder directory does not exist: {encoder}; update model.tactile_encoder_path"
        )
    return shared_preflight(
        settings,
        config,
        checkpoint_resolver=resolve_checkpoint,
    )


def _precompute_command(settings: LauncherSettings, uv_bin: str) -> list[str]:
    return [
        uv_bin,
        "run",
        "--no-sync",
        "python",
        "-m",
        "train_vtsmolvla.precompute",
        "--config",
        str(settings.config_path),
    ]


def _training_command(settings: LauncherSettings, uv_bin: str) -> list[str]:
    return [
        uv_bin,
        "run",
        "--no-sync",
        "python",
        "-m",
        "train_vtsmolvla.train",
        "--config",
        str(settings.config_path),
    ]


def run_pipeline(
    settings: LauncherSettings,
    config: dict[str, Any],
    *,
    uv_bin: str,
    now: datetime | None = None,
) -> int:
    """Run enabled cache precomputation, then training, with separate logs."""

    instant = now or datetime.now()
    cache = config.get("tactile_embedding_cache") or {}
    if bool(cache.get("enabled", False)):
        status = stream_command(
            _precompute_command(settings, uv_bin),
            cwd=settings.project_root,
            log_path=timestamped_log_path(settings, instant, kind="precompute"),
        )
        if status != 0:
            return status
    return stream_command(
        _training_command(settings, uv_bin),
        cwd=settings.project_root,
        log_path=timestamped_log_path(settings, instant, kind="train"),
    )


def launch(settings: LauncherSettings, config: dict[str, Any]) -> int:
    """Use the shared tmux handoff or run the VT pipeline in this process."""

    if shared_launcher.maybe_launch_tmux(
        settings,
        foreground_env="VTSMOLVLA_FOREGROUND",
        launcher_module="train_vtsmolvla.launcher",
        log_prefix="vtsmolvla",
    ):
        return 0
    return run_pipeline(settings, config, uv_bin=find_uv())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    settings, config = load_launcher_settings(args.config, project_root)
    preflight(settings, config)
    return launch(settings, config)


if __name__ == "__main__":
    raise SystemExit(main())
