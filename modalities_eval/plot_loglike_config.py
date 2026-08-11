from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "plot_loglike_modalities.yaml"

_PATH_KEYS = (
    "checkpoint_dir",
    "dataset_root",
    "output_dir",
    "output_path",
    "compare_reverse_dir",
)


def load_yaml_config(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def _section(cfg: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = cfg.get(name)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return dict(value)


def _coerce_path(value: Any) -> pathlib.Path | None:
    if value in (None, ""):
        return None
    return pathlib.Path(value)


def _coerce_rename_map(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(dict(value))
    raise ValueError("rename_map must be null, a JSON string, or a mapping")


def flatten_plot_loglike_defaults(
    cfg: Mapping[str, Any],
    *,
    script: str,
) -> dict[str, Any]:
    """Flatten shared + script-specific YAML sections into argparse defaults.

    ``script`` is ``\"reverse\"`` or ``\"forward\"``.
    """

    if script not in ("reverse", "forward"):
        raise ValueError(f"script must be 'reverse' or 'forward', got {script!r}")

    defaults: dict[str, Any] = {}
    for section_name in ("data", "integration", script):
        for key, value in _section(cfg, section_name).items():
            if key in _PATH_KEYS:
                defaults[key] = _coerce_path(value)
            elif key == "rename_map":
                defaults[key] = _coerce_rename_map(value)
            else:
                defaults[key] = value
    return defaults


def add_config_argument(
    parser: argparse.ArgumentParser,
    *,
    default: pathlib.Path = DEFAULT_CONFIG,
) -> None:
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=default,
        help=f"YAML config path (default: {default})",
    )


def parse_args_with_config(
    build_parser,
    *,
    script: str,
    argv: Sequence[str] | None = None,
    default_config: pathlib.Path = DEFAULT_CONFIG,
) -> argparse.Namespace:
    """Parse CLI with YAML defaults from ``data`` + ``integration`` + script section."""

    argv_list = list(argv) if argv is not None else None
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=pathlib.Path, default=default_config)
    pre_args, _ = pre.parse_known_args(argv_list)

    parser = build_parser()
    cfg = load_yaml_config(pre_args.config)
    parser.set_defaults(**flatten_plot_loglike_defaults(cfg, script=script))
    return parser.parse_args(argv_list)
