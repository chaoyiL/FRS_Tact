"""Dispatch SmolVLA deployment to the PyTorch vision or JAX FRS runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _backend(config_path: Path) -> str:
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config root must be a mapping: {config_path}")
    return str(config.get("backend", "pytorch_smolvla"))


def run(config_path: Path, max_iterations_override: int | None = None) -> None:
    backend = _backend(config_path)
    if backend == "pytorch_smolvla":
        from .pytorch_remote_client import run as run_pytorch

        run_pytorch(config_path, max_iterations_override=max_iterations_override)
        return
    if backend in {"jax_smolvla", "jax_smolvla_frs", "direct_tactile_decoder"}:
        from .jax_remote_client import run as run_jax

        run_jax(config_path, max_iterations_override=max_iterations_override)
        return
    raise ValueError(f"Unsupported SmolVLA deployment backend: {backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, max_iterations_override=args.max_iterations)


def __getattr__(name: str) -> Any:
    """Keep legacy diagnostic helpers available without loading JAX eagerly."""

    if name.startswith("__"):
        raise AttributeError(name)
    from . import jax_remote_client

    return getattr(jax_remote_client, name)


if __name__ == "__main__":
    main()
