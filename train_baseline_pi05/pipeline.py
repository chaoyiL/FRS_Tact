"""Ordered, process-isolated cache and decoder training pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .config import BaselineTrainConfig, load_config


_STAGES = (
    ("tactile_cache", "train_baseline_pi05.tactile_cache", "max_samples"),
    ("prepare_action_cache", "train_baseline_pi05.prepare_action_cache", "max_samples"),
    ("train", "train_baseline_pi05.train", "max_steps"),
)


def _positive(value: int | None, name: str) -> int | None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when provided")
    return value


def _resolved(config: BaselineTrainConfig, path: Path) -> Path:
    return (path if path.is_absolute() else config.config_path.parent / path).resolve()


def _input_status(config: BaselineTrainConfig) -> dict[str, dict[str, object]]:
    inputs = {
        "dataset": _resolved(config, config.dataset.root),
        "checkpoint": _resolved(config, config.source.checkpoint),
        "norm_stats": _resolved(config, config.source.norm_stats_dir),
        "encoder": _resolved(config, config.tactile.encoder_checkpoint),
    }
    result: dict[str, dict[str, object]] = {}
    for name, path in inputs.items():
        readable = path.exists() and os.access(path, os.R_OK)
        result[name] = {"path": str(path), "local": True, "readable": readable}
        if not readable:
            raise FileNotFoundError(f"configured local {name} input is not readable: {path}")
    return result


def check_config(config_path: Path, *, max_samples: int | None = None, max_steps: int | None = None) -> dict[str, object]:
    """Parse and report readiness without importing runtimes or writing state."""
    _positive(max_samples, "max_samples")
    _positive(max_steps, "max_steps")
    config = load_config(config_path)
    report: dict[str, object] = {
        "mode": "check",
        "config": str(config.config_path),
        "inputs": _input_status(config),
        "destinations": {
            "action_cache": str(_resolved(config, config.cache.action_root)),
            "tactile_cache": str(_resolved(config, config.cache.tactile_root)),
            "decoder": str(_resolved(config, config.decoder.output)),
        },
        "overrides": {"max_samples": max_samples, "max_steps": max_steps},
    }
    print(json.dumps(report, sort_keys=True))
    return report


def _write_run_metadata(config: BaselineTrainConfig, metadata: dict[str, object]) -> Path:
    output = _resolved(config, config.decoder.output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "pipeline_run.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def run_pipeline(config_path: str | Path, *, check: bool = False, max_samples: int | None = None, max_steps: int | None = None) -> dict[str, object] | None:
    """Run JAX cache producers before the separate PyTorch decoder process."""
    path = Path(config_path).resolve()
    if check:
        return check_config(path, max_samples=max_samples, max_steps=max_steps)
    _positive(max_samples, "max_samples")
    _positive(max_steps, "max_steps")
    config = load_config(path)
    overrides = {"max_samples": max_samples, "max_steps": max_steps}
    for stage_index, (_name, module, override) in enumerate(_STAGES, start=1):
        command = [sys.executable, "-m", module, "--config", str(config.config_path)]
        value = overrides[override]
        if value is not None:
            command.extend(["--" + override.replace("_", "-"), str(value)])
        print(f"[Pipeline {stage_index}/{len(_STAGES)}] Starting {_name}", flush=True)
        subprocess.run(command, check=True)
        print(f"[Pipeline {stage_index}/{len(_STAGES)}] Finished {_name}", flush=True)
    _write_run_metadata(config, {
        "config": str(config.config_path),
        "max_samples": max_samples,
        "max_steps": max_steps,
        "stages": [name for name, _module, _override in _STAGES],
    })
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check", action="store_true", help="read-only local input readiness report")
    parser.add_argument("--max-samples", type=int, help="positive cache-producer sample cap")
    parser.add_argument("--max-steps", type=int, help="positive decoder optimizer-step cap")
    args = parser.parse_args()
    run_pipeline(args.config, check=args.check, max_samples=args.max_samples, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
