"""Leave-one-out configs for the four optional gated FRS losses."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

LOSS_ABLATION_SWITCHES: tuple[str, ...] = (
    "aux_decode",
    "low_gate_safety",
    "rank",
    "repair",
)
DEFAULT_ABLATION_REPAIR_WEIGHT = 2.0


def build_loss_ablation_run(
    base_config: Mapping[str, Any],
    *,
    disabled: str,
    output_root: Path,
) -> dict[str, Any]:
    """Copy ``base_config`` and turn off exactly one optional loss."""

    if disabled not in LOSS_ABLATION_SWITCHES:
        raise ValueError(
            f"unknown ablation switch {disabled!r}; "
            f"expected one of {list(LOSS_ABLATION_SWITCHES)}"
        )
    config = copy.deepcopy(dict(base_config))
    training = dict(config.get("frs_training") or {})
    for switch in LOSS_ABLATION_SWITCHES:
        training[switch] = switch != disabled
    if training["repair"] and float(training.get("repair_weight") or 0.0) <= 0.0:
        # Official yaml currently keeps repair_weight at 0. Ablation still needs
        # a real repair term when that switch is supposed to stay on.
        training["repair_weight"] = DEFAULT_ABLATION_REPAIR_WEIGHT
    run_dir = (Path(output_root) / f"no_{disabled}").expanduser().resolve()
    training["output"] = str(run_dir)
    training["resume"] = "auto"
    config["frs_training"] = training
    return config


def build_loss_ablation_runs(
    base_config: Mapping[str, Any],
    *,
    output_root: Path,
) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            f"no_{disabled}",
            build_loss_ablation_run(
                base_config,
                disabled=disabled,
                output_root=output_root,
            ),
        )
        for disabled in LOSS_ABLATION_SWITCHES
    ]


def write_loss_ablation_configs(
    base_config: Mapping[str, Any],
    *,
    output_root: Path,
) -> list[tuple[str, Path]]:
    written: list[tuple[str, Path]] = []
    for name, config in build_loss_ablation_runs(base_config, output_root=output_root):
        path = Path(output_root).expanduser().resolve() / name / "ablation.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        written.append((name, path))
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write leave-one-out FRS loss ablation YAMLs under checkpoints/frs."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8") as file:
        base = yaml.safe_load(file) or {}
    if not isinstance(base, dict):
        raise ValueError(f"config root must be a mapping: {args.config}")
    for name, path in write_loss_ablation_configs(base, output_root=args.output_root):
        print(f"{name}\t{path}")


if __name__ == "__main__":
    main()
