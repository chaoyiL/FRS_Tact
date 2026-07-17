#!/usr/bin/env python
"""Convert a LeRobot SmolVLA checkpoint to a JAX-loadable checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.policies.smolvla_jax.checkpoint import (
    load_safetensors_params,
    parameter_summary,
    resolve_checkpoint,
    save_orbax_params,
    save_portable_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="PyTorch checkpoint directory or Hub repo id")
    parser.add_argument("--output", required=True, type=Path, help="JAX checkpoint directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument(
        "--format",
        choices=("safetensors", "orbax"),
        default="safetensors",
        help="Output storage format. Safetensors is portable and the default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = resolve_checkpoint(args.source)
    params = load_safetensors_params(source)
    summary = parameter_summary(params)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.inspect_only:
        return
    save = save_orbax_params if args.format == "orbax" else save_portable_params
    output = save(
        params,
        args.output,
        source_dir=source,
        overwrite=args.overwrite,
    )
    print(f"JAX checkpoint written to {output}")


if __name__ == "__main__":
    main()
