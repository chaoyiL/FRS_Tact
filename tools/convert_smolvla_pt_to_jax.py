#!/usr/bin/env python
"""Convert a LeRobot SmolVLA checkpoint to a JAX-loadable checkpoint."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_smolvla.checkpoint import (
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
