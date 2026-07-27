#!/usr/bin/env python
"""Filter a prepared cache by dropping high pred→gt MSE samples.

Keeps samples with MSE(predicted_actions, gt_actions) <= --max-mse (default 1.0)
and writes a new complete cache directory.

Example:
  uv run python filter_cache.py \\
    --cache-dir cache/tactile_test_05 \\
    --output-dir cache/tactile_test_05_mse1 \\
    --max-mse 1.0
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cache import filter_cache_by_mse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True, help="Source cache directory.")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Destination cache directory (must not already exist / must be empty).",
    )
    parser.add_argument(
        "--max-mse",
        type=float,
        default=1.0,
        help="Keep samples with MSE(pred, gt) <= this threshold (default: 1.0).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = filter_cache_by_mse(args.cache_dir, args.output_dir, max_mse=args.max_mse)
    filt = manifest["filter"]
    print(f"source={args.cache_dir.resolve()}")
    print(f"output={args.output_dir.resolve()}")
    print(
        f"kept={filt['kept_sample_count']}/{filt['source_sample_count']} "
        f"(dropped={filt['dropped_sample_count']}) max_mse={filt['max_mse']}"
    )
    print(
        f"kept_mse: mean={filt['kept_mse_mean']:.6f} "
        f"median={filt['kept_mse_median']:.6f} max={filt['kept_mse_max']:.6f}"
    )
    print(f"split: train={manifest['train_sample_count']} val={manifest['val_sample_count']}")


if __name__ == "__main__":
    main()
