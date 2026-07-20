from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence

from utils.cache import finalize_partial_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize an incomplete flow decoder cache so training can use completed samples."
    )
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--no-resplit",
        action="store_true",
        help="Keep existing per-sample split labels instead of re-splitting present episodes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = finalize_partial_cache(args.cache_dir, resplit=not args.no_resplit)
    print(
        f"cache finalized: {args.cache_dir} "
        f"samples={manifest['sample_count']} "
        f"train={manifest['train_sample_count']} "
        f"val={manifest['val_sample_count']}"
    )
    print(f"mean_source_inversion_mse={manifest['mean_source_inversion_mse']:.8f}")


if __name__ == "__main__":
    main()
