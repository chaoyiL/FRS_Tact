from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence

from tactile_flow_steering.utils.history_plot import plot_training_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot tactile flow steering training curves from history.csv."
    )
    parser.add_argument(
        "--history-path",
        type=pathlib.Path,
        required=True,
        help="Path to history.csv produced by tactile_flow_steering.train.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Output PNG path (default: <history-dir>/training_curves.png).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_path = plot_training_history(args.history_path, output_path=args.output)
    print(f"plot={output_path}")


if __name__ == "__main__":
    main()
