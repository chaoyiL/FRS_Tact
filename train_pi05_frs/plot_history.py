from __future__ import annotations

import argparse
import csv
import pathlib
from collections.abc import Sequence

from train_pi05_frs.utils.bimanual_visualize import (
    plot_bimanual_behavior,
    plot_bimanual_training_overview,
)
from train_pi05_frs.utils.history_plot import plot_training_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot tactile flow steering training curves from history.csv."
    )
    parser.add_argument(
        "--history-path",
        type=pathlib.Path,
        required=True,
        help="Path to history.csv produced by train_pi05_frs.train.",
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
    with args.history_path.open(encoding="utf-8", newline="") as file:
        fieldnames = set(csv.DictReader(file).fieldnames or ())
    if {"train_gate_w_left", "train_gate_w_right"}.issubset(fieldnames):
        output_dir = output_path.parent
        try:
            overview = plot_bimanual_training_overview(
                args.history_path,
                output_path=output_dir / "training_overview.png",
            )
            behavior = plot_bimanual_behavior(
                args.history_path,
                output_path=output_dir / "bimanual_behavior.png",
            )
        except Exception as exc:
            print(f"warning: could not render bimanual history dashboards: {exc}")
        else:
            print(f"plot={overview}")
            print(f"plot={behavior}")


if __name__ == "__main__":
    main()
