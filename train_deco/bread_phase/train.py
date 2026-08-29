"""Bread-only entrypoint for one phase-conditioned DECO checkpoint."""

from __future__ import annotations

import sys

from train_deco import train as generic_train

from .augmentation import BREAD_AUGMENTATION_ARGS


_BREAD_MODEL_ARGS = (
    "--stage", "1",
    "--dataset-format", "lerobot-v21",
    "--action-chunk-size", "32",
    "--bread-phase",
    "--use-task-condition",
)


def build_training_argv(argv: list[str]) -> list[str]:
    if "--augmentation-preset" in argv:
        raise ValueError("Bread phase training uses its fixed 0.8-1.2 augmentation")
    return [*argv, *_BREAD_MODEL_ARGS, *BREAD_AUGMENTATION_ARGS]


def main(argv: list[str] | None = None) -> None:
    generic_train.main(build_training_argv(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    main()

