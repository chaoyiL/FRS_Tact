"""Dedicated exporter for the single-weight, two-phase Bread policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_deco.export_torchscript import (
    BREAD_PHASE_VERSION,
    export_checkpoint,
)


def export_bread_phase_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path,
    image_height: int = 224,
    image_width: int = 224,
    device: str = "cpu",
) -> dict:
    """Export only checkpoints trained by the dedicated Bread phase path."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    if config.get("bread_phase_version") != BREAD_PHASE_VERSION:
        raise ValueError("Bread phase export requires bread-phase-v1 checkpoint metadata")
    return export_checkpoint(
        checkpoint_path,
        output_path,
        image_height,
        image_width,
        device,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "event": "bread_phase_torchscript_export_complete",
                **export_bread_phase_checkpoint(
                    args.checkpoint,
                    args.output,
                    args.image_height,
                    args.image_width,
                    args.device,
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
