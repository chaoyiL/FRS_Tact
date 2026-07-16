from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset


def create_dataset(root: Path, *, frames: int = 50) -> None:
    if root.exists():
        shutil.rmtree(root)
    features = {
        "observation.images.camera1": {
            "dtype": "image",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera2": {
            "dtype": "image",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera3": {
            "dtype": "image",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["action"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id="smolvla_likelihood_smoke",
        root=root,
        fps=30,
        robot_type="smoke_test",
        features=features,
        use_videos=False,
        image_writer_threads=1,
    )
    rng = np.random.default_rng(0)
    for frame in range(frames):
        dataset.add_frame(
            {
                "observation.images.camera1": rng.integers(
                    0, 256, (64, 64, 3), dtype=np.uint8
                ),
                "observation.images.camera2": rng.integers(
                    0, 256, (64, 64, 3), dtype=np.uint8
                ),
                "observation.images.camera3": rng.integers(
                    0, 256, (64, 64, 3), dtype=np.uint8
                ),
                "observation.state": np.full(6, frame / frames, dtype=np.float32),
                "actions": np.linspace(-0.2, 0.2, 6, dtype=np.float32),
                "task": "Move the robot arm.",
            }
        )
    dataset.save_episode()
    dataset.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny SmolVLA smoke-test dataset")
    parser.add_argument("root", type=Path)
    parser.add_argument("--frames", type=int, default=50)
    args = parser.parse_args()
    create_dataset(args.root.expanduser().resolve(), frames=args.frames)


if __name__ == "__main__":
    main()

