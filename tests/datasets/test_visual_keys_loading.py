from __future__ import annotations

import os
from pathlib import Path

import pytest

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.transforms import build_image_transforms

DATASET_ROOT = Path(
    os.environ.get(
        "LEROBOT_VISUAL_KEYS_TEST_DATASET",
        "/root/.cache/huggingface/dataset/lerobot_v30/chaoyi/tactile_test_05",
    )
)


def _dataset_is_available() -> bool:
    try:
        return DATASET_ROOT.is_dir()
    except OSError:
        return False


@pytest.mark.skipif(not _dataset_is_available(), reason="local tactile dataset not present")
def test_visual_keys_skips_unused_image_columns_and_transforms() -> None:
    visual_keys = [
        "observation.images.camera0",
        "observation.images.camera1",
    ]
    tf = build_image_transforms({"enable": True, "max_num_transforms": 2})
    ds = LeRobotDataset(
        "chaoyi/tactile_test_05",
        root=DATASET_ROOT,
        episodes=[0],
        image_transforms=tf,
        visual_keys=visual_keys,
    )
    assert set(ds.reader.visual_keys) == set(visual_keys)
    assert set(ds.hf_dataset.column_names).issuperset(visual_keys)
    assert "observation.images.tactile_left_0" not in ds.hf_dataset.column_names

    sample = ds[0]
    assert "observation.images.camera0" in sample
    assert "observation.images.camera1" in sample
    assert "observation.images.tactile_left_0" not in sample
