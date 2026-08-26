import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from train_deco.lerobot_vision_dataset import (
    CAMERA_NAMES,
    TACTILE_NAMES,
    _validate_info,
    build_lerobot_vision_datasets,
)


def _jpeg_bytes(rgb: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (5, 3), color=rgb)
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=100, subsampling=0)
    return encoded.getvalue()


def _info(include_tactile: bool = True) -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": [20]},
        "actions": {"dtype": "float32", "shape": [20]},
        **{
            name: {"dtype": "image", "shape": [3, 5, 3]}
            for name in CAMERA_NAMES
        },
    }
    if include_tactile:
        features.update(
            {
                name: {"dtype": "image", "shape": [3, 5, 3]}
                for name in TACTILE_NAMES
            }
        )
    return {
        "codebase_version": "v2.1",
        "video_path": None,
        "total_videos": 0,
        "fps": 30,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "chunks_size": 1000,
        "total_episodes": 2,
        "total_frames": 6,
        "total_tasks": 1,
        "features": features,
    }


def _write_fixture(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    info = _info()
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick tube"}) + "\n",
        encoding="utf-8",
    )
    episodes = []
    tactile_colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
    for episode_index in range(2):
        episodes.append({"episode_index": episode_index, "length": 3, "tasks": [0]})
        columns = {
            "observation.state": [np.full(20, row, dtype=np.float32) for row in range(3)],
            "actions": [
                np.full(20, 1.0, dtype=np.float32),
                np.full(20, 2.0, dtype=np.float32),
                np.zeros(20, dtype=np.float32),
            ],
            "frame_index": list(range(3)),
            "episode_index": [episode_index] * 3,
            "task_index": [0] * 3,
        }
        for name in CAMERA_NAMES:
            columns[name] = [{"bytes": _jpeg_bytes((16, 32, 64))}] * 3
        for name, color in zip(TACTILE_NAMES, tactile_colors):
            columns[name] = [{"bytes": _jpeg_bytes(color)}] * 3
        path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), path)
    (root / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes),
        encoding="utf-8",
    )


def test_stage2_schema_requires_each_named_rgb_tactile_field(tmp_path: Path) -> None:
    info = _info()
    missing = TACTILE_NAMES[2]
    del info["features"][missing]

    with pytest.raises(ValueError, match=missing):
        _validate_info(tmp_path, info, include_tactile=True)

    info = _info()
    wrong = TACTILE_NAMES[1]
    info["features"][wrong]["shape"] = [3, 5, 1]

    with pytest.raises(ValueError, match=wrong):
        _validate_info(tmp_path, info, include_tactile=True)

    info = _info()
    not_image = TACTILE_NAMES[0]
    info["features"][not_image]["dtype"] = "float32"

    with pytest.raises(ValueError, match=not_image):
        _validate_info(tmp_path, info, include_tactile=True)


def test_stage1_schema_accepts_the_existing_two_camera_contract(tmp_path: Path) -> None:
    assert _validate_info(tmp_path, _info(include_tactile=False), include_tactile=False) == (3, 5)


def test_stage2_dataset_returns_four_tactile_images_in_stable_order(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    train, _ = build_lerobot_vision_datasets(
        tmp_path,
        action_chunk_size=2,
        validation_ratio=0.5,
        split_seed=0,
        include_tactile=True,
    )
    sample = train[0]

    assert train.metadata["tactile_names"] == list(TACTILE_NAMES)
    assert sample["images"].shape == (2, 3, 3, 5)
    assert sample["tactile_images"].shape == (4, 3, 3, 5)
    dominant_channels = sample["tactile_images"].mean(dim=(2, 3)).argmax(dim=1)
    assert torch.equal(dominant_channels, torch.tensor([0, 1, 2, 0]))


@pytest.mark.skipif(
    not Path("data/lerobot_v30/KaiyueChen/pick_tube_06_old/meta/info.json").is_file(),
    reason="local v2.1 tactile dataset metadata is unavailable",
)
def test_local_v21_metadata_declares_all_known_224_rgb_tactile_fields() -> None:
    info_path = Path("data/lerobot_v30/KaiyueChen/pick_tube_06_old/meta/info.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))

    _validate_info(info_path.parent.parent, info, include_tactile=True)
    assert [info["features"][name]["shape"] for name in TACTILE_NAMES] == [
        [224, 224, 3]
    ] * 4
