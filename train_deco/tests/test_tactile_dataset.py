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
    _resolve_episode_task_ids,
    _validate_info,
    build_lerobot_vision_datasets,
)
from train_deco.prepare_lerobot_multiroot import write_multiroot_manifest
from train_deco.state_action_profiles import (
    SINGLE_RIGHT_ARM_7X10,
    SINGLE_RIGHT_ARM_PROFILE,
)


def _jpeg_bytes(rgb: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (5, 3), color=rgb)
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=100, subsampling=0)
    return encoded.getvalue()


def _info(
    include_tactile: bool = True,
    *,
    state_dim: int = 20,
    action_dim: int = 20,
) -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": [state_dim]},
        "actions": {"dtype": "float32", "shape": [action_dim]},
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


def _write_fixture(
    root: Path,
    *,
    state_dim: int = 20,
    action_dim: int = 20,
    episode_tasks: list[int | str] | None = None,
    camera0_color: tuple[int, int, int] = (16, 32, 64),
) -> None:
    (root / "meta").mkdir(parents=True)
    info = _info(state_dim=state_dim, action_dim=action_dim)
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick tube"}) + "\n",
        encoding="utf-8",
    )
    episodes = []
    episode_tasks = [0] if episode_tasks is None else episode_tasks
    tactile_colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
    for episode_index in range(2):
        episodes.append(
            {
                "episode_index": episode_index,
                "length": 3,
                "tasks": episode_tasks,
            }
        )
        columns = {
            "observation.state": [
                np.full(state_dim, row, dtype=np.float32) for row in range(3)
            ],
            "actions": [
                np.full(action_dim, 1.0, dtype=np.float32),
                np.full(action_dim, 2.0, dtype=np.float32),
                np.zeros(action_dim, dtype=np.float32),
            ],
            "frame_index": list(range(3)),
            "episode_index": [episode_index] * 3,
            "task_index": [0] * 3,
        }
        columns[CAMERA_NAMES[0]] = [{"bytes": _jpeg_bytes(camera0_color)}] * 3
        columns[CAMERA_NAMES[1]] = [{"bytes": _jpeg_bytes((16, 32, 64))}] * 3
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


def test_single_right_arm_manifest_is_explicit_and_builds_7x10_dataset(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "insert_01"
    manifest_path = tmp_path / "insert_01.json"
    _write_fixture(
        dataset_root,
        state_dim=7,
        action_dim=10,
        episode_tasks=["pick tube"],
    )

    with pytest.raises(ValueError, match="explicit handedness"):
        write_multiroot_manifest(
            [dataset_root],
            manifest_path,
            dataset_id="insert_01",
        )

    manifest = write_multiroot_manifest(
        [dataset_root],
        manifest_path,
        dataset_id="insert_01",
        state_action_profile=SINGLE_RIGHT_ARM_PROFILE,
    )
    assert manifest["state_action_profile"] == SINGLE_RIGHT_ARM_PROFILE

    train, val = build_lerobot_vision_datasets(
        manifest_path,
        action_chunk_size=2,
        validation_ratio=0.5,
        split_seed=0,
        include_tactile=False,
    )
    sample = train[0]
    assert sample["observation"].shape == (7,)
    assert sample["action"].shape == (2, 10)
    assert val.metadata == train.metadata
    assert train.metadata["state_action_profile"] == SINGLE_RIGHT_ARM_PROFILE
    assert train.metadata["controlled_arms"] == ["right"]
    assert train.metadata["state_columns"] == list(
        SINGLE_RIGHT_ARM_7X10.state_columns
    )
    assert train.metadata["action_columns"] == list(
        SINGLE_RIGHT_ARM_7X10.action_columns
    )


def test_single_right_arm_multiroot_accepts_standard_string_episode_tasks(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / "insert_01", tmp_path / "insert_02"]
    manifest_path = tmp_path / "insert_01_02.json"
    for root in roots:
        _write_fixture(
            root,
            state_dim=7,
            action_dim=10,
            episode_tasks=["perform manipulation task"],
            camera0_color=(0, 0, 0),
        )

    manifest = write_multiroot_manifest(
        roots,
        manifest_path,
        dataset_id="insert_01_02",
        state_action_profile=SINGLE_RIGHT_ARM_PROFILE,
        require_black_camera0=True,
    )
    assert [source["name"] for source in manifest["sources"]] == [
        "insert_01",
        "insert_02",
    ]
    assert [source["camera0_black_preflight"]["frames"] for source in manifest["sources"]] == [
        6,
        6,
    ]

    train, val = build_lerobot_vision_datasets(
        manifest_path,
        action_chunk_size=2,
        validation_ratio=0.5,
        split_seed=0,
        include_tactile=False,
    )
    assert train[0]["observation"].shape == (7,)
    assert train[0]["action"].shape == (2, 10)
    assert {row["source"] for row in train.manifest["splits"]["train"]["episodes"]} == {
        "insert_01",
        "insert_02",
    }
    assert {row["source"] for row in val.manifest["splits"]["val"]["episodes"]} == {
        "insert_01",
        "insert_02",
    }


def test_camera0_black_preflight_rejects_nonblack_pixels(tmp_path: Path) -> None:
    dataset_root = tmp_path / "insert_01"
    _write_fixture(
        dataset_root,
        state_dim=7,
        action_dim=10,
        camera0_color=(32, 0, 0),
    )

    with pytest.raises(ValueError, match="camera0.*not pure black"):
        write_multiroot_manifest(
            [dataset_root],
            tmp_path / "insert_01.json",
            state_action_profile=SINGLE_RIGHT_ARM_PROFILE,
            require_black_camera0=True,
        )


def test_camera0_black_preflight_rejects_duplicate_episode_metadata(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "insert_01"
    _write_fixture(
        dataset_root,
        state_dim=7,
        action_dim=10,
        camera0_color=(0, 0, 0),
    )
    episodes_path = dataset_root / "meta/episodes.jsonl"
    first_episode = episodes_path.read_text(encoding="utf-8").splitlines()[0]
    episodes_path.write_text(f"{first_episode}\n{first_episode}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate episode_index"):
        write_multiroot_manifest(
            [dataset_root],
            tmp_path / "insert_01.json",
            state_action_profile=SINGLE_RIGHT_ARM_PROFILE,
            require_black_camera0=True,
        )


def test_camera0_black_preflight_rejects_episode_row_count_mismatch(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "insert_01"
    _write_fixture(
        dataset_root,
        state_dim=7,
        action_dim=10,
        camera0_color=(0, 0, 0),
    )
    episode_path = dataset_root / "data/chunk-000/episode_000000.parquet"
    pq.write_table(pq.read_table(episode_path).slice(0, 2), episode_path)

    with pytest.raises(ValueError, match="camera0 row count disagrees"):
        write_multiroot_manifest(
            [dataset_root],
            tmp_path / "insert_01.json",
            state_action_profile=SINGLE_RIGHT_ARM_PROFILE,
            require_black_camera0=True,
        )


def test_standard_string_task_label_maps_to_declared_task_id(tmp_path: Path) -> None:
    assert _resolve_episode_task_ids(
        ["insert"],
        {3},
        {"insert": 3},
        root=tmp_path,
        episode_index=0,
    ) == (3,)


def test_unknown_string_task_is_rejected_when_mapping_is_ambiguous(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Unknown LeRobot episode task label"):
        _resolve_episode_task_ids(
            ["generic task"],
            {0, 1},
            {"insert": 0, "remove": 1},
            root=tmp_path,
            episode_index=0,
        )


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
