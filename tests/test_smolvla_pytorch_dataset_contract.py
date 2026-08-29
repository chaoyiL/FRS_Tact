from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from train_smolvla.torch_train import _select_dataset_cameras, validate_dataset_contract


def _config(root: Path) -> dict:
    return {
        "dataset": {
            "expected_fps": 30,
            "state_dim": 20,
            "action_dim": 20,
            "image_keys": [
                "observation.images.camera0",
                "observation.images.camera1",
            ],
        },
        "datasets": [{"repo_id": "test/dataset", "root": str(root)}],
    }


def test_visual_contract_allows_and_prunes_tactile_cameras(tmp_path: Path) -> None:
    features = {
        "observation.state": {"dtype": "float32", "shape": [20]},
        "action": {"dtype": "float32", "shape": [20]},
        "observation.images.camera0": {"dtype": "video", "shape": [3, 480, 640]},
        "observation.images.camera1": {"dtype": "video", "shape": [3, 480, 640]},
        "observation.images.tactile_left_0": {
            "dtype": "video",
            "shape": [3, 224, 224],
        },
    }
    info = {"codebase_version": "v3.0", "fps": 30, "features": features}
    info_path = tmp_path / "meta/info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(json.dumps(info), encoding="utf-8")

    validate_dataset_contract(_config(tmp_path))

    metadata = SimpleNamespace(
        features=features,
        info={"features": dict(features)},
        stats={key: {"mean": [0.0]} for key in features},
    )
    dataset = SimpleNamespace(
        meta=metadata,
        delta_timestamps={key: [0.0] for key in features},
        reader=SimpleNamespace(delta_indices={key: [0] for key in features}),
    )
    _select_dataset_cameras(
        dataset,
        {"observation.images.camera0", "observation.images.camera1"},
    )
    assert "observation.images.tactile_left_0" not in metadata.info["features"]
    assert "observation.images.tactile_left_0" not in metadata.stats
    assert "observation.images.tactile_left_0" not in dataset.delta_timestamps
    assert "observation.images.tactile_left_0" not in dataset.reader.delta_indices
