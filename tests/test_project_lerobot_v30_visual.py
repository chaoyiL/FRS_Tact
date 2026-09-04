from __future__ import annotations

import json
from pathlib import Path

from datasets import Features, Image
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tools.project_lerobot_v30_visual import project_dataset


RGB_KEYS = (
    "observation.images.camera0",
    "observation.images.camera1",
)
TACTILE_KEY = "observation.images.tactile_left_0"


def _write_image_parquet(path: Path) -> None:
    image_value = {"bytes": b"encoded-image", "path": None}
    frame = pd.DataFrame(
        {
            "observation.state": [[0.0, 1.0], [2.0, 3.0]],
            "action": [[1.0, 2.0], [3.0, 4.0]],
            RGB_KEYS[0]: [image_value, image_value],
            RGB_KEYS[1]: [image_value, image_value],
            TACTILE_KEY: [image_value, image_value],
            "episode_index": [0, 0],
        }
    )
    features = Features.from_arrow_schema(pa.Schema.from_pandas(frame))
    for key in (*RGB_KEYS, TACTILE_KEY):
        features[key] = Image()
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False, schema=features.arrow_schema)


def test_project_v30_dataset_keeps_only_selected_rgb_streams(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    data_path = root / "data/chunk-000/file-000.parquet"
    episode_path = root / "meta/episodes/chunk-000/file-000.parquet"
    _write_image_parquet(data_path)
    episode_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [0],
            "dataset_from_index": [0],
            "dataset_to_index": [2],
            f"stats/{TACTILE_KEY}/count": [2],
        }
    ).to_parquet(episode_path, index=False)

    features = {
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [2]},
        RGB_KEYS[0]: {"dtype": "image", "shape": [3, 2, 2]},
        RGB_KEYS[1]: {"dtype": "image", "shape": [3, 2, 2]},
        TACTILE_KEY: {"dtype": "image", "shape": [3, 2, 2]},
        "episode_index": {"dtype": "int64", "shape": [1]},
    }
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "features": features,
                "video_path": None,
            }
        ),
        encoding="utf-8",
    )
    (root / "meta/stats.json").write_text(
        json.dumps({key: {"count": [2]} for key in features}),
        encoding="utf-8",
    )

    project_dataset(root, list(RGB_KEYS))

    assert set(pq.read_schema(data_path).names) == {
        "observation.state",
        "action",
        *RGB_KEYS,
        "episode_index",
    }
    assert f"stats/{TACTILE_KEY}/count" not in pq.read_schema(episode_path).names
    projected_info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    projected_stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
    assert TACTILE_KEY not in projected_info["features"]
    assert TACTILE_KEY not in projected_stats
    project_dataset(root, list(RGB_KEYS), check=True)
