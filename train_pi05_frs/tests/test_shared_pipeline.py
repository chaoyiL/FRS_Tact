from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from train_pi05_frs.tools.train_shared_frs import DEFAULT_CONFIG, load_shared_config, native_config

ROOT = Path(__file__).resolve().parents[2]


def test_native_configs_preserve_each_vla_interface():
    config = load_shared_config(DEFAULT_CONFIG)
    for source in config["sources"]:
        source = {
            **source,
            "checkpoint": f"/workspace/checkpoints/{source['name']}",
            "norm_stats_asset_id": "task_stats",
        }
        native, record = native_config(config, source, check_paths=False)
        assert native["checkpoint"].endswith(source["name"])
        assert native["model"]["action_dim"] == source["native_action_dim"]
        assert native["model"]["freeze_tactile_encoder"] is True
        assert native["action_cache"]["num_train_episodes"] == 300
        assert native["action_cache"]["num_val_episodes"] == 35
        assert record["active_wrists"] == source["active_wrists"]
        assert native["model"]["state_dim"] == (7 if source["native_action_dim"] == 10 else 20)


def test_shared_check_remains_dependency_light_and_mutation_free(tmp_path, monkeypatch):
    import train_pi05_frs.tools.train_shared_frs as pipeline

    calls = []
    monkeypatch.setattr(pipeline, "preflight_environment", lambda: None)
    original = pipeline.native_config

    def fake_native(config, source, *, check_paths):
        calls.append(source["name"])
        assert check_paths
        return {}, {"checkpoint": "/existing/frozen/checkpoint"}

    pipeline.native_config = fake_native
    try:
        pipeline.main(["--check"])
        assert calls == ["insert", "press", "bread", "pick"]
    finally:
        pipeline.native_config = original


def test_shared_pipeline_prepares_four_sources_then_trains_once(tmp_path, monkeypatch):
    import train_pi05_frs.shared_training as training
    import train_pi05_frs.tools.train_shared_frs as pipeline

    config = load_shared_config(DEFAULT_CONFIG)
    config["workspace"] = str(tmp_path / "work")
    stages, trained, arrow_paths = [], [], []
    monkeypatch.setenv("FRS_ARROW_TMPDIR", str(tmp_path / "scratch"))
    monkeypatch.setattr(pipeline, "load_shared_config", lambda _: config)
    monkeypatch.setattr(pipeline, "preflight_environment", lambda: None)
    monkeypatch.setattr(
        pipeline, "native_config", lambda cfg, src, **kw: ({"task": src["name"]}, {"name": src["name"]})
    )
    monkeypatch.setattr(training, "train_shared", lambda cfg, sources, **kw: trained.append(sources))

    class Process:
        def __init__(self, command, **kwargs):
            self.stdout = io.StringIO("synthetic stage output\n")
            stages.append((Path(command[4]).stem, command[2], command[5:]))
            arrow = Path(kwargs["env"]["HF_DATASETS_CACHE"])
            assert arrow.parent == tmp_path / "scratch"
            assert arrow.is_dir()
            (arrow / "generated.arrow").write_bytes(b"temporary dataset")
            arrow_paths.append(arrow)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return 0

    monkeypatch.setattr(pipeline.subprocess, "Popen", Process)
    pipeline.main([])
    assert [name for name, _, _ in stages] == [
        name for name in ("insert", "press", "bread", "pick") for _ in range(3)
    ]
    assert all(stages[index][2] == ["--checkpoint-smoke"] for index in (0, 3, 6, 9))
    assert len(trained) == 1 and [s["name"] for s in trained[0]] == ["insert", "press", "bread", "pick"]
    assert len(set(arrow_paths)) == 4
    for start in range(0, 12, 3):
        assert arrow_paths[start] == arrow_paths[start + 1] == arrow_paths[start + 2]
    assert all(not path.exists() for path in arrow_paths)
    log = next((tmp_path / "work/logs").glob("pipeline_*.log")).read_text()
    assert log.count("synthetic stage output") == 12


@pytest.mark.parametrize("failure_stage", [1, 2, 3])
def test_shared_pipeline_stops_before_training_after_failed_native_stage(
    tmp_path, monkeypatch, failure_stage
):
    import train_pi05_frs.shared_training as training
    import train_pi05_frs.tools.train_shared_frs as pipeline

    config = load_shared_config(DEFAULT_CONFIG)
    config["workspace"] = str(tmp_path / "work")
    count, arrow_paths = [], []
    monkeypatch.delenv("FRS_ARROW_TMPDIR", raising=False)
    persistent = tmp_path / "work/action_cache/previous_task/keep.npy"
    persistent.parent.mkdir(parents=True)
    persistent.write_bytes(b"completed action cache")
    monkeypatch.setattr(pipeline, "load_shared_config", lambda _: config)
    monkeypatch.setattr(pipeline, "preflight_environment", lambda: None)
    monkeypatch.setattr(
        pipeline, "native_config", lambda cfg, src, **kw: ({"task": src["name"]}, {"name": src["name"]})
    )
    monkeypatch.setattr(
        training,
        "train_shared",
        lambda *a, **kw: pytest.fail("FRS must not train after a failed cache stage"),
    )

    class Process:
        def __init__(self, command, **kwargs):
            self.stdout = io.StringIO("synthetic stage output\n")
            count.append(command)
            arrow = Path(kwargs["env"]["HF_DATASETS_CACHE"])
            (arrow / "partial.arrow.incomplete").write_bytes(b"partial dataset")
            arrow_paths.append(arrow)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return 1 if len(count) == failure_stage else 0

    monkeypatch.setattr(pipeline.subprocess, "Popen", Process)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.main([])
    assert len(count) == failure_stage
    assert all(not path.exists() for path in arrow_paths)
    assert persistent.read_bytes() == b"completed action cache"
    log = next((tmp_path / "work/logs").glob("pipeline_*.log")).read_text()
    assert "CalledProcessError" in log


def test_v21_subset_conversion_preserves_all_modalities_and_reindexes(tmp_path):
    raw = tmp_path / "raw"
    (raw / "meta").mkdir(parents=True)
    features = {
        key: {"dtype": "int64", "shape": [1], "names": None}
        for key in ("episode_index", "index", "frame_index", "task_index")
    }
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    for key, width in (("observation.state", 7), ("actions", 10)):
        features[key] = {"dtype": "float32", "shape": [width], "names": None}
    keys = [
        f"observation.images.{name}"
        for name in (
            "camera0",
            "camera1",
            "tactile_left_0",
            "tactile_right_0",
            "tactile_left_1",
            "tactile_right_1",
        )
    ]
    for key in keys:
        features[key] = {"dtype": "image", "shape": [4, 4, 3], "names": ["height", "width", "channel"]}
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test",
        "fps": 30,
        "total_episodes": 6,
        "total_frames": 18,
        "total_tasks": 1,
        "chunks_size": 1000,
        "total_chunks": 1,
        "total_videos": 0,
        "splits": {"train": "0:6"},
        "features": features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
    }
    (raw / "meta/info.json").write_text(json.dumps(info))
    episodes = [{"episode_index": i, "length": 3, "tasks": ["test"]} for i in range(6)]
    (raw / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
    (raw / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "test"}\n')
    stats = []
    for ep in range(6):
        values = {}
        for key, width, center in (
            ("observation.state", 7, 0.5),
            ("actions", 10, 0.5),
            ("episode_index", 1, ep),
            ("index", 1, ep * 3 + 1),
        ):
            values[key] = {
                "min": [center - 0.1] * width,
                "max": [center + 0.1] * width,
                "mean": [center] * width,
                "std": [0.1] * width,
                "count": [3],
            }
        stats.append({"episode_index": ep, "stats": values})
    (raw / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(row) + "\n" for row in stats))
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(buffer, format="PNG")
    selected = []
    for ep in (2, 5):
        table = pa.table(
            {
                "episode_index": [ep] * 3,
                "index": [ep * 3 + i for i in range(3)],
                "frame_index": [0, 1, 2],
                "task_index": [0] * 3,
                "timestamp": np.array([0, 1 / 30, 2 / 30], np.float32),
                "observation.state": [[float(ep)] * 7] * 3,
                "actions": [[float(ep)] * 10] * 3,
                **{key: [{"bytes": buffer.getvalue(), "path": None}] * 3 for key in keys},
            }
        )
        relative = info["data_path"].format(episode_chunk=0, episode_index=ep)
        path = raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        selected.append({"episode_index": ep, "path": relative, "length": 3})
    entry = tmp_path / "entry.json"
    entry.write_text(json.dumps({"episodes": selected}))
    destination = tmp_path / "normalized"
    script = """
import json, sys
from pathlib import Path
from train_pi05_frs.tools.prepare_shared_data import normalize_v21_subset
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset
normalize_v21_subset(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), Path(sys.argv[3]))
convert_dataset("org/subset", root=Path(sys.argv[3]), push_to_hub=False, force_conversion=True)
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONSAFEPATH": "1",
        "HF_HOME": str(tmp_path / "huggingface"),
        "HF_DATASETS_CACHE": str(tmp_path / "arrow_cache"),
    }
    result = subprocess.run(
        [sys.executable, "-c", script, str(entry), str(raw), str(destination)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = json.loads((destination / "meta/info.json").read_text())
    assert result["codebase_version"] == "v3.0"
    assert (result["total_episodes"], result["total_frames"]) == (2, 6)
    assert set(keys) <= set(result["features"])
    assert "action" in result["features"]
    table = pa.concat_tables(
        [pq.read_table(path) for path in sorted((destination / "data").rglob("*.parquet"))]
    )
    assert table["index"].to_pylist() == list(range(6))
    assert table["episode_index"].to_pylist() == [0, 0, 0, 1, 1, 1]
    assert [row[0] for row in table["action"].to_pylist()] == [2, 2, 2, 5, 5, 5]
    assert all(table[key][0].as_py()["bytes"] for key in keys)


@pytest.mark.parametrize(
    "script",
    [
        "scripts/download_frs_shared_data.sh",
        "scripts/download_frs_shared_ckpts.sh",
        "train_pi05_frs/scripts/start_frs_pi05_shared_train.sh",
    ],
)
def test_shared_shell_syntax_and_help(script):
    subprocess.run(["bash", "-n", str(ROOT / script)], check=True)
    result = subprocess.run(
        ["bash", str(ROOT / script), "--help"], capture_output=True, text=True, check=True
    )
    assert "--config" in result.stdout
