import json
import os
import signal
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
DATASETS = [f"pick_tube_{index:02d}" for index in range(1, 7)]
REPO_IDS = [f"KaiyueChen/{name}" for name in DATASETS]
TACTILE_KEYS = [
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
]


def _feature(dtype: str, shape: list[int]) -> dict[str, object]:
    return {"dtype": dtype, "shape": shape, "names": None, "fps": 30}


def _info(version: str = "v3.0") -> dict[str, object]:
    features = {
        "observation.images.camera0": _feature("image", [224, 224, 3]),
        "observation.images.camera1": _feature("image", [224, 224, 3]),
        "observation.state": _feature("float32", [20]),
        "actions": _feature("float32", [20]),
        "timestamp": _feature("float32", [1]),
        "frame_index": _feature("int64", [1]),
        "episode_index": _feature("int64", [1]),
        "index": _feature("int64", [1]),
        "task_index": _feature("int64", [1]),
    }
    features.update({key: _feature("image", [224, 224, 3]) for key in TACTILE_KEYS})
    return {
        "codebase_version": version,
        "fps": 30,
        "features": features,
        "total_episodes": 1,
        "total_frames": 1,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "robot_type": "bimanual",
        "splits": {"train": "0:1"},
    }


def _stats() -> dict[str, object]:
    def vector_stats() -> dict[str, object]:
        return {
            "min": [0.0] * 20,
            "max": [1.0] * 20,
            "mean": [0.5] * 20,
            "std": [0.1] * 20,
            "count": [1],
        }

    return {
        "observation.state": vector_stats(),
        "actions": vector_stats(),
    }


def _scalar_stats(values: list[int], *, quantiles: bool = True) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    stats: dict[str, object] = {
        "min": [float(array.min())],
        "max": [float(array.max())],
        "mean": [float(array.mean())],
        "std": [float(array.std())],
        "count": [len(values)],
    }
    if quantiles:
        for key, quantile in (("q01", 0.01), ("q50", 0.5), ("q99", 0.99)):
            stats[key] = [float(np.quantile(array, quantile))]
    return stats


def _episode_stats_row(
    episode_index: int,
    indexes: list[int],
    frame_indexes: list[int],
    *,
    stale_indexes: bool = False,
) -> dict[str, object]:
    row_stats = _stats()
    if stale_indexes:
        stale = _scalar_stats([99] * len(indexes))
        row_stats.update(
            {
                "index": stale,
                "episode_index": stale,
                "frame_index": stale,
            }
        )
    else:
        row_stats.update(
            {
                "index": _scalar_stats(indexes),
                "episode_index": _scalar_stats([episode_index] * len(indexes)),
                "frame_index": _scalar_stats(frame_indexes),
            }
        )
    return {"episode_index": episode_index, "stats": row_stats}


def _write_dataset(root: Path, version: str = "v3.0") -> None:
    rows = 2 if version == "v2.1" else 1
    info = _info(version)
    info["total_frames"] = rows
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(info), encoding="utf-8"
    )
    (root / "payload.bin").write_bytes(b"source-snapshot")
    data_path = root / "data" / "chunk-000" / "episode_000000.parquet"
    data_path.parent.mkdir(parents=True)
    if version == "v2.1":
        indexes = [0, 0]
        frame_indexes = [5, 5]
    else:
        indexes = [0]
        frame_indexes = [0]
    global_stats = _stats()
    global_stats.update(
        {
            "index": _scalar_stats(indexes),
            "episode_index": _scalar_stats([0] * rows),
            "frame_index": _scalar_stats(frame_indexes),
        }
    )
    (root / "meta" / "stats.json").write_text(
        json.dumps(global_stats), encoding="utf-8"
    )
    if version == "v2.1":
        episode_stats = _episode_stats_row(
            0, indexes, frame_indexes, stale_indexes=True
        )
        (root / "meta" / "episodes_stats.jsonl").write_text(
            json.dumps(episode_stats) + "\n", encoding="utf-8"
        )
    pq.write_table(
        pa.table(
            {
                "index": pa.array(indexes, type=pa.int64()),
                "episode_index": pa.array([0] * rows, type=pa.int64()),
                "frame_index": pa.array(frame_indexes, type=pa.int64()),
            }
        ),
        data_path,
    )


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _read_events(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@dataclass
class Harness:
    project: Path
    storage: Path
    hf_home: Path
    source_cache: Path
    encoder_dir: Path
    event_log: Path
    loader_log: Path

    def run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.project / "scripts" / "download_data.sh"), *arguments],
            cwd=self.project,
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            check=False,
        )

    @property
    def events(self) -> list[dict[str, object]]:
        return _read_events(self.event_log)

    def source_root(self, name: str) -> Path:
        return (
            self.source_cache
            / f"datasets--KaiyueChen--{name}"
            / "snapshots"
            / "test-snapshot"
        )

    def final_root(self, name: str) -> Path:
        return self.storage / "lerobot_v30" / "KaiyueChen" / name

    def work_root(self, name: str) -> Path:
        return self.storage / "lerobot_v30_work" / "KaiyueChen" / name


def _make_harness(tmp_path: Path, *, source_version: str = "v3.0") -> Harness:
    project = tmp_path / "project"
    scripts = project / "scripts"
    configs = project / "configs"
    tools = project / "tools"
    fake_bin = tmp_path / "fake-bin"
    fake_package = tmp_path / "fake-package"
    storage = tmp_path / "persistent-storage"
    hf_home = tmp_path / "custom-huggingface-root"
    source_cache = hf_home / "datasets"
    encoder_dir = tmp_path / "yaml-selected-encoder"
    event_log = tmp_path / "events.jsonl"
    loader_log = tmp_path / "loader.jsonl"
    venv = tmp_path / "named-frs-environment"

    scripts.mkdir(parents=True)
    configs.mkdir()
    tools.mkdir()
    fake_bin.mkdir()
    storage.mkdir()
    (fake_package / "lerobot" / "datasets").mkdir(parents=True)
    (venv / "bin").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "download_data.sh", scripts / "download_data.sh")
    _write_executable(
        venv / "bin" / "python",
        f"#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} \"$@\"\n",
    )
    repair_helper = ROOT / "scripts" / "repair_v21_indexes.py"
    if repair_helper.is_file():
        shutil.copy2(repair_helper, scripts / repair_helper.name)

    for config_name in (
        "train_vtsmolvla_jax_tactile16.yaml",
        "train_vtsmolvla_jax_tactile32.yaml",
    ):
        (configs / config_name).write_text(
            f"model:\n  tactile_encoder_path: {encoder_dir}\n", encoding="utf-8"
        )

    for package_file in (
        fake_package / "lerobot" / "__init__.py",
        fake_package / "lerobot" / "datasets" / "__init__.py",
    ):
        package_file.write_text("", encoding="utf-8")
    (fake_package / "lerobot" / "datasets" / "lerobot_dataset.py").write_text(
        """
import json
import os
from pathlib import Path


def _log(kind, **values):
    path = Path(os.environ["FAKE_LOADER_LOG"])
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"kind": kind, **values}) + "\\n")


class LeRobotDatasetMetadata:
    def __init__(self, repo_id, root=None, **kwargs):
        self.repo_id = repo_id
        self.root = Path(root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.features = self.info["features"]
        self.stats = json.loads((self.root / "meta" / "stats.json").read_text())
        self.total_episodes = self.info["total_episodes"]
        _log("metadata", repo_id=repo_id, root=str(self.root))


class LeRobotDataset:
    def __init__(self, repo_id, root=None, episodes=None, **kwargs):
        self.repo_id = repo_id
        self.root = Path(root)
        self.episodes = episodes
        _log("dataset", repo_id=repo_id, root=str(self.root), episodes=episodes)

    def __getitem__(self, index):
        _log("sample", repo_id=self.repo_id, index=index)
        fail_once = self.root / ".fail_sample_once"
        if fail_once.exists():
            fail_once.unlink()
            raise RuntimeError("forced one-time sample decode failure")
        if (self.root / ".fail_sample").exists():
            raise RuntimeError("forced sample decode failure")
        return {
            "index": index,
            "frame_index": 0,
            "episode_index": 0,
            "observation.state": [0.0] * 20,
            "actions": [0.0] * 20,
        }
""".lstrip(),
        encoding="utf-8",
    )

    _write_executable(
        fake_bin / "uv",
        """#!/usr/bin/env python3
import json
import os
import shutil
import sys
from pathlib import Path


args = sys.argv[1:]
event = {
    "kind": "uv",
    "args": args,
    "HF_HOME": os.environ.get("HF_HOME"),
    "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
    "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
    "HF_LEROBOT_HOME": os.environ.get("HF_LEROBOT_HOME"),
}
with Path(os.environ["FAKE_EVENT_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(event) + "\\n")

if "download" in args:
    repo_id = args[args.index("download") + 1]
    cache_root = Path(args[args.index("--cache-dir") + 1])
    ready_path = os.environ.get("FAKE_DOWNLOAD_SIGNAL_READY")
    if ready_path:
        Path(ready_path).write_text(repo_id, encoding="utf-8")
        import time
        time.sleep(60)
        Path(os.environ["FAKE_DOWNLOAD_SIGNAL_SENTINEL"]).write_text(
            "continued-after-signal", encoding="utf-8"
        )
    print(cache_root / f"datasets--{repo_id.replace('/', '--')}" / "snapshots" / "test-snapshot")
    raise SystemExit(0)

if "lerobot.datasets.v30.convert_dataset_v21_to_v30" in args:
    root_arg = next(value for value in args if value.startswith("--root="))
    root = Path(root_arg.split("=", 1)[1])
    repo_arg = next(value for value in args if value.startswith("--repo-id="))
    repo_id = repo_arg.split("=", 1)[1]
    old = root.with_name(root.name + "_old")
    converted = root.with_name(root.name + "_v30")
    if os.environ.get("FAKE_CONVERT_FAIL_REPO") == repo_id:
        old.mkdir(parents=True, exist_ok=True)
        (old / "partial-old").write_text("partial", encoding="utf-8")
        shutil.copytree(root, converted)
        (converted / "partial-v30").write_text("partial", encoding="utf-8")
        raise SystemExit(23)
    shutil.copytree(root, converted)
    info_path = converted / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["codebase_version"] = "v3.0"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    episode_stats_path = converted / "meta" / "episodes_stats.jsonl"
    if episode_stats_path.is_file():
        rows = [json.loads(line) for line in episode_stats_path.read_text(encoding="utf-8").splitlines()]
        global_stats_path = converted / "meta" / "stats.json"
        global_stats = json.loads(global_stats_path.read_text(encoding="utf-8"))
        for key in ("index", "episode_index", "frame_index"):
            feature_stats = [row["stats"][key] for row in rows]
            counts = [float(stats["count"][0]) for stats in feature_stats]
            total_count = sum(counts)
            mean = sum(stats["mean"][0] * count for stats, count in zip(feature_stats, counts)) / total_count
            variance = sum(
                count * (stats["std"][0] ** 2 + (stats["mean"][0] - mean) ** 2)
                for stats, count in zip(feature_stats, counts)
            ) / total_count
            aggregated = {
                "min": [min(stats["min"][0] for stats in feature_stats)],
                "max": [max(stats["max"][0] for stats in feature_stats)],
                "mean": [mean],
                "std": [variance ** 0.5],
                "count": [int(total_count)],
            }
            quantile_keys = [
                name for name in feature_stats[0]
                if name.startswith("q") and name[1:].isdigit()
                and all(name in stats for stats in feature_stats)
            ]
            for quantile_key in quantile_keys:
                aggregated[quantile_key] = [
                    sum(
                        stats[quantile_key][0] * count
                        for stats, count in zip(feature_stats, counts)
                    ) / total_count
                ]
            global_stats[key] = aggregated
        global_stats_path.write_text(json.dumps(global_stats), encoding="utf-8")
    root.rename(old)
    converted.rename(root)

raise SystemExit(0)
""",
    )

    _write_executable(
        scripts / "download_ckpt.sh",
        """#!/usr/bin/env bash
"${FRS_VENV_DIR}/bin/python" - "$@" <<'PY'
import json
import os
import sys
from pathlib import Path


with Path(os.environ["FAKE_EVENT_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"kind": "encoder", "call": "scripts/download_ckpt.sh", "args": sys.argv[1:]}) + "\\n")
if os.environ.get("FAKE_ENCODER_FAIL") == "1":
    raise SystemExit(29)
output = Path(os.environ["FAKE_ENCODER_DIR"])
output.mkdir(parents=True, exist_ok=True)
(output / "checkpoint.json").write_text(json.dumps({"params_file": "params.npz"}), encoding="utf-8")
if os.environ.get("FAKE_ENCODER_OMIT_PARAMS") != "1":
    (output / "params.npz").write_bytes(b"minimal-encoder")
PY
""",
    )
    _write_executable(
        tools / "precompute_tactile_embeddings.py",
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
with Path(os.environ["FAKE_EVENT_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"kind": "precompute"}) + "\\n")
raise SystemExit(97)
""",
    )

    env_values = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FRS_STORAGE_ROOT": str(storage),
        "FRS_VENV_DIR": str(venv),
        "UV_PROJECT_ENVIRONMENT": str(venv),
        "UV_CACHE_DIR": str(tmp_path / "custom-uv-cache"),
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(tmp_path / "custom-hub-cache"),
        "HF_DATASETS_CACHE": str(tmp_path / "custom-arrow-cache"),
        "HF_LEROBOT_HOME": str(tmp_path / "custom-lerobot-home"),
        "TMPDIR": str(tmp_path / "custom-tmp"),
        "PYTHONPATH": str(fake_package),
        "FAKE_EVENT_LOG": str(event_log),
        "FAKE_LOADER_LOG": str(loader_log),
        "FAKE_ENCODER_DIR": str(encoder_dir),
    }
    (project / ".env.frs").write_text(
        "".join(
            f"export {key}={shlex.quote(value)}\n" for key, value in env_values.items()
        ),
        encoding="utf-8",
    )

    harness = Harness(
        project=project,
        storage=storage,
        hf_home=hf_home,
        source_cache=source_cache,
        encoder_dir=encoder_dir,
        event_log=event_log,
        loader_log=loader_log,
    )
    for name in DATASETS:
        _write_dataset(harness.source_root(name), source_version)
    return harness


def _uv_download_repos(events: list[dict[str, object]]) -> list[str]:
    repos = []
    for event in events:
        args = event.get("args", [])
        if event.get("kind") == "uv" and "download" in args:
            repos.append(args[args.index("download") + 1])
    return repos


def _converter_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        event
        for event in events
        if event.get("kind") == "uv"
        and "lerobot.datasets.v30.convert_dataset_v21_to_v30"
        in event.get("args", [])
    ]


def _read_index_columns(root: Path) -> tuple[list[int], list[int], list[int]]:
    indexes: list[int] = []
    episodes: list[int] = []
    frames: list[int] = []
    for path in sorted((root / "data").glob("*/*.parquet")):
        table = pq.read_table(path, columns=["index", "episode_index", "frame_index"])
        indexes.extend(table["index"].to_pylist())
        episodes.extend(table["episode_index"].to_pylist())
        frames.extend(table["frame_index"].to_pylist())
    return indexes, episodes, frames


def _assert_scalar_stats(actual: dict[str, object], values: list[int]) -> None:
    expected = _scalar_stats(values)
    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)


def test_repair_helper_updates_multiple_episode_index_stats_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v21"
    info = _info("v2.1")
    info["total_episodes"] = 2
    info["total_frames"] = 5
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    episode_rows = []
    for episode, length in enumerate((3, 2)):
        path = root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "index": pa.array([52_802] * length, type=pa.int64()),
                    "episode_index": pa.array([8] * length, type=pa.int64()),
                    "frame_index": pa.array([61_810] * length, type=pa.int64()),
                }
            ),
            path,
        )
        episode_rows.append(
            _episode_stats_row(
                episode,
                [52_802] * length,
                [61_810] * length,
                stale_indexes=True,
            )
        )
    non_index_stats = [row["stats"]["actions"] for row in episode_rows]
    (root / "meta" / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episode_rows), encoding="utf-8"
    )

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "repair_v21_indexes.py"), str(root)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    repaired_rows = [
        json.loads(line)
        for line in (root / "meta" / "episodes_stats.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    expected_values = (
        ([0, 1, 2], [0, 0, 0], [0, 1, 2]),
        ([3, 4], [1, 1], [0, 1]),
    )
    for episode, (row, (indexes, episodes, frames)) in enumerate(
        zip(repaired_rows, expected_values, strict=True)
    ):
        assert row["episode_index"] == episode
        _assert_scalar_stats(row["stats"]["index"], indexes)
        _assert_scalar_stats(row["stats"]["episode_index"], episodes)
        _assert_scalar_stats(row["stats"]["frame_index"], frames)
        assert row["stats"]["actions"] == non_index_stats[episode]


def test_repair_helper_fixes_61811_rows_with_52802_maximum_index(tmp_path: Path) -> None:
    root = tmp_path / "v21"
    first_length = 52_803
    second_length = 61_811 - first_length
    info = _info("v2.1")
    info["total_episodes"] = 2
    info["total_frames"] = 61_811
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    episode_stats_rows = []
    for episode, length in enumerate((first_length, second_length)):
        path = root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        start = 0 if episode == 0 else first_length
        pq.write_table(
            pa.table(
                {
                    "index": pa.array(
                        [(start + offset) % 52_803 for offset in range(length)],
                        type=pa.int64(),
                    ),
                    "episode_index": pa.array([episode] * length, type=pa.int64()),
                    "frame_index": pa.array(
                        [start + offset for offset in range(length)], type=pa.int64()
                    ),
                }
            ),
            path,
        )
        episode_stats_rows.append(
            _episode_stats_row(
                episode,
                [52_802] * length,
                [61_810] * length,
                stale_indexes=True,
            )
        )
    (root / "meta" / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episode_stats_rows),
        encoding="utf-8",
    )
    before, _, _ = _read_index_columns(root)
    assert len(before) == 61_811
    assert max(before) == 52_802

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "repair_v21_indexes.py"), str(root)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    indexes, episodes, frames = _read_index_columns(root)
    assert indexes == list(range(61_811))
    assert episodes == [0] * first_length + [1] * second_length
    assert frames == list(range(first_length)) + list(range(second_length))


def test_requires_task1_environment_file(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path)
    (harness.project / ".env.frs").unlink()

    result = harness.run()

    assert result.returncode != 0
    assert ".env.frs" in result.stderr
    assert harness.events == []


def test_requires_named_project_environment(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path)
    (tmp_path / "named-frs-environment" / "bin" / "python").unlink()

    result = harness.run()

    assert result.returncode != 0
    assert "FRS_VENV_DIR" in result.stderr
    assert harness.events == []


def test_orchestrates_fixed_datasets_loader_validation_and_minimal_encoder(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path, source_version="v3.0")

    result = harness.run()

    assert result.returncode == 0, result.stderr
    assert _uv_download_repos(harness.events) == REPO_IDS
    assert [
        [event["call"], *event["args"]]
        for event in harness.events
        if event["kind"] == "encoder"
    ] == [["scripts/download_ckpt.sh"]]
    assert not [event for event in harness.events if event["kind"] == "precompute"]
    loader_events = _read_events(harness.loader_log)
    assert [event["repo_id"] for event in loader_events if event["kind"] == "metadata"] == REPO_IDS
    assert [event["repo_id"] for event in loader_events if event["kind"] == "dataset"] == REPO_IDS
    assert [event["index"] for event in loader_events if event["kind"] == "sample"] == [
        0
    ] * len(DATASETS)
    for event in [event for event in harness.events if event["kind"] == "uv"]:
        assert event["HF_HOME"] == str(harness.hf_home)
        assert event["HF_HUB_CACHE"] == str(tmp_path / "custom-hub-cache")
        assert event["HF_DATASETS_CACHE"] == str(tmp_path / "custom-arrow-cache")
        assert event["HF_LEROBOT_HOME"] == str(tmp_path / "custom-lerobot-home")


def test_uses_printed_snapshot_and_dereferences_relative_hf_symlinks_before_cleanup(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path, source_version="v3.0")
    source = harness.source_root(DATASETS[0])
    repo_cache = source.parents[1]
    blob = repo_cache / "blobs" / "payload-blob"
    blob.parent.mkdir(parents=True, exist_ok=True)
    source_payload = source / "payload.bin"
    source_payload.replace(blob)
    source_payload.symlink_to("../../blobs/payload-blob")
    decoy = repo_cache / "snapshots" / "newer-decoy"
    _write_dataset(decoy, "broken")
    os.utime(decoy, (4_000_000_000, 4_000_000_000))

    result = harness.run("--cleanup-source")

    assert result.returncode == 0, result.stderr
    final_payload = harness.final_root(DATASETS[0]) / "payload.bin"
    assert final_payload.is_file()
    assert not final_payload.is_symlink()
    assert final_payload.read_bytes() == b"source-snapshot"
    assert not repo_cache.exists()


def test_rejects_symlinked_derived_ancestor_without_touching_escape_target(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path)
    outside = tmp_path / "outside-work"
    outside.mkdir()
    sentinel = outside / "keep"
    sentinel.write_text("untouched", encoding="utf-8")
    work_parent = harness.storage / "lerobot_v30_work"
    work_parent.parent.mkdir(parents=True, exist_ok=True)
    work_parent.symlink_to(outside, target_is_directory=True)

    result = harness.run()

    assert result.returncode != 0
    assert "symlink" in result.stderr.lower() or "符号链接" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert not (outside / "KaiyueChen").exists()
    assert harness.events == []


def test_lock_creation_rejects_preexisting_symlink_without_following_it(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path)
    lock_root = harness.storage / ".locks"
    lock_root.mkdir(parents=True)
    outside = tmp_path / "outside-lock"
    outside.write_text("do-not-truncate", encoding="utf-8")
    (lock_root / "frs-download-data.lock").symlink_to(outside)

    result = harness.run()

    assert result.returncode != 0
    assert "下载" in result.stderr and "运行" in result.stderr
    assert outside.read_text(encoding="utf-8") == "do-not-truncate"
    assert harness.events == []


def test_existing_valid_v3_destinations_skip_download_and_rerun_idempotently(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path, source_version="v2.1")
    for name in DATASETS:
        _write_dataset(harness.final_root(name), "v3.0")

    first = harness.run()
    second = harness.run()

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert _uv_download_repos(harness.events) == []
    assert _converter_events(harness.events) == []
    assert len([event for event in harness.events if event["kind"] == "encoder"]) == 2
    assert len(
        [event for event in _read_events(harness.loader_log) if event["kind"] == "sample"]
    ) == 2 * len(DATASETS)


def test_existing_v3_with_invalid_schema_is_rebuilt_from_source(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path, source_version="v3.0")
    for name in DATASETS:
        _write_dataset(harness.final_root(name), "v3.0")
    broken_info_path = harness.final_root(DATASETS[0]) / "meta" / "info.json"
    broken_info = json.loads(broken_info_path.read_text(encoding="utf-8"))
    broken_info["features"]["observation.state"]["shape"] = [19]
    broken_info_path.write_text(json.dumps(broken_info), encoding="utf-8")

    result = harness.run()

    assert result.returncode == 0, result.stderr
    assert _uv_download_repos(harness.events) == [REPO_IDS[0]]
    repaired = json.loads(broken_info_path.read_text(encoding="utf-8"))
    assert repaired["features"]["observation.state"]["shape"] == [20]


def test_bad_candidate_does_not_replace_a_structurally_valid_existing_final(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path, source_version="v3.0")
    for name in DATASETS:
        _write_dataset(harness.final_root(name), "v3.0")
    protected_final = harness.final_root(DATASETS[0])
    sentinel = protected_final / "valid-final-sentinel"
    sentinel.write_text("preserve-me", encoding="utf-8")
    (protected_final / ".fail_sample_once").touch()
    candidate_info_path = harness.source_root(DATASETS[0]) / "meta" / "info.json"
    candidate_info = json.loads(candidate_info_path.read_text(encoding="utf-8"))
    candidate_info["features"]["actions"]["shape"] = [19]
    candidate_info_path.write_text(json.dumps(candidate_info), encoding="utf-8")

    result = harness.run()

    assert result.returncode != 0
    assert "actions" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve-me"
    final_info = json.loads(
        (protected_final / "meta" / "info.json").read_text(encoding="utf-8")
    )
    assert final_info["features"]["actions"]["shape"] == [20]


def test_v3_index_stats_must_match_projected_parquet_rows(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path, source_version="v3.0")
    stats_path = harness.source_root(DATASETS[0]) / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["index"]["max"] = [99.0]
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    result = harness.run()

    assert result.returncode != 0
    assert "index.max" in result.stderr
    assert not harness.final_root(DATASETS[0]).exists()
    assert not [event for event in harness.events if event["kind"] == "encoder"]


def test_v21_sources_are_copied_converted_and_preserved_by_default(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path, source_version="v2.1")

    result = harness.run()

    assert result.returncode == 0, result.stderr
    converters = _converter_events(harness.events)
    assert len(converters) == 6
    for name, event in zip(DATASETS, converters, strict=True):
        args = event["args"]
        assert f"--repo-id=KaiyueChen/{name}" in args
        assert f"--root={harness.work_root(name)}" in args
        assert harness.source_root(name).exists()
        source_episode_stats = json.loads(
            (harness.source_root(name) / "meta" / "episodes_stats.jsonl")
            .read_text(encoding="utf-8")
            .strip()
        )
        assert source_episode_stats["stats"]["index"]["min"] == [99.0]
        assert harness.work_root(name).with_name(name + "_old").exists()
        info = json.loads(
            (harness.final_root(name) / "meta" / "info.json").read_text(encoding="utf-8")
        )
        assert info["codebase_version"] == "v3.0"
        indexes, episodes, frames = _read_index_columns(harness.final_root(name))
        assert indexes == [0, 1]
        assert episodes == [0, 0]
        assert frames == [0, 1]
        final_stats = json.loads(
            (harness.final_root(name) / "meta" / "stats.json").read_text(
                encoding="utf-8"
            )
        )
        _assert_scalar_stats(final_stats["index"], indexes)
        _assert_scalar_stats(final_stats["episode_index"], episodes)
        _assert_scalar_stats(final_stats["frame_index"], frames)


def test_failed_conversion_preserves_source_even_when_cleanup_was_requested(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path, source_version="v2.1")
    with (harness.project / ".env.frs").open("a", encoding="utf-8") as stream:
        stream.write("export FAKE_CONVERT_FAIL_REPO=KaiyueChen/pick_tube_01\n")

    result = harness.run("--cleanup-source")

    assert result.returncode != 0
    assert len(_converter_events(harness.events)) == 1
    assert "转换失败" in result.stderr
    assert harness.source_root("pick_tube_01").exists()
    work_root = harness.work_root("pick_tube_01")
    assert work_root.exists()
    assert work_root.with_name(work_root.name + "_old").joinpath("partial-old").is_file()
    assert work_root.with_name(work_root.name + "_v30").joinpath("partial-v30").is_file()
    assert not [event for event in harness.events if event["kind"] == "encoder"]

    with (harness.project / ".env.frs").open("a", encoding="utf-8") as stream:
        stream.write("unset FAKE_CONVERT_FAIL_REPO\n")
    recovered = harness.run()

    assert recovered.returncode == 0, recovered.stderr
    assert harness.source_root("pick_tube_01").exists()
    assert not work_root.with_name(work_root.name + "_v30").exists()
    assert not work_root.with_name(work_root.name + "_old").joinpath("partial-old").exists()


def test_schema_validation_failure_blocks_encoder_and_preserves_source(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path, source_version="v3.0")
    info_path = harness.source_root(DATASETS[0]) / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["actions"]["shape"] = [19]
    info_path.write_text(json.dumps(info), encoding="utf-8")

    result = harness.run("--cleanup-source")

    assert result.returncode != 0
    assert "actions" in result.stderr
    assert harness.source_root(DATASETS[0]).exists()
    assert not [event for event in harness.events if event["kind"] == "encoder"]


def test_sample_decode_failure_blocks_encoder(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path, source_version="v3.0")
    (harness.source_root(DATASETS[0]) / ".fail_sample").touch()

    result = harness.run()

    assert result.returncode != 0
    assert "forced sample decode failure" in result.stderr
    assert not [event for event in harness.events if event["kind"] == "encoder"]


def test_encoder_failure_is_propagated_after_dataset_validation(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path, source_version="v3.0")
    with (harness.project / ".env.frs").open("a", encoding="utf-8") as stream:
        stream.write("export FAKE_ENCODER_FAIL=1\n")

    result = harness.run()

    assert result.returncode == 29
    assert all(harness.final_root(name).exists() for name in DATASETS)


def test_encoder_metadata_must_resolve_to_an_existing_params_archive(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path, source_version="v3.0")
    with (harness.project / ".env.frs").open("a", encoding="utf-8") as stream:
        stream.write("export FAKE_ENCODER_OMIT_PARAMS=1\n")

    result = harness.run()

    assert result.returncode != 0
    assert "params.npz" in result.stderr


def test_second_process_is_rejected_by_scoped_download_lock(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path)
    lock_path = harness.storage / ".locks" / "frs-download-data.lock"
    lock_path.mkdir(parents=True)

    result = harness.run()

    assert result.returncode != 0
    assert "下载" in result.stderr and "运行" in result.stderr
    assert harness.events == []


@pytest.mark.parametrize(
    ("sent_signal", "expected_returncode"),
    ((signal.SIGINT, 130), (signal.SIGTERM, 143)),
)
def test_signal_exits_with_shell_status_releases_lock_and_stops_execution(
    tmp_path: Path,
    sent_signal: signal.Signals,
    expected_returncode: int,
) -> None:
    harness = _make_harness(tmp_path)
    ready = tmp_path / f"signal-{sent_signal.name}-ready"
    sentinel = tmp_path / f"signal-{sent_signal.name}-post-signal"
    with (harness.project / ".env.frs").open("a", encoding="utf-8") as stream:
        stream.write(
            f"export FAKE_DOWNLOAD_SIGNAL_READY={shlex.quote(str(ready))}\n"
        )
        stream.write(
            "export FAKE_DOWNLOAD_SIGNAL_SENTINEL="
            f"{shlex.quote(str(sentinel))}\n"
        )
    process = subprocess.Popen(
        ["bash", str(harness.project / "scripts" / "download_data.sh")],
        cwd=harness.project,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.is_file(), "fake download did not reach the signal rendezvous"
        lock_path = harness.storage / ".locks" / "frs-download-data.lock"
        assert lock_path.is_dir()
        os.killpg(process.pid, sent_signal)
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)

    assert process.returncode == expected_returncode, (stdout, stderr)
    assert not lock_path.exists()
    assert not sentinel.exists()
    assert not [event for event in harness.events if event["kind"] == "encoder"]


def test_help_and_unknown_option_contract(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path)

    help_result = harness.run("--help")
    bad_result = harness.run("--unexpected")

    assert help_result.returncode == 0
    assert "--cleanup-source" in help_result.stdout
    assert bad_result.returncode != 0
    assert "用法" in bad_result.stderr
    assert harness.events == []


def test_cleanup_source_removes_only_known_sources_and_conversion_leftovers(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path, source_version="v2.1")
    unrelated_source = harness.source_cache / "datasets--someone--unrelated"
    unrelated_source.mkdir(parents=True)
    (unrelated_source / "keep").touch()
    unrelated_work = harness.storage / "lerobot_v30_work" / "someone" / "keep"
    unrelated_work.mkdir(parents=True)

    result = harness.run("--cleanup-source")

    assert result.returncode == 0, result.stderr
    for name in DATASETS:
        assert not harness.source_root(name).parents[1].exists()
        assert not harness.work_root(name).exists()
        assert not harness.work_root(name).with_name(name + "_old").exists()
        assert not harness.work_root(name).with_name(name + "_v30").exists()
        assert harness.final_root(name).exists()
    assert unrelated_source.exists()
    assert unrelated_work.exists()
    assert harness.encoder_dir.joinpath("checkpoint.json").is_file()
    assert harness.encoder_dir.joinpath("params.npz").is_file()
