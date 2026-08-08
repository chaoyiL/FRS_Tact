import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = [f"pick_tube_{index:02d}" for index in range(1, 5)]
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


def _write_dataset(root: Path, version: str = "v3.0") -> None:
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(_info(version)), encoding="utf-8"
    )
    (root / "meta" / "stats.json").write_text(
        json.dumps(_stats()), encoding="utf-8"
    )
    (root / "payload.bin").write_bytes(b"source-snapshot")


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
    (fake_package / "lerobot" / "datasets").mkdir(parents=True)
    (venv / "bin").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "download_data.sh", scripts / "download_data.sh")
    (venv / "bin" / "python").symlink_to(sys.executable)

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

if "lerobot.datasets.v30.convert_dataset_v21_to_v30" in args:
    root_arg = next(value for value in args if value.startswith("--root="))
    root = Path(root_arg.split("=", 1)[1])
    repo_arg = next(value for value in args if value.startswith("--repo-id="))
    repo_id = repo_arg.split("=", 1)[1]
    if os.environ.get("FAKE_CONVERT_FAIL_REPO") == repo_id:
        raise SystemExit(23)
    old = root.with_name(root.name + "_old")
    old.mkdir(parents=True, exist_ok=True)
    (old / "conversion-leftover").write_text("keep unless cleanup", encoding="utf-8")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["codebase_version"] = "v3.0"
    info_path.write_text(json.dumps(info), encoding="utf-8")

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
    assert [event["index"] for event in loader_events if event["kind"] == "sample"] == [0] * 4
    for event in [event for event in harness.events if event["kind"] == "uv"]:
        assert event["HF_HOME"] == str(harness.hf_home)
        assert event["HF_HUB_CACHE"] == str(tmp_path / "custom-hub-cache")
        assert event["HF_DATASETS_CACHE"] == str(tmp_path / "custom-arrow-cache")
        assert event["HF_LEROBOT_HOME"] == str(tmp_path / "custom-lerobot-home")


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
    assert len([event for event in _read_events(harness.loader_log) if event["kind"] == "sample"]) == 8


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


def test_v21_sources_are_copied_converted_and_preserved_by_default(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path, source_version="v2.1")

    result = harness.run()

    assert result.returncode == 0, result.stderr
    converters = _converter_events(harness.events)
    assert len(converters) == 4
    for name, event in zip(DATASETS, converters, strict=True):
        args = event["args"]
        assert f"--repo-id=KaiyueChen/{name}" in args
        assert f"--root={harness.work_root(name)}" in args
        assert harness.source_root(name).exists()
        assert harness.work_root(name).with_name(name + "_old").exists()
        info = json.loads(
            (harness.final_root(name) / "meta" / "info.json").read_text(encoding="utf-8")
        )
        assert info["codebase_version"] == "v3.0"


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
    assert not [event for event in harness.events if event["kind"] == "encoder"]


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
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("w", encoding="utf-8") as held_lock:
        fcntl.flock(held_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = harness.run()

    assert result.returncode != 0
    assert "下载" in result.stderr and "运行" in result.stderr
    assert harness.events == []


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
