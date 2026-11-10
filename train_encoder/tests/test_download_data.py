from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from train_encoder.download_data import download_task, episode_path, required_image_keys
from train_encoder.utils.data import batches, build_future_records, resolve_data_keys
from train_encoder.utils.image_dataset import create_image_dataset


@pytest.fixture
def hub(tmp_path, monkeypatch):
    source = tmp_path / "hub"
    names = ["pick_01", "insert_01", "bread_03", "two_tubes_01"]
    tasks, calls = [], []
    for task_i, name in enumerate(names):
        repo = f"KaiyueChen/{name}"
        root = source / repo
        (root / "meta").mkdir(parents=True)
        keys = required_image_keys(["left", "right"])
        info = {
            "codebase_version": "v2.1",
            "chunks_size": 1000,
            "total_episodes": 8,
            "features": {key: {"dtype": "image"} for key in keys},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        }
        (root / "meta/info.json").write_text(json.dumps(info))
        episodes = [{"episode_index": i, "length": 3} for i in range(8)]
        (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes))
        (root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "test"}\n')
        for episode in range(8):
            buffer = io.BytesIO()
            Image.fromarray(np.full((4, 4, 3), episode * 20, dtype=np.uint8)).save(buffer, format="PNG")
            data = {
                "episode_index": [episode] * 3,
                "timestamp": [0.0, 0.1, 0.2],
                "index": list(range(episode * 3, episode * 3 + 3)),
                "state": [[0.0]] * 3,
            }
            data.update({key: [{"bytes": buffer.getvalue(), "path": None}] * 3 for key in keys})
            table = pa.table(data)
            # Full HF feature metadata, including columns excluded by the image-only reader.
            from datasets import Features, Image as HfImage

            features = Features.from_arrow_schema(table.schema)
            for key in keys:
                features[key] = HfImage()
            table = table.replace_schema_metadata(
                {b"huggingface": json.dumps({"info": {"features": features.to_dict()}}).encode()}
            )
            path = root / episode_path(info, episode)
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path)
        tasks.append(
            {
                "name": name,
                "dataset_repo_ids": [repo],
                "sides": ["right"] if task_i < 2 else ["left", "right"],
                "num_train_episodes": 3,
                "num_val_episodes": 2,
            }
        )

    class Api:
        def dataset_info(self, repo_id, **kwargs):
            root = source / repo_id
            return SimpleNamespace(
                sha="a" * 40,
                siblings=[
                    SimpleNamespace(rfilename=p.relative_to(root).as_posix(), size=p.stat().st_size)
                    for p in root.rglob("*")
                    if p.is_file()
                ],
            )

    def snapshot(repo_id, *, repo_type, revision, local_dir, allow_patterns, **kwargs):
        assert repo_type == "dataset"
        assert revision == "a" * 40
        calls.append((repo_id, list(allow_patterns)))
        for relative in allow_patterns:
            destination = Path(local_dir) / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / repo_id / relative, destination)
        return str(local_dir)

    def download_file(repo_id, filename, *, repo_type, revision, local_dir, **kwargs):
        snapshot(
            repo_id, repo_type=repo_type, revision=revision, local_dir=local_dir, allow_patterns=[filename]
        )
        return str(Path(local_dir) / filename)

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download_file)
    return tasks, calls


def prepare(tasks, output, *, dry_run=False):
    return [
        download_task(
            task, output_dir=output, seed=0, task_index=i, revision="main", max_workers=2, dry_run=dry_run
        )
        for i, task in enumerate(tasks)
    ]


def test_exact_download_partition_resume_and_local_loading(tmp_path, hub, monkeypatch):
    tasks, calls = hub
    output = tmp_path / "subset"
    entries = prepare(tasks, output)
    for entry in entries:
        assert len(entry["episodes"]) == 5
        assert len(entry["train_episode_ids"]) == 3
        assert len(entry["val_episode_ids"]) == 2
        assert not set(entry["train_episode_ids"]) & set(entry["val_episode_ids"])
    episode_calls = [paths for _, paths in calls if paths[0].startswith("data/")]
    assert len(episode_calls) == 20
    assert all(len(paths) == 1 for paths in episode_calls)
    before = len(calls)
    assert entries == prepare(tasks, output)
    assert len(calls) == before  # Fully downloaded data works offline.
    removed = output / entries[0]["root"] / entries[0]["episodes"][0]["path"]
    removed.unlink()
    prepare(tasks, output)
    assert len(calls[-1][1]) == 1
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"format": "tactile-encoder-v21-subset-v1", "tasks": entries}))
    monkeypatch.setattr(
        "train_encoder.utils.image_dataset._import_lerobot_dataset_module",
        lambda: pytest.fail("Local subsets must not invoke LeRobot/Hub downloads"),
    )
    dataset = create_image_dataset(
        [task["dataset_repo_ids"][0] for task in tasks], image_size=4, dataset_manifest=manifest
    ).dataset
    records = build_future_records(dataset, tasks=tasks, split_seed=0)
    assert len(records.train_episodes) == 12
    assert len(records.val_episodes) == 8
    for entry, selected in zip(entries, records.task_selections, strict=True):
        local = {e["episode_index"]: i for i, e in enumerate(entry["episodes"])}
        offset = next(iter(dataset.episodes_by_repo()[entry["repo_id"]]))
        assert selected["train_episodes"] == [offset + local[e] for e in entry["train_episode_ids"]]
    loaded = list(
        batches(
            dataset,
            records.split_records("train"),
            resolve_data_keys(),
            batch_size=8,
            image_size=4,
            shuffle=True,
            seed=2,
            num_workers=2,
            prefetch_batches=2,
        )
    )
    assert sum(len(b["side_id"]) for b in loaded) == 36
    assert all(np.all(b["side_id"][b["episode_index"] < 10] == 1) for b in loaded)
    loaded_mp = list(
        batches(
            dataset,
            records.split_records("train"),
            resolve_data_keys(),
            batch_size=8,
            image_size=4,
            shuffle=True,
            seed=2,
            num_workers=2,
            prefetch_batches=2,
            loader="mp",
            dataset_repo_ids=[task["dataset_repo_ids"][0] for task in tasks],
        )
    )
    assert len(loaded_mp) == len(loaded)
    for thread_batch, mp_batch in zip(loaded, loaded_mp, strict=True):
        for key in ("dataset_index", "episode_index", "side_id"):
            np.testing.assert_array_equal(thread_batch[key], mp_batch[key])
        np.testing.assert_allclose(
            thread_batch["current_rgb"], mp_batch["current_rgb"].astype(np.float32) / 255.0
        )
    with pytest.raises(ValueError, match="configuration differ"):
        build_future_records(dataset, tasks=tasks, split_seed=1)


def test_dry_run_fetches_only_metadata(tmp_path, hub):
    tasks, calls = hub
    prepare(tasks, tmp_path / "subset", dry_run=True)
    assert len(calls) == 4
    assert all(all(path.startswith("meta/") for path in paths) for _, paths in calls)
    assert not list((tmp_path / "subset").rglob("*.parquet"))
    assert not (tmp_path / "subset/manifest.json").exists()


def test_corrupt_parquet_is_rejected_before_ready_manifest(tmp_path, hub):
    tasks, _ = hub
    output = tmp_path / "subset"
    entry = prepare(tasks, output)[0]
    path = output / entry["root"] / entry["episodes"][0]["path"]
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(pa.ArrowInvalid):
        prepare(tasks, output)
    assert not (output / "manifest.json").exists()


def test_changed_budget_cannot_reuse_old_selection(tmp_path, hub):
    tasks, _ = hub
    output = tmp_path / "subset"
    prepare(tasks, output, dry_run=True)
    tasks[0]["num_val_episodes"] = 1
    with pytest.raises(ValueError, match="Existing selection differs"):
        prepare(tasks, output, dry_run=True)


def test_shell_help_and_syntax():
    script = Path(__file__).resolve().parents[2] / "scripts/download_encoder_data.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    result = subprocess.run(["bash", str(script), "--help"], check=True, capture_output=True, text=True)
    assert "300 条训练 + 35 条" in result.stdout


def test_rate_limit_after_partial_download_retries_only_missing_episodes(tmp_path, hub, monkeypatch):
    import huggingface_hub
    import requests
    from huggingface_hub.errors import HfHubHTTPError

    from utils import hf_download

    tasks, calls = hub
    output = tmp_path / "subset"
    selected = prepare(tasks[:1], output, dry_run=True)[0]
    bad = selected["episodes"][1]["path"]
    download_file = huggingface_hub.hf_hub_download
    episode_attempts, sleeps = [], []

    def interrupted_download(**kwargs):
        filename = kwargs["filename"]
        first_failure = filename == bad and bad not in episode_attempts
        episode_attempts.append(filename)
        if first_failure:
            response = requests.Response()
            response.status_code = 429
            response.headers["Retry-After"] = "120"
            raise HfHubHTTPError("Xet token rate limit", response=response)
        return download_file(**kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", interrupted_download)
    monkeypatch.setattr(hf_download.time, "sleep", sleeps.append)
    entry = prepare(tasks[:1], output)[0]
    assert sleeps == [120]
    assert episode_attempts.count(bad) == 2
    assert all(episode_attempts.count(row["path"]) == 1 for row in entry["episodes"] if row["path"] != bad)
    assert len(entry["train_episode_ids"]) == 3
    assert len(entry["val_episode_ids"]) == 2
    assert len([paths for _, paths in calls if paths[0].startswith("data/")]) == 5


@pytest.mark.parametrize("existing_bad_file", [False, True])
@pytest.mark.parametrize("failure", ["size", "range"])
def test_size_mismatch_repairs_one_episode_and_preserves_completed_files(
    tmp_path, hub, monkeypatch, existing_bad_file, failure
):
    import huggingface_hub
    import requests
    from huggingface_hub.errors import HfHubHTTPError

    tasks, calls = hub
    output = tmp_path / "subset"
    entry = prepare(tasks[:1], output, dry_run=True)[0]
    root = output / entry["root"]
    first, bad, *remaining = entry["episodes"]
    snapshot = huggingface_hub.snapshot_download
    download_file = huggingface_hub.hf_hub_download
    snapshot(
        entry["repo_id"],
        repo_type="dataset",
        revision=entry["revision"],
        local_dir=root,
        allow_patterns=[first["path"]],
    )
    preserved = (root / first["path"]).stat().st_mtime_ns
    if existing_bad_file:
        destination = root / bad["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"bad")
    normal, forced = [], []

    def download_with_broken_partial(**kwargs):
        assert kwargs["repo_id"] == entry["repo_id"]
        assert kwargs["revision"] == entry["revision"]
        if kwargs["force_download"]:
            forced.append(kwargs)
            assert kwargs["filename"] == bad["path"]
        else:
            normal.append(kwargs["filename"])
            if kwargs["filename"] == bad["path"]:
                if failure == "range":
                    response = requests.Response()
                    response.status_code = 416
                    # Real CDN URLs can be opaque: repair must not parse a filename from them.
                    response.url = "https://cdn.example.test/opaque-object"
                    raise HfHubHTTPError("Range Not Satisfiable", response=response)
                raise OSError(
                    f"Consistency check failed: file should be of size {bad['size_bytes']} but has size {bad['size_bytes'] + 20} ({Path(bad['path']).name})."
                )
        return download_file(**kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download_with_broken_partial)
    resumed = prepare(tasks[:1], output)[0]
    assert resumed == entry
    assert len(forced) == 1
    assert {row["path"] for row in remaining} <= set(normal)
    assert first["path"] not in normal
    assert (root / first["path"]).stat().st_mtime_ns == preserved
