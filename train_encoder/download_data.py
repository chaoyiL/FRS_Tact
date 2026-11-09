"""Download exact episode subsets of the four v2.1 encoder datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

import yaml

from train_encoder.utils.tasks import normalize_tasks, sample_task_episodes
from utils.cache import atomic_write_json
from utils.hf_download import retry_hf_call

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FORMAT = "tactile-encoder-v21-subset-v1"


def episode_path(info: dict, episode: int) -> str:
    path = info["data_path"].format(episode_index=episode, episode_chunk=episode // int(info["chunks_size"]))
    if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
        raise ValueError(f"Unsafe dataset file path: {path!r}")
    return path


def build_plan(
    task: dict,
    *,
    info: dict,
    episodes: list[dict],
    files: dict[str, int],
    revision: str,
    requested_revision: str,
    seed: int,
    task_index: int,
) -> dict:
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"{task['name']}: expected v2.1 episode Parquet files.")
    if any(feature.get("dtype") == "video" for feature in info["features"].values()):
        raise ValueError("This downloader expects embedded images, not separate video files.")
    if len(task["dataset_repo_ids"]) != 1:
        raise ValueError("The subset downloader requires exactly one repo per task.")
    by_id = {int(row["episode_index"]): row for row in episodes}
    if len(by_id) != len(episodes) or len(by_id) != info["total_episodes"]:
        raise ValueError("Episode metadata does not match total_episodes.")
    train, val = sample_task_episodes(
        sorted(by_id),
        train_count=task["num_train_episodes"],
        val_count=task["num_val_episodes"],
        seed=seed,
        task_index=task_index,
    )
    selected = []
    for episode in sorted((*train, *val)):
        path = episode_path(info, episode)
        if path not in files or not files[path]:
            raise ValueError(f"Missing remote episode file or size: {path}")
        selected.append(
            {
                "episode_index": episode,
                "path": path,
                "length": int(by_id[episode]["length"]),
                "size_bytes": files[path],
            }
        )
    return {
        "name": task["name"],
        "repo_id": task["dataset_repo_ids"][0],
        "sides": task["sides"],
        "revision": revision,
        "requested_revision": requested_revision,
        "split_seed": seed,
        "task_index": task_index,
        "source_total_episodes": info["total_episodes"],
        "train_episode_ids": train,
        "val_episode_ids": val,
        "episodes": selected,
    }


def validate_local_episode(path: Path, episode: dict, image_keys: list[str]) -> None:
    import pyarrow.parquet as pq

    if not path.is_file() or path.stat().st_size != episode["size_bytes"]:
        raise ValueError(f"Episode file missing or size mismatch: {path}")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != episode["length"]:
        raise ValueError(f"Episode length mismatch: {path}")
    required = {"episode_index", "timestamp", *image_keys}
    if not required <= set(parquet.schema_arrow.names):
        raise ValueError(f"Episode lacks encoder columns: {path}")
    ids = parquet.read(columns=["episode_index"]).column(0).to_numpy()
    if not len(ids) or not (ids == episode["episode_index"]).all():
        raise ValueError(f"Wrong episode_index in {path}")


def required_image_keys(sides: list[str]) -> list[str]:
    return [
        key
        for side, index in (("left", 0), ("right", 1))
        if side in sides
        for key in (
            f"observation.images.camera{index}",
            f"observation.images.tactile_left_{index}",
            f"observation.images.tactile_right_{index}",
        )
    ]


def download_task(
    task: dict,
    *,
    output_dir: Path,
    seed: int,
    task_index: int,
    revision: str,
    max_workers: int,
    dry_run: bool,
) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    if len(task["dataset_repo_ids"]) != 1:
        raise ValueError("The subset downloader requires exactly one repo per task.")
    repo_id = task["dataset_repo_ids"][0]
    if (
        len(PurePosixPath(repo_id).parts) != 2
        or ".." in PurePosixPath(repo_id).parts
        or repo_id.startswith("/")
    ):
        raise ValueError(f"Invalid repo id: {repo_id!r}")
    root = output_dir / "datasets" / repo_id
    plan_path = root / "selection.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if (
            plan["repo_id"] != repo_id
            or plan["name"] != task["name"]
            or plan["sides"] != task["sides"]
            or plan["split_seed"] != seed
            or plan["task_index"] != task_index
            or plan["requested_revision"] != revision
            or len(plan["train_episode_ids"]) != task["num_train_episodes"]
            or len(plan["val_episode_ids"]) != task["num_val_episodes"]
        ):
            raise ValueError(
                f"Existing selection differs from configuration: {plan_path}. Use a new --output-dir."
            )
    else:
        remote = retry_hf_call(HfApi().dataset_info, repo_id, revision=revision, files_metadata=True)
        retry_hf_call(
            snapshot_download,
            repo_id,
            repo_type="dataset",
            revision=remote.sha,
            local_dir=root,
            allow_patterns=["meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl"],
            max_workers=max_workers,
        )
        info = json.loads((root / "meta/info.json").read_text())
        episodes = [
            json.loads(line)
            for line in (root / "meta/episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        plan = build_plan(
            task,
            info=info,
            episodes=episodes,
            files={s.rfilename: s.size for s in remote.siblings},
            revision=remote.sha,
            requested_revision=revision,
            seed=seed,
            task_index=task_index,
        )
        atomic_write_json(plan_path, plan)
    keys = required_image_keys(task["sides"])
    info = json.loads((root / "meta/info.json").read_text())
    if not set(keys) <= info["features"].keys():
        raise ValueError(f"{repo_id}: missing required image keys {keys}")
    size_gib = sum(row["size_bytes"] for row in plan["episodes"]) / 1024**3
    print(
        f"{repo_id}: train={len(plan['train_episode_ids'])} val={len(plan['val_episode_ids'])} "
        f"files={len(plan['episodes'])} size={size_gib:.2f} GiB commit={plan['revision']}",
        flush=True,
    )
    if not dry_run:
        def fetch_missing():
            # Recompute after a partial transfer so retries avoid even metadata requests
            # for episodes that finished downloading during the previous attempt.
            missing = [
                row["path"]
                for row in plan["episodes"]
                if not (root / row["path"]).is_file()
                or (root / row["path"]).stat().st_size != row["size_bytes"]
            ]
            if missing:
                snapshot_download(
                    repo_id,
                    repo_type="dataset",
                    revision=plan["revision"],
                    local_dir=root,
                    allow_patterns=missing,
                    max_workers=max_workers,
                )

        retry_hf_call(fetch_missing)
        for row in plan["episodes"]:
            validate_local_episode(root / row["path"], row, keys)
    return {**plan, "root": root.relative_to(output_dir).as_posix()}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "train_encoder/configs/train.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    config = yaml.safe_load(args.config.read_text())
    tasks = normalize_tasks(config["tasks"])
    default_manifest = Path(config["dataset_manifest"])
    if not default_manifest.is_absolute():
        default_manifest = PROJECT_ROOT / default_manifest
    output_dir = (args.output_dir or default_manifest.parent).resolve()
    manifest = {"format": MANIFEST_FORMAT, "tasks": []}
    for task_index, task in enumerate(tasks):
        manifest["tasks"].append(
            download_task(
                task,
                output_dir=output_dir,
                seed=int(config["split_seed"]),
                task_index=task_index,
                revision=args.revision,
                max_workers=args.max_workers,
                dry_run=args.dry_run,
            )
        )
    if args.dry_run:
        print("Dry run complete: no episode files downloaded; no ready manifest written.")
        return
    path = output_dir / "manifest.json"
    atomic_write_json(path, manifest)
    print(f"Verified local subset: {path}", flush=True)
    print(f"Train: python -m train_encoder.train --config {args.config.resolve()} --dataset-manifest {path}")


if __name__ == "__main__":
    main()
