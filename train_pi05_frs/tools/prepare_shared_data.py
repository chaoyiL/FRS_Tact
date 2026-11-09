"""Download only selected v2.1 episodes and prepare complete local v3 subsets for FRS."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import yaml

from train_pi05_frs.tools.train_shared_frs import DEFAULT_CONFIG, load_shared_config
from utils.hf_download import retry_hf_call


def normalize_v21_subset(entry: dict, raw_root: Path, destination: Path) -> None:
    """Reindex only a generated subset, retaining every state/action/image column."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    info = json.loads((raw_root / "meta/info.json").read_text())
    rows = entry["episodes"]
    ids = [row["episode_index"] for row in rows]
    episodes = {
        row["episode_index"]: row
        for row in (
            json.loads(line)
            for line in (raw_root / "meta/episodes.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    stats = {
        row["episode_index"]: row
        for row in (
            json.loads(line)
            for line in (raw_root / "meta/episodes_stats.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    (destination / "meta").mkdir(parents=True)
    shutil.copyfile(raw_root / "meta/tasks.jsonl", destination / "meta/tasks.jsonl")
    new_episodes, new_stats, offset = [], [], 0
    for local, original in enumerate(ids):
        table = pq.read_table(raw_root / rows[local]["path"])
        if table.num_rows != rows[local]["length"]:
            raise ValueError("Selected Parquet length changed")
        old_start = int(table["index"][0].as_py())
        for key, values in (
            ("episode_index", np.full(len(table), local)),
            ("index", np.arange(offset, offset + len(table))),
        ):
            column = table.schema.get_field_index(key)
            table = table.set_column(
                column, table.schema.field(column), pa.array(values, type=table.schema.field(column).type)
            )
        path = destination / info["data_path"].format(
            episode_index=local, episode_chunk=local // info["chunks_size"]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        new_episodes.append({**episodes[original], "episode_index": local})
        stat = json.loads(json.dumps(stats[original]))
        stat["episode_index"] = local
        for key, delta in (("episode_index", local - original), ("index", offset - old_start)):
            for field in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
                if field in stat["stats"].get(key, {}):
                    stat["stats"][key][field] = (np.asarray(stat["stats"][key][field]) + delta).tolist()
        new_stats.append(stat)
        offset += len(table)
    for filename, values in (("episodes.jsonl", new_episodes), ("episodes_stats.jsonl", new_stats)):
        (destination / "meta" / filename).write_text("".join(json.dumps(value) + "\n" for value in values))
    info.update(
        total_episodes=len(rows),
        total_frames=offset,
        total_chunks=(len(rows) + info["chunks_size"] - 1) // info["chunks_size"],
        splits={"train": f"0:{len(rows)}"},
    )
    (destination / "meta/info.json").write_text(json.dumps(info, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args(argv)
    config = load_shared_config(args.config.resolve())
    work = Path(config["workspace"])
    os.environ.setdefault("HF_HOME", str(work / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets_arrow"))
    raw_manifest = work / "datasets/raw/manifest.json"
    plan = {
        "split_seed": config["seed"],
        "dataset_manifest": str(raw_manifest),
        "tasks": [
            {
                "name": s["name"],
                "dataset_repo_ids": [s["repo_id"]],
                "sides": s["active_wrists"],
                "num_train_episodes": config["num_train_episodes"],
                "num_val_episodes": config["num_val_episodes"],
            }
            for s in config["sources"]
        ],
    }
    work.mkdir(parents=True, exist_ok=True)
    plan_path = work / "subset_download.yaml"
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False))
    from train_encoder.download_data import main as download

    download_args = ["--config", str(plan_path), "--max-workers", str(args.max_workers)]
    if args.dry_run:
        download_args.append("--dry-run")
    download(download_args)
    if args.dry_run:
        return
    from huggingface_hub import snapshot_download

    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset
    from utils.cache import atomic_write_json

    manifest = json.loads(raw_manifest.read_text())
    by_repo = {entry["repo_id"]: entry for entry in manifest["tasks"]}
    for source in config["sources"]:
        entry = by_repo[source["repo_id"]]
        root = raw_manifest.parent / entry["root"]
        destination = Path(source["dataset_root"])
        identity = {
            "repo_id": entry["repo_id"],
            "revision": entry["revision"],
            "source_episode_ids": [row["episode_index"] for row in entry["episodes"]],
        }
        marker = destination / "frs_subset_source.json"
        if destination.exists():
            if (
                marker.is_file()
                and json.loads(marker.read_text()) == identity
                and json.loads((destination / "meta/info.json").read_text())["codebase_version"] == "v3.0"
            ):
                print(f"Reusing prepared FRS v3 subset: {destination}", flush=True)
                continue
            raise FileExistsError(
                f"Existing dataset is not this generated subset: {destination}; use a new root"
            )
        retry_hf_call(
            snapshot_download,
            entry["repo_id"],
            repo_type="dataset",
            revision=entry["revision"],
            local_dir=root,
            allow_patterns=["meta/episodes_stats.jsonl"],
            max_workers=args.max_workers,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Temporary conversion copies are owned by this run and are removed on completion/failure.
        with tempfile.TemporaryDirectory(
            prefix=f".{source['name']}_convert_", dir=destination.parent
        ) as temporary:
            stage = Path(temporary) / "dataset"
            normalize_v21_subset(entry, root, stage)
            convert_dataset(entry["repo_id"], root=stage, push_to_hub=False, force_conversion=True)
            atomic_write_json(stage / "frs_subset_source.json", identity)
            shutil.move(str(stage), str(destination))
        print(f"Prepared 335-episode FRS dataset: {destination}", flush=True)


if __name__ == "__main__":
    main()
