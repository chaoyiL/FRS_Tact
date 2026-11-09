"""Download the four frozen native Pi0.5 checkpoints used by shared FRS."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import validate_repo_id

from train_pi05_frs.tools.train_shared_frs import DEFAULT_CONFIG, load_shared_config

MARKER = ".frs_pi05_source.json"


def download_plan(source: dict, api: HfApi) -> dict:
    repo, revision = source["checkpoint_repo"], source["checkpoint_revision"]
    validate_repo_id(repo)
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{source['name']}: checkpoint_revision must be a pinned commit SHA")
    target = Path(source["checkpoint"]).expanduser()
    if not target.is_absolute():
        raise ValueError("Checkpoint download targets must be absolute paths")
    asset = source["norm_stats_asset_id"]
    if not isinstance(asset, str) or not asset or "/" in asset or asset in (".", ".."):
        raise ValueError("Set an explicit norm_stats_asset_id before downloading")
    norm_path = f"assets/{asset}/norm_stats.json"
    info = api.model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError(f"{repo}: resolved checkpoint revision does not match configuration")
    # Include the entire Orbax parameter tree, including sharding/OCDBT metadata.
    # train_state contains optimizer state and is not needed for frozen inference.
    files = {}
    for item in info.siblings:
        name = item.rfilename
        if not (name.startswith("params/") or name in (norm_path, "_CHECKPOINT_METADATA")):
            continue
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"Unsafe checkpoint filename: {name}")
        if not isinstance(item.size, int) or item.size <= 0:
            raise ValueError(f"Missing size or empty checkpoint file: {name}")
        files[name] = item.size
    required = {"params/_METADATA", "params/manifest.ocdbt", norm_path, "_CHECKPOINT_METADATA"}
    if required - files.keys() or not any(name.startswith("params/d/") for name in files):
        raise ValueError(f"{repo}: incomplete native Orbax params/assets layout")
    return {"source": source, "target": target, "files": files, "norm_path": norm_path}


def validate_local(plan: dict) -> None:
    root, source = plan["target"], plan["source"]
    for name, size in plan["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Checkpoint file escapes download target: {path}")
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Missing/incomplete checkpoint file: {path}")
    stats = json.loads((root / plan["norm_path"]).read_text())["norm_stats"]
    width = source["native_action_dim"]
    for key, expected in (("state", 7 if width == 10 else 20), ("actions", width)):
        for field in ("mean", "std", "q01", "q99"):
            if len(stats[key][field]) != expected:
                raise ValueError(f"{source['name']}: wrong {key}.{field} dimension; expected {expected}")


def download_one(plan: dict, *, max_workers: int) -> None:
    root, source = plan["target"], plan["source"]
    identity = {
        "format_version": 1,
        "repo_id": source["checkpoint_repo"],
        "revision": source["checkpoint_revision"],
    }
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError(f"Invalid checkpoint directory: {root}")
    marker = root / MARKER
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError(f"Checkpoint belongs to a different repository/revision: {root}")
    elif root.exists() and any(root.iterdir()):
        raise ValueError(f"Existing nonempty checkpoint has no {MARKER}; choose a new target: {root}")
    root.mkdir(parents=True, exist_ok=True)
    # Written before download so an interrupted download can be safely resumed.
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(identity, indent=2) + "\n")
    temporary.replace(marker)
    snapshot_download(
        repo_id=identity["repo_id"],
        repo_type="model",
        revision=identity["revision"],
        local_dir=root,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
        allow_patterns=list(plan["files"]),
        max_workers=max_workers,
    )
    validate_local(plan)
    print(f"Validated {source['name']}: {root}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Inspect remote metadata; download no weights")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    config = load_shared_config(args.config.resolve())
    os.environ.setdefault("HF_HOME", str(Path(config["workspace"]) / "huggingface"))
    # Pass cache root explicitly: huggingface_hub reads environment at import time.
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    api = HfApi()
    plans = [download_plan(source, api) for source in config["sources"]]
    targets = [plan["target"].resolve() for plan in plans]
    if any(
        a == b or a in b.parents or b in a.parents for i, a in enumerate(targets) for b in targets[i + 1 :]
    ):
        raise ValueError("Checkpoint download targets must be separate directories")
    for plan in plans:
        source = plan["source"]
        print(
            f"{source['name']}: {source['checkpoint_repo']} @ {source['checkpoint_revision']}\n"
            f"  -> {plan['target']} ({sum(plan['files'].values()) / 2**30:.2f} GiB)",
            flush=True,
        )
    print(f"Total: {sum(sum(p['files'].values()) for p in plans) / 2**30:.2f} GiB", flush=True)
    if not args.dry_run:
        for plan in plans:
            download_one(plan, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
