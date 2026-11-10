"""Validate and upload one immutable shared Pi0.5 FRS checkpoint snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_pi05_frs.tools.train_shared_frs import DEFAULT_CONFIG, load_shared_config


def inspect_shared_checkpoint(directory: Path) -> tuple[Path, dict]:
    from train_pi05_frs.utils.checkpoint import _load_metadata

    # Resolve best once, before hashing or uploading: it can point outside best/.
    snapshot, metadata = _load_metadata(directory)
    if int(metadata.get("version", 0)) != 3:
        raise ValueError("Shared FRS uploader requires a v3 checkpoint with file checksums")
    extra = metadata.get("extra_metadata", {})
    decoder = metadata.get("decoder_config", {})
    if extra.get("shared_frs_interface_version") != 1 or (
        decoder.get("action_dim") != 20
        or decoder.get("num_tactile_tokens") != 4
        or not decoder.get("tactile_token_identity")
    ):
        raise ValueError("Checkpoint is not the shared 20D / 4-token Pi0.5 FRS")
    return snapshot, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Target HF model repository, OWNER/REPO")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG.with_name("train_pi05_frs_shared_four_tasks_4090.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, help="Default: best/ under the configured output")
    parser.add_argument("--dry-run", action="store_true", help="Validate locally; make no Hub requests")
    parser.add_argument("--private", action="store_true", help="Create a new repository as private")
    args = parser.parse_args(argv)
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(args.repo_id)
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = Path(load_shared_config(args.config.resolve())["output"]) / "best"
    snapshot, metadata = inspect_shared_checkpoint(checkpoint)
    files = sorted(metadata["files"])
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "checkpoint": str(checkpoint),
                "resolved_snapshot": str(snapshot),
                "epoch": metadata["epoch"],
                "files": files,
                "total_bytes": sum((snapshot / name).stat().st_size for name in files),
                "metrics": metadata["metrics"],
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        print("Validation passed. No Hub requests were made.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    # Keep all checksum-listed files, including optimizer files when present.
    # Omitting those while leaving checkpoint.json intact breaks the v3 loader.
    # This uploads the pinned generation as regular files at the repository root.
    # It does not upload historical generations, VLA weights, data or action caches.
    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(snapshot),
        allow_patterns=files,
        commit_message=f"Upload shared Pi0.5 FRS checkpoint at epoch {metadata['epoch']}",
    )
    print(f"Uploaded: https://huggingface.co/{args.repo_id}", flush=True)
    print(f"Commit: {result}", flush=True)


if __name__ == "__main__":
    main()
