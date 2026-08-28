"""Create a lightweight manifest for several compatible LeRobot v2.1 roots."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError

from .state_action_profiles import PROFILES, resolve_state_action_profile


MANIFEST_FORMAT = "deco-lerobot-v21-multiroot-v1"
REQUIRED_FEATURES = (
    "observation.images.camera0",
    "observation.images.camera1",
    "observation.state",
    "actions",
)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"LeRobot info.json is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_tasks(root: Path) -> list[dict]:
    path = root / "meta/tasks.jsonl"
    if not path.is_file():
        raise ValueError(f"LeRobot tasks.jsonl is missing: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _source_contract(root: Path, info: dict) -> dict:
    if info.get("codebase_version") != "v2.1":
        raise ValueError(
            f"Expected LeRobot v2.1 data at {root}, got {info.get('codebase_version')!r}"
        )
    features = info.get("features", {})
    missing = [key for key in REQUIRED_FEATURES if key not in features]
    if missing:
        raise ValueError(f"LeRobot features missing at {root}: {missing}")
    return {
        "fps": float(info["fps"]),
        "camera0_shape": features[REQUIRED_FEATURES[0]]["shape"],
        "camera1_shape": features[REQUIRED_FEATURES[1]]["shape"],
        "state_shape": features["observation.state"]["shape"],
        "action_shape": features["actions"]["shape"],
        "video_path": info.get("video_path"),
        "tasks": _read_tasks(root),
    }


def _episode_path(root: Path, info: dict, episode_index: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    if chunk_size <= 0:
        raise ValueError(f"LeRobot chunks_size must be positive, got {chunk_size}")
    episode_chunk = episode_index // chunk_size
    relative = str(info["data_path"]).format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
        chunk_index=episode_chunk,
        file_index=episode_index,
    )
    return root / relative


def _require_camera0_pure_black(root: Path, info: dict) -> dict[str, int | bool]:
    episodes_path = root / "meta/episodes.jsonl"
    if not episodes_path.is_file():
        raise ValueError(f"LeRobot episodes.jsonl is missing: {episodes_path}")
    episodes = []
    seen_episode_indices = set()
    for line_number, line in enumerate(
        episodes_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            episode = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL at {episodes_path}:{line_number}"
            ) from exc
        episode_index = int(episode["episode_index"])
        if episode_index in seen_episode_indices:
            raise ValueError(
                f"Duplicate episode_index in episodes.jsonl: {episode_index}"
            )
        seen_episode_indices.add(episode_index)
        length = int(episode["length"])
        if length <= 0:
            raise ValueError(
                f"Episode length must be positive: episode={episode_index}, length={length}"
            )
        episodes.append((episode_index, length))
    expected_episode_count = int(info.get("total_episodes", -1))
    if len(episodes) != expected_episode_count:
        raise ValueError(
            "LeRobot total_episodes disagrees with episodes.jsonl during camera0 preflight: "
            f"info={expected_episode_count}, rows={len(episodes)}"
        )
    metadata_frame_count = sum(length for _, length in episodes)
    expected_frame_count = int(info.get("total_frames", -1))
    if metadata_frame_count != expected_frame_count:
        raise ValueError(
            "LeRobot total_frames disagrees with episodes.jsonl during camera0 preflight: "
            f"info={expected_frame_count}, metadata={metadata_frame_count}"
        )

    checked_payloads: set[str] = set()
    frame_count = 0
    for episode_index, expected_length in episodes:
        parquet_path = _episode_path(root, info, episode_index)
        if not parquet_path.is_file():
            raise ValueError(f"LeRobot episode Parquet is missing: {parquet_path}")
        try:
            column = pq.read_table(
                parquet_path,
                columns=["observation.images.camera0"],
            )["observation.images.camera0"]
        except Exception as exc:
            raise ValueError(
                f"Cannot read camera0 for black-image preflight: {parquet_path}"
            ) from exc
        if len(column) != expected_length:
            raise ValueError(
                "camera0 row count disagrees with episodes.jsonl: "
                f"episode={episode_index}, metadata={expected_length}, parquet={len(column)}"
            )
        for row_index, encoded in enumerate(column.to_pylist()):
            frame_count += 1
            payload = encoded.get("bytes") if isinstance(encoded, dict) else None
            if not payload:
                raise ValueError(
                    "Missing embedded JPEG bytes during camera0 black-image preflight: "
                    f"root={root}, episode={episode_index}, row={row_index}"
                )
            digest = hashlib.sha256(payload).hexdigest()
            if digest in checked_payloads:
                continue
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError(
                    "Cannot decode camera0 during black-image preflight: "
                    f"root={root}, episode={episode_index}, row={row_index}"
                ) from exc
            if np.any(pixels):
                raise ValueError(
                    "camera0 is not pure black: "
                    f"root={root}, episode={episode_index}, row={row_index}, "
                    f"max_pixel={int(pixels.max())}"
                )
            checked_payloads.add(digest)
    if not checked_payloads:
        raise ValueError(f"camera0 black-image preflight found no images: {root}")
    if frame_count != expected_frame_count:
        raise ValueError(
            "camera0 preflight frame count disagrees with info.json: "
            f"info={expected_frame_count}, checked={frame_count}"
        )
    return {
        "pure_black": True,
        "frames": frame_count,
        "unique_payloads_decoded": len(checked_payloads),
    }


def write_multiroot_manifest(
    roots: list[str | Path],
    output_path: str | Path,
    dataset_id: str | None = None,
    state_action_profile: str | None = None,
    require_black_camera0: bool = False,
) -> dict:
    resolved = sorted({Path(root).expanduser().resolve() for root in roots})
    if not resolved:
        raise ValueError("At least one LeRobot dataset root is required")
    validated_sources = []
    expected_contract = None
    names = set()
    resolved_profile = None
    for root in resolved:
        info = _read_json(root / "meta/info.json")
        contract = _source_contract(root, info)
        if expected_contract is None:
            expected_contract = contract
        elif contract != expected_contract:
            raise ValueError(f"LeRobot source contract differs: {root}")
        current_profile = resolve_state_action_profile(
            state_action_profile,
            tuple(int(value) for value in contract["state_shape"]),
            tuple(int(value) for value in contract["action_shape"]),
        )
        if resolved_profile is None:
            resolved_profile = current_profile
        elif current_profile != resolved_profile:
            raise ValueError(f"LeRobot state/action profile differs: {root}")
        name = root.name
        if name in names:
            raise ValueError(f"Duplicate LeRobot source name: {name}")
        names.add(name)
        validated_sources.append((root, info, name))

    sources = []
    for root, info, name in validated_sources:
        camera0_black_preflight = (
            _require_camera0_pure_black(root, info)
            if require_black_camera0
            else None
        )
        source = {
            "name": name,
            "path": str(root),
            "source_dataset_id": info.get("repo_id") or name,
            "total_episodes": int(info["total_episodes"]),
            "total_frames": int(info["total_frames"]),
        }
        if camera0_black_preflight is not None:
            source["camera0_black_preflight"] = camera0_black_preflight
        sources.append(source)
    payload = {
        "format": MANIFEST_FORMAT,
        "dataset_id": dataset_id or "+".join(source["name"] for source in sources),
        "state_action_profile": resolved_profile.name,
        "sources": sources,
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument("--state-action-profile", choices=sorted(PROFILES))
    parser.add_argument("--require-black-camera0", action="store_true")
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()
    manifest = write_multiroot_manifest(
        args.roots,
        args.output,
        args.dataset_id,
        args.state_action_profile,
        args.require_black_camera0,
    )
    print(json.dumps({"event": "lerobot_multiroot_manifest_ready", **manifest}))


if __name__ == "__main__":
    main()
