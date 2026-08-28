"""Create a lightweight manifest for several compatible LeRobot v2.1 roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def write_multiroot_manifest(
    roots: list[str | Path],
    output_path: str | Path,
    dataset_id: str | None = None,
    state_action_profile: str | None = None,
) -> dict:
    resolved = sorted({Path(root).expanduser().resolve() for root in roots})
    if not resolved:
        raise ValueError("At least one LeRobot dataset root is required")
    sources = []
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
        sources.append(
            {
                "name": name,
                "path": str(root),
                "source_dataset_id": info.get("repo_id") or name,
                "total_episodes": int(info["total_episodes"]),
                "total_frames": int(info["total_frames"]),
            }
        )
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
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()
    manifest = write_multiroot_manifest(
        args.roots,
        args.output,
        args.dataset_id,
        args.state_action_profile,
    )
    print(json.dumps({"event": "lerobot_multiroot_manifest_ready", **manifest}))


if __name__ == "__main__":
    main()
