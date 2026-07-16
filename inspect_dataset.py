#!/usr/bin/env python
"""Inspect a local LeRobot dataset and save inference-relevant schema info as JSON.

Usage:
    uv sync --extra infer
    uv run python inspect_dataset.py --dataset-path /path/to/dataset
    uv run python inspect_dataset.py --dataset-path /path/to/dataset -o report.json
    uv run python inspect_dataset.py --dataset-path /path/to/dataset --model lerobot/smolvla_base --print-text
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR
from lerobot.utils.feature_utils import dataset_to_policy_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a LeRobot dataset and print schema info for SmolVLA inference."
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        help="Local root directory of the LeRobot dataset (contains meta/, data/, videos/).",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Dataset repo id. Defaults to the dataset directory name.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Frame index to sample for dtype/shape inspection.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional SmolVLA checkpoint id/path to compare expected policy features.",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        type=Path,
        default=None,
        help="Path to save the inspection report as JSON. Defaults to <dataset_name>_inspect.json.",
    )
    parser.add_argument(
        "--print-text",
        action="store_true",
        help="Also print a human-readable summary to stdout.",
    )
    parser.add_argument(
        "--no-load-videos",
        action="store_true",
        help="Skip loading the full dataset (metadata only, no frame sample).",
    )
    return parser.parse_args()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _tensor_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu()
        return {
            "type": "torch.Tensor",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "min": float(arr.min()) if arr.numel() else None,
            "max": float(arr.max()) if arr.numel() else None,
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "numpy.ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "min": float(value.min()) if value.size else None,
            "max": float(value.max()) if value.size else None,
        }
    if isinstance(value, (int, float, str, bool)):
        return {"type": type(value).__name__, "value": value}
    return {"type": type(value).__name__, "repr": repr(value)}


def _safe_dataset_to_policy_features(features: dict[str, dict]) -> dict[str, Any]:
    patched_features = {}
    for key, feat in features.items():
        feat_copy = dict(feat)
        if feat_copy.get("dtype") in ("image", "video") and "names" not in feat_copy:
            shape = feat_copy.get("shape", ())
            if len(shape) == 3:
                feat_copy["names"] = ["height", "width", "channels"]
        patched_features[key] = feat_copy
    return dataset_to_policy_features(patched_features)


def _feature_groups(features: dict[str, dict]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "images": [],
        "state": [],
        "action": [],
        "other": [],
    }
    for key in features:
        if key.startswith(f"{OBS_IMAGES}."):
            groups["images"].append(key)
        elif key == OBS_STATE:
            groups["state"].append(key)
        elif key == ACTION:
            groups["action"].append(key)
        else:
            groups["other"].append(key)
    return groups


def _stats_summary(stats: dict[str, Any] | None) -> dict[str, Any]:
    if not stats:
        return {}
    summary: dict[str, Any] = {}
    for key, values in stats.items():
        if not isinstance(values, dict):
            continue
        entry: dict[str, Any] = {}
        for stat_name, stat_value in values.items():
            if isinstance(stat_value, (list, tuple, np.ndarray)):
                flat = np.asarray(stat_value).reshape(-1)
                entry[stat_name] = {
                    "shape": list(np.asarray(stat_value).shape),
                    "preview": flat[: min(8, flat.size)].tolist(),
                }
            else:
                entry[stat_name] = stat_value
        summary[key] = entry
    return summary


def _chw_to_hwc_uint8(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image tensor/array, got shape {image.shape}")
    if image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
        image = image * 255.0
    return image.astype(np.uint8)


def frame_to_raw_observation(sample: dict[str, Any], features: dict[str, dict]) -> dict[str, Any]:
    """Convert a dataset frame into the raw dict expected by infer_smolvla helpers."""
    observation: dict[str, Any] = {}
    for key, feat in features.items():
        if key not in sample:
            continue
        value = sample[key]
        if feat.get("dtype") in ("image", "video"):
            observation[key] = _chw_to_hwc_uint8(value)
        elif key == OBS_STATE:
            state = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            observation[key] = state.astype(np.float32)
    if "task" in sample:
        observation["task"] = sample["task"]
    return observation


def _build_infer_snippet(raw_obs: dict[str, Any], task: str | None) -> str:
    lines = [
        "obs = {",
    ]
    for key, value in raw_obs.items():
        if key == "task":
            continue
        if key.startswith(f"{OBS_IMAGES}."):
            shape = tuple(value.shape) if hasattr(value, "shape") else "?"
            lines.append(f'    "{key}": <np.ndarray shape={shape}, dtype=uint8, HWC>,')
        elif key == OBS_STATE:
            shape = tuple(value.shape) if hasattr(value, "shape") else "?"
            lines.append(f'    "{key}": <np.ndarray shape={shape}, dtype=float32>,')
    lines.append("}")
    if task:
        lines.append(f'obs["task"] = {task!r}')
    lines.extend(
        [
            "",
            "observation = prepare_observation_for_inference(",
            "    obs,",
            "    policy.config.device,",
            f"    task={task!r},",
            "    robot_type=meta.info.robot_type,",
            ")",
        ]
    )
    return "\n".join(lines)


def _compare_with_model(model_id: str, policy_features: dict[str, Any]) -> dict[str, Any]:
    from lerobot.policies.smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(model_id)
    expected = {
        key: {"type": feat.type.name, "shape": list(feat.shape)}
        for key, feat in policy.config.input_features.items()
    }
    provided = {
        key: {"type": feat.type.name, "shape": list(feat.shape)}
        for key, feat in policy_features.items()
        if feat.type in (FeatureType.VISUAL, FeatureType.STATE, FeatureType.ACTION)
    }
    expected_visual = {k for k, v in policy.config.input_features.items() if v.type == FeatureType.VISUAL}
    provided_visual = {k for k, v in policy_features.items() if v.type == FeatureType.VISUAL}
    rename_suggestions = {}
    if expected_visual and provided_visual and expected_visual != provided_visual:
        exp = sorted(expected_visual)
        got = sorted(provided_visual)
        for src, dst in zip(got, exp, strict=False):
            rename_suggestions[src] = dst
    return {
        "model": model_id,
        "expected_input_features": expected,
        "dataset_policy_features": provided,
        "missing_from_dataset": sorted(set(expected) - set(provided)),
        "extra_in_dataset": sorted(set(provided) - set(expected)),
        "suggested_rename_map": rename_suggestions,
        "action_dim_model": policy.config.action_feature.shape[0] if policy.config.action_feature else None,
        "action_dim_dataset": policy_features[ACTION].shape[0] if ACTION in policy_features else None,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not (dataset_path / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset root (missing meta/info.json): {dataset_path}")

    repo_id = args.repo_id or dataset_path.name
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_path)
    features = meta.info.features
    groups = _feature_groups(features)
    policy_features = _safe_dataset_to_policy_features(features)

    report: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "repo_id": repo_id,
        "meta": {
            "codebase_version": meta.info.codebase_version,
            "fps": meta.info.fps,
            "robot_type": meta.info.robot_type,
            "total_episodes": meta.info.total_episodes,
            "total_frames": meta.info.total_frames,
            "total_tasks": meta.info.total_tasks,
            "splits": meta.info.splits,
        },
        "feature_groups": groups,
        "features": features,
        "policy_features": {
            key: {"type": feat.type.name, "shape": list(feat.shape)}
            for key, feat in policy_features.items()
        },
        "tasks": meta.tasks.reset_index().to_dict(orient="records"),
        "stats_summary": _stats_summary(meta.stats),
    }

    if not args.no_load_videos:
        try:
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=dataset_path,
                revision=meta.info.codebase_version,
            )
        except Exception as exc:
            report["sample_error"] = (
                "Failed to load dataset frames from local files. "
                "If this is a metadata-only checkout, rerun with --no-load-videos. "
                f"Original error: {exc}"
            )
            return report

        if args.index < 0 or args.index >= len(dataset):
            raise IndexError(f"index {args.index} out of range for dataset of length {len(dataset)}")
        sample = dataset[args.index]
        raw_obs = frame_to_raw_observation(sample, features)
        report["sample_index"] = args.index
        report["sample_keys"] = sorted(sample.keys())
        report["sample_summary"] = {key: _tensor_summary(sample[key]) for key in sorted(sample.keys())}
        report["raw_observation_for_inference"] = {
            key: _tensor_summary(value) for key, value in raw_obs.items()
        }
        report["infer_code_snippet"] = _build_infer_snippet(raw_obs, sample.get("task"))
        report["dataset_length"] = len(dataset)

    if args.model:
        report["model_comparison"] = _compare_with_model(args.model, policy_features)

    return report


def print_text_report(report: dict[str, Any]) -> None:
    meta = report["meta"]
    print("=" * 72)
    print("LeRobot Dataset Inspection Report")
    print("=" * 72)
    print(f"dataset_path : {report['dataset_path']}")
    print(f"repo_id      : {report['repo_id']}")
    print(f"version      : {meta['codebase_version']}")
    print(f"fps          : {meta['fps']}")
    print(f"robot_type   : {meta['robot_type']}")
    print(f"episodes     : {meta['total_episodes']}")
    print(f"frames       : {meta['total_frames']}")
    print(f"tasks        : {meta['total_tasks']}")

    print("\n[Feature groups]")
    for group, keys in report["feature_groups"].items():
        print(f"  {group}: {keys or '-'}")

    print("\n[Features from meta/info.json]")
    for key, feat in report["features"].items():
        shape = feat.get("shape")
        names = feat.get("names")
        print(f"  - {key}: dtype={feat.get('dtype')}, shape={shape}, names={names}")

    print("\n[Policy features]")
    for key, feat in report["policy_features"].items():
        print(f"  - {key}: type={feat['type']}, shape={feat['shape']}")

    print("\n[Tasks]")
    for task in report["tasks"][:20]:
        print(f"  - task_index={task.get('task_index')}, task={task.get('task')!r}")
    if len(report["tasks"]) > 20:
        print(f"  ... ({len(report['tasks']) - 20} more)")

    if report.get("stats_summary"):
        print("\n[Stats summary]")
        for key, values in report["stats_summary"].items():
            print(f"  - {key}: {list(values.keys())}")

    if "sample_summary" in report:
        print(f"\n[Sample frame @ index {report['sample_index']}]")
        for key, summary in report["sample_summary"].items():
            print(f"  - {key}: {summary}")

        print("\n[Raw observation for infer_smolvla]")
        for key, summary in report["raw_observation_for_inference"].items():
            print(f"  - {key}: {summary}")
        if "task" in report["sample_summary"]:
            print(f"  - task: {report['sample_summary']['task']}")

        print("\n[Suggested infer snippet]")
        print(report["infer_code_snippet"])
    elif report.get("sample_error"):
        print(f"\n[Sample frame] unavailable: {report['sample_error']}")

    if "model_comparison" in report:
        cmp = report["model_comparison"]
        print(f"\n[Model comparison: {cmp['model']}]")
        print(f"  missing_from_dataset: {cmp['missing_from_dataset']}")
        print(f"  extra_in_dataset    : {cmp['extra_in_dataset']}")
        print(f"  action_dim_model    : {cmp['action_dim_model']}")
        print(f"  action_dim_dataset  : {cmp['action_dim_dataset']}")
        if cmp["suggested_rename_map"]:
            print(f"  suggested_rename_map: {json.dumps(cmp['suggested_rename_map'], ensure_ascii=False)}")


def save_json_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    args = parse_args()
    report = build_report(args)

    output_json = args.output_json or Path(f"{args.dataset_path.expanduser().resolve().name}_inspect.json")
    saved_path = save_json_report(report, output_json)
    print(f"Saved inspection report to {saved_path}")

    if args.print_text:
        print()
        print_text_report(report)


if __name__ == "__main__":
    main()
