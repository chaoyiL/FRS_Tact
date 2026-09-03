#!/usr/bin/env python3
"""Compare saved deployment RGB observations with LeRobot parquet images.

This utility is deliberately independent of policy and robot deployment code.  It
only decodes images, computes CPU image statistics, and writes JSON/CSV reports.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from deploy_deco.domain_gap import basic_image_metrics


CAMERAS = ("camera0", "camera1")
IMAGE_COLUMNS = {camera: f"observation.images.{camera}" for camera in CAMERAS}
HISTOGRAM_NAMES = ("red", "green", "blue", "luma")
STEP_PATTERN = re.compile(r"step_(\d+)$")


def jensen_shannon_distance(first, second) -> float:
    """Return the finite, base-2 Jensen-Shannon distance between histograms."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("histograms must be one-dimensional with matching shapes")
    if np.any(first < 0) or np.any(second < 0):
        raise ValueError("histograms cannot contain negative values")
    first_sum = float(first.sum())
    second_sum = float(second.sum())
    if first_sum <= 0 or second_sum <= 0:
        raise ValueError("histograms must have positive mass")
    first = first / first_sum
    second = second / second_sum
    midpoint = 0.5 * (first + second)

    def _kl_divergence(distribution, target) -> float:
        nonzero = distribution > 0
        return float(np.sum(distribution[nonzero] * np.log2(distribution[nonzero] / target[nonzero])))

    divergence = 0.5 * _kl_divergence(first, midpoint) + 0.5 * _kl_divergence(second, midpoint)
    return float(np.sqrt(max(divergence, 0.0)))


def _rgb_histograms(image: np.ndarray, bins: int = 64) -> dict[str, np.ndarray]:
    unit = image.astype(np.float32) / 255.0
    luma = 0.2126 * unit[..., 0] + 0.7152 * unit[..., 1] + 0.0722 * unit[..., 2]
    values = {
        "red": unit[..., 0],
        "green": unit[..., 1],
        "blue": unit[..., 2],
        "luma": luma,
    }
    histograms = {}
    for name, channel in values.items():
        histogram, _ = np.histogram(channel, bins=bins, range=(0.0, 1.0))
        histogram = histogram.astype(np.float64)
        histograms[name] = histogram / histogram.sum()
    return histograms


def _decode_rgb(cell: dict, parquet_path: Path) -> np.ndarray:
    if cell is None:
        raise ValueError(f"missing image value in {parquet_path}")
    encoded = cell.get("bytes")
    if encoded is not None:
        source = io.BytesIO(encoded)
    else:
        image_path = cell.get("path")
        if not image_path:
            raise ValueError(f"image value has neither bytes nor path in {parquet_path}")
        source = Path(image_path)
        if not source.is_absolute():
            source = parquet_path.parent / source
    with Image.open(source) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _sample_indices(frame_count: int, frames_per_episode: int) -> list[int]:
    if frames_per_episode <= 0 or frames_per_episode >= frame_count:
        return list(range(frame_count))
    return np.linspace(0, frame_count - 1, num=frames_per_episode, dtype=np.int64).tolist()


def load_saved_frames(saved_obs_dir, legacy_saved_rgb_swap: bool) -> list[dict]:
    """Load saved camera images in numeric step order as Pillow RGB arrays."""
    saved_obs_dir = Path(saved_obs_dir)
    step_dirs = []
    for path in saved_obs_dir.iterdir():
        match = STEP_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            step_dirs.append((int(match.group(1)), path))
    frames = []
    for step_index, step_dir in sorted(step_dirs):
        for camera in CAMERAS:
            image_path = step_dir / f"{camera}_rgb.jpg"
            if not image_path.is_file():
                continue
            with Image.open(image_path) as image:
                image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if legacy_saved_rgb_swap:
                image_rgb = image_rgb[..., [2, 1, 0]].copy()
            frames.append(
                {
                    "camera": camera,
                    "step_index": step_index,
                    "path": str(image_path.resolve()),
                    "image": image_rgb,
                }
            )
    return frames


def _mean_metrics(records: list[dict]) -> dict[str, float]:
    names = records[0]["metrics"]
    return {name: float(np.mean([record["metrics"][name] for record in records])) for name in names}


def _mean_histograms(records: list[dict]) -> dict[str, np.ndarray]:
    return {
        name: np.mean([record["histograms"][name] for record in records], axis=0)
        for name in HISTOGRAM_NAMES
    }


def _serializable_histograms(histograms: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {name: [float(value) for value in histogram] for name, histogram in histograms.items()}


def _metric_variability(episode_summaries: list[dict]) -> dict[str, dict[str, float]]:
    metric_names = episode_summaries[0]["metrics_mean"]
    output = {}
    for name in metric_names:
        values = np.asarray([episode["metrics_mean"][name] for episode in episode_summaries])
        output[name] = {
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
        }
    return output


def _leave_one_episode_out_distances(episode_histograms: list[dict]) -> dict[str, dict]:
    result = {}
    for name in HISTOGRAM_NAMES:
        values = []
        if len(episode_histograms) > 1:
            for index, histogram_set in enumerate(episode_histograms):
                others = [item[name] for other_index, item in enumerate(episode_histograms) if other_index != index]
                values.append(jensen_shannon_distance(histogram_set[name], np.mean(others, axis=0)))
        result[name] = {
            "values": [float(value) for value in values],
            "median": float(np.median(values)) if values else 0.0,
            "maximum": float(max(values)) if values else 0.0,
        }
    return result


def _frame_record(*, source: str, camera: str, image: np.ndarray, **identity) -> dict:
    return {
        "source": source,
        "camera": camera,
        **identity,
        "metrics": basic_image_metrics(image),
        "histograms": _rgb_histograms(image),
    }


def _csv_record(record: dict) -> dict:
    row = {
        "source": record["source"],
        "camera": record["camera"],
        "episode_id": record.get("episode_id", ""),
        "frame_index": record.get("frame_index", ""),
        "step_index": record.get("step_index", ""),
        "image_path": record.get("image_path", ""),
    }
    row.update(record["metrics"])
    return row


def analyze(
    reference_parquets,
    saved_obs_dir,
    legacy_saved_bgr: bool,
    hf_revision: str,
    frames_per_episode: int = 30,
) -> dict:
    """Analyze two cameras without importing or running the policy stack."""
    if frames_per_episode < 0:
        raise ValueError("frames_per_episode must be non-negative")
    reference_paths = [Path(path) for path in reference_parquets]
    if not reference_paths:
        raise ValueError("at least one reference parquet is required")

    reference_by_camera = {camera: [] for camera in CAMERAS}
    reference_by_episode = {camera: {} for camera in CAMERAS}
    episode_ids = []
    total_reference_frames = 0

    columns = [*IMAGE_COLUMNS.values(), "episode_index", "frame_index"]
    for parquet_path in reference_paths:
        table = pq.read_table(parquet_path, columns=columns)
        frame_count = table.num_rows
        if frame_count == 0:
            raise ValueError(f"reference parquet has no frames: {parquet_path}")
        total_reference_frames += frame_count
        episode_values = table.column("episode_index").to_pylist()
        unique_episode_ids = sorted(set(int(value) for value in episode_values))
        if len(unique_episode_ids) != 1:
            raise ValueError(f"expected one episode per parquet: {parquet_path}")
        episode_id = unique_episode_ids[0]
        if episode_id in episode_ids:
            raise ValueError(f"duplicate episode id {episode_id}")
        episode_ids.append(episode_id)
        selected_indices = _sample_indices(frame_count, frames_per_episode)
        frame_indices = table.column("frame_index").to_pylist()
        for camera in CAMERAS:
            cells = table.column(IMAGE_COLUMNS[camera])
            episode_records = []
            for row_index in selected_indices:
                image = _decode_rgb(cells[row_index].as_py(), parquet_path)
                record = _frame_record(
                    source="reference",
                    camera=camera,
                    image=image,
                    episode_id=episode_id,
                    frame_index=int(frame_indices[row_index]),
                    image_path=str(parquet_path.resolve()),
                )
                episode_records.append(record)
                reference_by_camera[camera].append(record)
            reference_by_episode[camera][episode_id] = episode_records

    saved_frames = load_saved_frames(saved_obs_dir, legacy_saved_rgb_swap=legacy_saved_bgr)
    deployment_by_camera = {camera: [] for camera in CAMERAS}
    for saved_frame in saved_frames:
        deployment_by_camera[saved_frame["camera"]].append(
            _frame_record(
                source="deployment",
                camera=saved_frame["camera"],
                image=saved_frame["image"],
                step_index=saved_frame["step_index"],
                image_path=saved_frame["path"],
            )
        )
    for camera in CAMERAS:
        if not reference_by_camera[camera]:
            raise ValueError(f"no reference frames found for {camera}")
        if not deployment_by_camera[camera]:
            raise ValueError(f"no deployment frames found for {camera}")

    camera_summaries = {}
    all_records = []
    for camera in CAMERAS:
        reference_records = reference_by_camera[camera]
        deployment_records = deployment_by_camera[camera]
        reference_histograms = _mean_histograms(reference_records)
        deployment_histograms = _mean_histograms(deployment_records)
        episode_summaries = []
        episode_histograms = []
        for episode_id in sorted(reference_by_episode[camera]):
            records = reference_by_episode[camera][episode_id]
            histograms = _mean_histograms(records)
            episode_histograms.append(histograms)
            episode_summaries.append(
                {
                    "episode_id": episode_id,
                    "sampled_frames": len(records),
                    "metrics_mean": _mean_metrics(records),
                    "histograms": _serializable_histograms(histograms),
                }
            )
        reference_variability = _metric_variability(episode_summaries)
        deployment_metrics = _mean_metrics(deployment_records)
        outside_range = [
            name
            for name, value in deployment_metrics.items()
            if value < reference_variability[name]["minimum"] or value > reference_variability[name]["maximum"]
        ]
        histogram_js = {
            name: jensen_shannon_distance(reference_histograms[name], deployment_histograms[name])
            for name in HISTOGRAM_NAMES
        }
        baseline_js = _leave_one_episode_out_distances(episode_histograms)
        histogram_outliers = [
            name
            for name in HISTOGRAM_NAMES
            if len(episode_summaries) > 1 and histogram_js[name] > baseline_js[name]["maximum"]
        ]
        camera_summaries[camera] = {
            "reference": {
                "sampled_frames": len(reference_records),
                "metrics_mean": _mean_metrics(reference_records),
                "histograms": _serializable_histograms(reference_histograms),
            },
            "deployment": {
                "frames": len(deployment_records),
                "metrics_mean": deployment_metrics,
                "histograms": _serializable_histograms(deployment_histograms),
            },
            "reference_episode_summaries": episode_summaries,
            "reference_episode_metric_range": reference_variability,
            "histogram_js_distance": {name: float(value) for name, value in histogram_js.items()},
            "reference_leave_one_episode_out_js": baseline_js,
            "descriptive_flags": {
                "deployment_metrics_outside_reference_episode_range": outside_range,
                "histogram_distances_above_episode_baseline": histogram_outliers,
            },
        }
        all_records.extend(reference_records)
        all_records.extend(deployment_records)

    unique_steps = sorted({frame["step_index"] for frame in saved_frames})
    sampled_counts = {len(reference_by_camera[camera]) for camera in CAMERAS}
    if len(sampled_counts) != 1:
        raise AssertionError("camera reference sample counts diverged")
    summary = {
        "hf_revision": hf_revision,
        "legacy_saved_rgb_swap": bool(legacy_saved_bgr),
        "histogram_bins": 64,
        "frames_per_episode_requested": frames_per_episode,
        "reference": {
            "parquet_paths": [str(path.resolve()) for path in reference_paths],
            "episode_ids": sorted(episode_ids),
            "episode_count": len(episode_ids),
            "total_frames": total_reference_frames,
            "sampled_frames_per_camera": sampled_counts.pop(),
        },
        "deployment": {
            "saved_obs_dir": str(Path(saved_obs_dir).resolve()),
            "step_indices": unique_steps,
            "total_steps": len(unique_steps),
            "frames_per_camera": {
                camera: len(deployment_by_camera[camera]) for camera in CAMERAS
            },
        },
        "cameras": camera_summaries,
        "limitations": [
            "The deployment report is descriptive and cannot establish policy-failure causality.",
            "Sparse saved frames cannot measure 30 Hz flicker or auto-exposure dynamics.",
        ],
        "_frame_metrics": [_csv_record(record) for record in all_records],
    }
    return summary


def write_report(summary: dict, output_dir) -> None:
    """Write a strict finite JSON summary and one row per sampled frame to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_summary = {key: value for key, value in summary.items() if not key.startswith("_")}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(json_summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

    rows = summary.get("_frame_metrics", [])
    if not rows:
        raise ValueError("summary has no frame metrics")
    with (output_dir / "frame_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-parquet", action="append", required=True, type=Path)
    parser.add_argument("--saved-obs-dir", required=True, type=Path)
    parser.add_argument("--legacy-saved-rgb-swap", action="store_true")
    parser.add_argument("--hf-revision", default="")
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=30,
        help="Deterministically sample this many frames per episode; 0 uses every frame.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = analyze(
        args.reference_parquet,
        args.saved_obs_dir,
        legacy_saved_bgr=args.legacy_saved_rgb_swap,
        hf_revision=args.hf_revision,
        frames_per_episode=args.frames_per_episode,
    )
    write_report(summary, args.output_dir)
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Wrote {args.output_dir / 'frame_metrics.csv'}")


if __name__ == "__main__":
    main()
