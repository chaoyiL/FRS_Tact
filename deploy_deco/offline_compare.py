"""Compare DECO predictions with recorded action trajectories from a deployment trial."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .policy import DECOPolicy


def _load_trial(path: Path):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    records = {}
    with (path / "steps.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[int(record["iter_idx"])] = record
    frames = []
    for item in manifest["image_batches"]:
        index = int(item["iter_idx"])
        if index in records:
            frames.append((index, path / item["directory"]))
    return manifest, records, frames


def _observation(directory: Path, state) -> dict:
    return {
        "observation.images.camera0": np.asarray(
            Image.open(directory / "observation.images.camera0.png").convert("RGB")
        ),
        "observation.images.camera1": np.asarray(
            Image.open(directory / "observation.images.camera1.png").convert("RGB")
        ),
        "observation.state": np.asarray(state, dtype=np.float32),
    }


def _reference_chunk(records, index: int, horizon: int) -> np.ndarray | None:
    rows = []
    for offset in range(horizon):
        record = records.get(index + offset)
        if record is None or record.get("raw_action") is None:
            return None
        action = np.asarray(record["raw_action"], dtype=np.float32)
        if action.shape != (1, 20):
            return None
        rows.append(action[0])
    return np.stack(rows)


def _cumulative_xyz(action: np.ndarray, start: int) -> np.ndarray:
    return np.cumsum(action[:, start : start + 3], axis=0) * 1000.0


def _path_length(action: np.ndarray, start: int, steps: int) -> float:
    return float(np.linalg.norm(action[:steps, start : start + 3], axis=1).sum() * 1000.0)


def _endpoint_error_mm(prediction: np.ndarray, reference: np.ndarray, start: int, steps: int) -> float:
    pred = prediction[:steps, start : start + 3].sum(axis=0)
    ref = reference[:steps, start : start + 3].sum(axis=0)
    return float(np.linalg.norm(pred - ref) * 1000.0)


def _plot_examples(examples, output: Path, policy_type: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), sharex=True)
    axis_names = ("x", "y", "z")
    colors = ("tab:red", "tab:green", "tab:blue")
    for row, example in enumerate(examples):
        steps = np.arange(1, 33)
        for column, (start, hand) in enumerate(((0, "Left"), (10, "Right"))):
            pred = _cumulative_xyz(example["prediction"], start)
            ref = _cumulative_xyz(example["reference"], start)
            axis = axes[row, column]
            for dimension, (name, color) in enumerate(zip(axis_names, colors)):
                axis.plot(steps, pred[:, dimension], color=color, label=f"DECO {name}")
                axis.plot(
                    steps,
                    ref[:, dimension],
                    color=color,
                    linestyle="--",
                    alpha=0.72,
                    label=f"{policy_type.upper()} {name}",
                )
            axis.axvline(8, color="black", linestyle=":", linewidth=1)
            axis.set_title(f"{example['label']} (iter {example['iter_idx']}) - {hand} TCP")
            axis.set_ylabel("Approx. cumulative delta (mm)")
            axis.grid(alpha=0.25)
        grip = axes[row, 2]
        grip.plot(steps, example["prediction"][:, 9], color="tab:orange", label="DECO left")
        grip.plot(steps, example["reference"][:, 9], color="tab:orange", linestyle="--", label=f"{policy_type.upper()} left")
        grip.plot(steps, example["prediction"][:, 19], color="tab:purple", label="DECO right")
        grip.plot(steps, example["reference"][:, 19], color="tab:purple", linestyle="--", label=f"{policy_type.upper()} right")
        grip.axvline(8, color="black", linestyle=":", linewidth=1)
        grip.set_title(f"{example['label']} (iter {example['iter_idx']}) - grippers")
        grip.set_ylabel("Absolute gripper command")
        grip.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Predicted / recorded step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6)
    grip_handles, grip_labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(grip_handles, grip_labels, loc="lower center", ncol=4)
    fig.suptitle(
        "DECO predicted 32-step chunks vs recorded RDP execution\n"
        "Solid = DECO, dashed = recorded RDP; dotted line = deployed 8-step replan boundary",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_overview(rows, output: Path, policy_type: str) -> None:
    times = np.asarray([row["time_s"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    for column, (hand, start) in enumerate((("Left", 0), ("Right", 10))):
        axis = axes[0, column]
        axis.plot(times, [row[f"deco_{hand.lower()}_path_mm"] for row in rows], label="DECO planned", linewidth=1.2)
        axis.plot(times, [row[f"ref_{hand.lower()}_path_mm"] for row in rows], label=f"Recorded {policy_type.upper()}", linewidth=1.0, alpha=0.8)
        axis.set_title(f"{hand}: next-8-step TCP path length")
        axis.set_ylabel("Path length (mm)")
        axis.grid(alpha=0.25)
        axis.legend()
        grip = axes[1, column]
        grip_index = start + 9
        grip.plot(times, [row["prediction"][7, grip_index] for row in rows], label="DECO step 8", linewidth=1.2)
        grip.plot(times, [row["reference"][7, grip_index] for row in rows], label=f"Recorded {policy_type.upper()} step 8", linewidth=1.0, alpha=0.8)
        grip.set_title(f"{hand}: gripper command at step 8")
        grip.set_ylabel("Absolute gripper command")
        grip.set_xlabel("Trial time (s)")
        grip.grid(alpha=0.25)
        grip.legend()
    fig.suptitle("DECO vs recorded RDP behavior over the full trial (saved image frames)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def compare(trial: Path, checkpoint: Path, output_dir: Path, seed: int = 1000) -> dict:
    manifest, records, frames = _load_trial(trial)
    policy_type = str(manifest.get("policy_type", "reference"))
    policy = DECOPolicy(checkpoint, device="cuda:0", verify_hash=True)
    first_index, first_directory = frames[0]
    first_observation = _observation(first_directory, records[first_index]["state"])
    for warmup_seed in range(5):
        policy.predict(first_observation, seed=warmup_seed)
    torch.cuda.synchronize()

    first_time = float(records[first_index]["recorded_at"])
    rows = []
    latencies = []
    for order, (index, directory) in enumerate(frames):
        reference = _reference_chunk(records, index, policy.action_horizon)
        if reference is None:
            continue
        observation = _observation(directory, records[index]["state"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = policy.predict(observation, seed=seed + order)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)
        rows.append(
            {
                "iter_idx": index,
                "time_s": float(records[index]["recorded_at"]) - first_time,
                "prediction": prediction,
                "reference": reference,
                "deco_left_path_mm": _path_length(prediction, 0, 8),
                "ref_left_path_mm": _path_length(reference, 0, 8),
                "deco_right_path_mm": _path_length(prediction, 10, 8),
                "ref_right_path_mm": _path_length(reference, 10, 8),
            }
        )

    representative = []
    for position, label in ((0, "start"), (len(rows) // 2, "middle"), (len(rows) - 1, "end")):
        representative.append({"label": label, **rows[position]})

    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "deco_vs_rdp_examples.png"
    overview_path = output_dir / "deco_vs_rdp_overview.png"
    _plot_examples(representative, examples_path, policy_type)
    _plot_overview(rows, overview_path, policy_type)

    left_errors = [_endpoint_error_mm(row["prediction"], row["reference"], 0, 8) for row in rows]
    right_errors = [_endpoint_error_mm(row["prediction"], row["reference"], 10, 8) for row in rows]
    summary = {
        "trial": str(trial),
        "trial_policy_type": policy_type,
        "trial_result": manifest.get("result_label"),
        "checkpoint": str(checkpoint),
        "compared_frames": len(rows),
        "horizon": policy.action_horizon,
        "executed_prefix_steps": 8,
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "next_8_step_endpoint_difference_mm": {
            "left_mean": float(np.mean(left_errors)),
            "left_median": float(np.median(left_errors)),
            "right_mean": float(np.mean(right_errors)),
            "right_median": float(np.median(right_errors)),
        },
        "next_8_step_path_length_mm": {
            "deco_left_mean": float(np.mean([row["deco_left_path_mm"] for row in rows])),
            "rdp_left_mean": float(np.mean([row["ref_left_path_mm"] for row in rows])),
            "deco_right_mean": float(np.mean([row["deco_right_path_mm"] for row in rows])),
            "rdp_right_mean": float(np.mean([row["ref_right_path_mm"] for row in rows])),
        },
        "plots": [str(examples_path), str(overview_path)],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    summary = compare(
        args.trial.expanduser().resolve(),
        args.checkpoint.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
