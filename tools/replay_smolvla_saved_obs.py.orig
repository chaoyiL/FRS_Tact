#!/usr/bin/env python3
"""Counterfactual, offline-only SmolVLA replay from saved observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.analyze_smolvla_online_run import load_chunk_trace, load_saved_observations, reconstruct_state
from tools.eval_smolvla_pytorch_offline import _predict_chunk, frame_seed, load_eval_runtime


ACTION_SHAPE = (20, 20)
METRIC_DIMS = (2, 9, 19)
GRIPPER_DIMS = (9, 19)


def _sorted_mapping(observations: list[Any], chunks: list[Any]) -> list[tuple[Any, Any]]:
    """Return the required step/sequence pairing, rejecting ambiguous artifacts."""
    ordered_observations = sorted(observations, key=lambda observation: int(observation.step))
    ordered_chunks = sorted(chunks, key=lambda chunk: int(chunk.obs_seq))
    if len(ordered_observations) != len(ordered_chunks):
        raise ValueError("saved observations and action chunks must have the same count")
    if len({int(observation.step) for observation in ordered_observations}) != len(ordered_observations):
        raise ValueError("saved observations contain duplicate steps")
    if len({int(chunk.obs_seq) for chunk in ordered_chunks}) != len(ordered_chunks):
        raise ValueError("chunk trace contains duplicate obs_seq values")
    pairs: list[tuple[Any, Any]] = []
    for observation, chunk in zip(ordered_observations, ordered_chunks, strict=True):
        step = int(observation.step)
        expected_sequence = step // 10 + 1
        if step < 0 or step % 10 or int(chunk.obs_seq) != expected_sequence:
            raise ValueError(
                "saved observations must map step 0 to obs_seq 1, step 10 to obs_seq 2, and so on"
            )
        pairs.append((observation, chunk))
    return pairs


def _action_array(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != ACTION_SHAPE or not np.isfinite(array).all():
        raise ValueError(f"{label} must have finite shape {ACTION_SHAPE}, got {array.shape}")
    return array


def _dimension_metrics(replay: np.ndarray, live: np.ndarray, dimension: int) -> dict[str, float | int]:
    replay_dimension = replay[..., dimension]
    live_dimension = live[..., dimension]
    metrics: dict[str, float | int] = {
        "replay_min": float(replay_dimension.min()),
        "live_min": float(live_dimension.min()),
        "mae": float(np.mean(np.abs(replay_dimension - live_dimension))),
    }
    if dimension in GRIPPER_DIMS:
        metrics["replay_le_0_09_count"] = int(np.count_nonzero(replay_dimension <= 0.09))
        metrics["live_le_0_09_count"] = int(np.count_nonzero(live_dimension <= 0.09))
    return metrics


def _metrics(replay: np.ndarray, live: np.ndarray) -> dict[str, dict[str, float | int]]:
    return {f"dim_{dimension}": _dimension_metrics(replay, live, dimension) for dimension in METRIC_DIMS}


def run_saved_obs_replay(
    *,
    config_path: Path | str,
    obs_dir: Path | str,
    trace_dir: Path | str,
    output_dir: Path | str,
    device: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Replay saved frames through the local runtime without starting any service."""
    runtime = load_eval_runtime(config_path, device=device)
    observations = load_saved_observations(Path(obs_dir))
    chunks = load_chunk_trace(Path(trace_dir) / "chunk_trace.jsonl")
    pairs = _sorted_mapping(observations, chunks)
    reference = pairs[0][0]
    if hasattr(runtime.policy, "reset"):
        runtime.policy.reset()

    steps: list[int] = []
    sequences: list[int] = []
    states: list[np.ndarray] = []
    replay_actions: list[np.ndarray] = []
    live_actions: list[np.ndarray] = []
    per_step: list[dict[str, Any]] = []
    for observation, chunk in pairs:
        step = int(observation.step)
        state = np.asarray(reconstruct_state(observation, reference), dtype=np.float32)
        if state.shape != (20,) or not np.isfinite(state).all():
            raise ValueError(f"reconstructed state for saved step {step} must have finite shape (20,)")
        runtime.torch.manual_seed(frame_seed(seed, episode_index=0, frame_index=step))
        replay = _action_array(
            _predict_chunk(
                runtime,
                {
                    "observation.state": state,
                    "observation.images.camera0": observation.camera0_rgb,
                    "observation.images.camera1": observation.camera1_rgb,
                },
            ),
            label=f"replay action for saved step {step}",
        )
        live = _action_array(chunk.raw_actions, label=f"live action for obs_seq {chunk.obs_seq}")
        steps.append(step)
        sequences.append(int(chunk.obs_seq))
        states.append(state)
        replay_actions.append(replay)
        live_actions.append(live)
        per_step.append({"step": step, "obs_seq": int(chunk.obs_seq), "metrics": _metrics(replay, live)})

    replay_array = np.stack(replay_actions)
    live_array = np.stack(live_actions)
    state_array = np.stack(states)
    summary: dict[str, Any] = {
        "counterfactual": True,
        "state_reference": "saved_step_0_approximation",
        "full_array_inequality_note": (
            "Full-array inequality is not runtime mismatch proof because historical warmup reference and RNG are missing."
        ),
        "counts": {"saved_observations": len(pairs), "action_chunks": len(pairs)},
        "per_step": per_step,
        "overall": _metrics(replay_array, live_array),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "replay_predictions.npz",
        steps=np.asarray(steps, dtype=np.int64),
        obs_seq=np.asarray(sequences, dtype=np.int64),
        replay_actions=replay_array,
        live_actions=live_array,
        approximate_states=state_array,
    )
    (output / "replay_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary | {"steps": steps, "obs_seq": sequences}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline counterfactual replay of saved SmolVLA observations.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--obs-dir", required=True, type=Path)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", default=0, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        run_saved_obs_replay(
            config_path=args.config,
            obs_dir=args.obs_dir,
            trace_dir=args.trace_dir,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
        )
    except (RuntimeError, ValueError) as exc:
        _build_arg_parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
