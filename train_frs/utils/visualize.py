from __future__ import annotations

import pathlib
from collections import defaultdict

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_frs.utils.data import TactileConditionedBatches
from train_frs.utils.metrics import EvaluationResult
from train_frs.utils.model import FlowSolver
from train_frs.utils.model import TactileConditionedFlowDecoder
from train_frs.utils.model import decode_actions
from utils.cache import CachedPairs


def _select_ranked_positions(values: np.ndarray, count: int) -> list[int]:
    if count <= 0:
        return []
    order = np.argsort(values)
    positions = [int(order[0]), int(order[-1])]
    if count > 2:
        mid_positions = np.linspace(0, len(order) - 1, count - 2, dtype=int)
        positions.extend(int(order[index]) for index in mid_positions)
    unique_positions = sorted(set(positions))
    return unique_positions[:count]


def write_evaluation_plots(
    *,
    output_dir: pathlib.Path,
    result: EvaluationResult,
    pairs: CachedPairs,
    model: TactileConditionedFlowDecoder,
    conditioner: TactileConditionedBatches,
    num_steps: int,
    solver: FlowSolver,
    num_trajectory_samples: int,
    num_episode_strips: int,
) -> list[pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    written.append(_plot_metric_histograms(output_dir / "metrics_histogram.png", result))
    written.append(_plot_metric_scatter(output_dir / "metrics_scatter.png", result))
    written.append(_plot_per_episode_mse(output_dir / "per_episode_mse.png", result, pairs))
    if num_trajectory_samples > 0:
        written.append(
            _plot_action_trajectories(
                path=output_dir / "action_trajectories.png",
                result=result,
                pairs=pairs,
                model=model,
                conditioner=conditioner,
                num_steps=num_steps,
                solver=solver,
                num_samples=num_trajectory_samples,
            )
        )
    if num_episode_strips > 0:
        written.append(
            _plot_episode_action_strips(
                path=output_dir / "episode_action_strips.png",
                result=result,
                pairs=pairs,
                model=model,
                conditioner=conditioner,
                num_steps=num_steps,
                solver=solver,
                num_episodes=num_episode_strips,
            )
        )
    return written


def _plot_metric_histograms(path: pathlib.Path, result: EvaluationResult) -> pathlib.Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    bins = min(30, max(5, len(result.sample_mse) // 3))
    metrics = [
        ("flow_loss", result.sample_flow_loss, f"Flow loss (primary={result.target})", "#4C72B0"),
        ("mse_gt", result.sample_mse_gt, "MSE vs GT", "#C44E52"),
        ("mse_pred", result.sample_mse_pred, "MSE vs predicted", "#4C72B0"),
        ("mae", result.sample_mae, f"MAE (primary={result.target})", "#DD8452"),
    ]
    for axis, (_, values, title, color) in zip(axes.flat, metrics):
        axis.hist(values, bins=bins, color=color, edgecolor="white")
        axis.axvline(float(np.mean(values)), color="#333333", linestyle="--", linewidth=1.5, label="mean")
        axis.set_title(title)
        axis.set_xlabel("per-sample value")
        axis.set_ylabel("count")
        axis.legend()
    fig.suptitle(
        f"Validation metric distributions (primary target={result.target})",
        fontsize=14,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_metric_scatter(path: pathlib.Path, result: EvaluationResult) -> pathlib.Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].scatter(
        result.sample_mse_gt,
        result.sample_mse_pred,
        alpha=0.65,
        s=24,
        c=result.sample_flow_loss,
        cmap="viridis",
    )
    lo = float(min(result.sample_mse_gt.min(), result.sample_mse_pred.min()))
    hi = float(max(result.sample_mse_gt.max(), result.sample_mse_pred.max()))
    axes[0].plot([lo, hi], [lo, hi], color="#888888", linestyle="--", linewidth=1.2, label="y=x")
    axes[0].set_xlabel("MSE vs GT")
    axes[0].set_ylabel("MSE vs predicted")
    axes[0].set_title("Per-sample MSE: GT vs predicted")
    axes[0].legend(loc="upper left")

    scatter = axes[1].scatter(
        result.sample_flow_loss,
        result.sample_mse,
        alpha=0.65,
        s=24,
        c=result.sample_mae,
        cmap="viridis",
    )
    fig.colorbar(scatter, ax=axes[1], label=f"MAE (primary={result.target})")
    axes[1].set_xlabel("flow loss (t=0.5, primary target)")
    axes[1].set_ylabel(f"reconstruction MSE (primary={result.target})")
    axes[1].set_title("Flow loss vs primary reconstruction error")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_per_episode_mse(
    path: pathlib.Path,
    result: EvaluationResult,
    pairs: CachedPairs,
) -> pathlib.Path:
    episode_indices = pairs.arrays["episode_index"]
    grouped_gt: dict[int, list[float]] = defaultdict(list)
    grouped_pred: dict[int, list[float]] = defaultdict(list)
    for cache_index, mse_gt, mse_pred in zip(
        result.cache_indices, result.sample_mse_gt, result.sample_mse_pred
    ):
        episode = int(episode_indices[cache_index])
        grouped_gt[episode].append(float(mse_gt))
        grouped_pred[episode].append(float(mse_pred))

    episodes = sorted(grouped_gt)
    data_gt = [grouped_gt[episode] for episode in episodes]
    data_pred = [grouped_pred[episode] for episode in episodes]
    labels = [str(episode) for episode in episodes]
    positions = np.arange(1, len(episodes) + 1, dtype=np.float64)

    fig, axis = plt.subplots(figsize=(max(8, len(episodes) * 0.55), 5), constrained_layout=True)
    width = 0.35
    bp_gt = axis.boxplot(
        data_gt,
        positions=positions - width / 2,
        widths=width,
        patch_artist=True,
        manage_ticks=False,
    )
    bp_pred = axis.boxplot(
        data_pred,
        positions=positions + width / 2,
        widths=width,
        patch_artist=True,
        manage_ticks=False,
    )
    for box in bp_gt["boxes"]:
        box.set_facecolor("#C44E52")
        box.set_alpha(0.55)
    for box in bp_pred["boxes"]:
        box.set_facecolor("#4C72B0")
        box.set_alpha(0.55)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_xlabel("episode index")
    axis.set_ylabel("reconstruction MSE")
    axis.set_title("Per-episode reconstruction error (gt vs predicted)")
    axis.legend(
        [bp_gt["boxes"][0], bp_pred["boxes"][0]],
        ["mse_gt", "mse_pred"],
        loc="upper right",
    )
    if len(episodes) > 12:
        axis.tick_params(axis="x", rotation=45)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _decode_with_tactile(
    model: TactileConditionedFlowDecoder,
    pairs: CachedPairs,
    conditioner: TactileConditionedBatches,
    cache_indices: np.ndarray,
    gate_weights: np.ndarray | None = None,
    *,
    num_steps: int,
    solver: FlowSolver,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_base = jnp.asarray(pairs.arrays["x_base"][cache_indices])
    gt_action = np.asarray(pairs.arrays["gt_action"][cache_indices], dtype=np.float32)
    predicted_action = np.asarray(pairs.arrays["target"][cache_indices], dtype=np.float32)
    tactile_seq = conditioner.encode_cache_indices(cache_indices)
    decoded = np.asarray(
        decode_actions(
            model,
            x_base,
            tactile_seq,
            None if gate_weights is None else jnp.asarray(gate_weights),
            num_steps=num_steps,
            solver=solver,
        ),
        dtype=np.float32,
    )
    return gt_action, predicted_action, decoded


def _plot_action_trajectories(
    *,
    path: pathlib.Path,
    result: EvaluationResult,
    pairs: CachedPairs,
    model: TactileConditionedFlowDecoder,
    conditioner: TactileConditionedBatches,
    num_steps: int,
    solver: FlowSolver,
    num_samples: int,
) -> pathlib.Path:
    positions = _select_ranked_positions(result.sample_mse, num_samples)
    if not positions:
        return path

    cache_indices = result.cache_indices[positions]
    selected_gate = (
        None if result.sample_gate_w is None else result.sample_gate_w[positions]
    )
    gt_action, predicted_action, decoded = _decode_with_tactile(
        model,
        pairs,
        conditioner,
        cache_indices,
        selected_gate,
        num_steps=num_steps,
        solver=solver,
    )
    action_horizon = gt_action.shape[1]
    action_dim = gt_action.shape[2]
    dims_to_plot = min(3, action_dim)
    timesteps = np.arange(action_horizon)

    fig, axes = plt.subplots(
        len(positions),
        1,
        figsize=(10, 3.2 * len(positions)),
        constrained_layout=True,
        squeeze=False,
    )
    episode_indices = pairs.arrays["episode_index"]
    dataset_indices = pairs.arrays["dataset_index"]

    for row, position in enumerate(positions):
        axis = axes[row, 0]
        cache_index = int(result.cache_indices[position])
        for dim in range(dims_to_plot):
            color = _DIM_COLORS[dim % len(_DIM_COLORS)]
            axis.plot(
                timesteps,
                gt_action[row, :, dim],
                color=color,
                linestyle="-",
                linewidth=1.8,
                label=f"gt dim {dim}",
            )
            axis.plot(
                timesteps,
                predicted_action[row, :, dim],
                color=color,
                linestyle=":",
                linewidth=1.5,
                label=f"pred dim {dim}",
            )
            axis.plot(
                timesteps,
                decoded[row, :, dim],
                color=color,
                linestyle="--",
                linewidth=1.8,
                label=f"decoded dim {dim}",
            )
        axis.set_xlabel("action horizon step")
        axis.set_ylabel("normalized action")
        axis.set_title(
            f"cache={cache_index} episode={int(episode_indices[cache_index])} "
            f"dataset={int(dataset_indices[cache_index])} "
            f"mse_gt={result.sample_mse_gt[position]:.4f} "
            f"mse_pred={result.sample_mse_pred[position]:.4f}"
        )
        axis.legend(loc="upper right", fontsize=8, ncol=3)

    fig.suptitle(
        f"Decoded vs GT / predicted (ranked by primary={result.target}, first {dims_to_plot} dims)",
        fontsize=13,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


_DIM_COLORS = ("#4C72B0", "#55A868", "#C44E52")


def _validation_episode_cache_indices(pairs: CachedPairs, episode_index: int) -> np.ndarray:
    val_indices = pairs.indices("val")
    episode_indices = pairs.arrays["episode_index"]
    dataset_indices = pairs.arrays["dataset_index"]
    mask = episode_indices[val_indices] == episode_index
    selected = val_indices[mask]
    return selected[np.argsort(dataset_indices[selected])]


def _plot_episode_action_strips(
    *,
    path: pathlib.Path,
    result: EvaluationResult,
    pairs: CachedPairs,
    model: TactileConditionedFlowDecoder,
    conditioner: TactileConditionedBatches,
    num_steps: int,
    solver: FlowSolver,
    num_episodes: int,
) -> pathlib.Path:
    episode_indices = pairs.arrays["episode_index"]
    grouped: dict[int, list[float]] = defaultdict(list)
    for cache_index, mse in zip(result.cache_indices, result.sample_mse):
        grouped[int(episode_indices[cache_index])].append(float(mse))
    if not grouped:
        return path

    episodes = np.asarray(sorted(grouped), dtype=np.int64)
    means = np.asarray([float(np.mean(grouped[int(episode)])) for episode in episodes], dtype=np.float32)
    selected_positions = _select_ranked_positions(means, num_episodes)
    if not selected_positions:
        return path
    selected_episodes = [int(episodes[position]) for position in selected_positions]

    fig, axes = plt.subplots(
        len(selected_episodes),
        1,
        figsize=(12, 3.0 * len(selected_episodes)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, episode_index in enumerate(selected_episodes):
        axis = axes[row, 0]
        cache_indices = _validation_episode_cache_indices(pairs, episode_index)
        if result.sample_gate_w is None:
            selected_gate = None
        else:
            gate_by_index = {
                int(index): float(weight)
                for index, weight in zip(
                    result.cache_indices, result.sample_gate_w, strict=True
                )
            }
            selected_gate = np.asarray(
                [gate_by_index[int(index)] for index in cache_indices], dtype=np.float32
            )
        gt_action, predicted_action, decoded = _decode_with_tactile(
            model,
            pairs,
            conditioner,
            cache_indices,
            selected_gate,
            num_steps=num_steps,
            solver=solver,
        )
        concatenated_gt = gt_action.reshape(-1, gt_action.shape[-1])
        concatenated_pred = predicted_action.reshape(-1, predicted_action.shape[-1])
        concatenated_decoded = decoded.reshape(-1, decoded.shape[-1])
        dims_to_plot = min(3, concatenated_gt.shape[-1])
        timesteps = np.arange(concatenated_gt.shape[0])
        for dim in range(dims_to_plot):
            color = _DIM_COLORS[dim % len(_DIM_COLORS)]
            axis.plot(
                timesteps,
                concatenated_gt[:, dim],
                color=color,
                linewidth=1.6,
                label=f"gt dim {dim}",
            )
            axis.plot(
                timesteps,
                concatenated_pred[:, dim],
                color=color,
                linewidth=1.4,
                linestyle=":",
                label=f"pred dim {dim}",
            )
            axis.plot(
                timesteps,
                concatenated_decoded[:, dim],
                color=color,
                linewidth=1.6,
                linestyle="--",
                label=f"decoded dim {dim}",
            )
        axis.set_xlabel("concatenated action steps")
        axis.set_ylabel("normalized action")
        axis.set_title(
            f"episode={episode_index} mean_primary_mse={means[selected_positions[row]]:.4f}"
        )
        axis.legend(loc="upper right", fontsize=8, ncol=3)

    fig.suptitle(
        f"Episode action strips vs GT / predicted (ranked by primary={result.target})",
        fontsize=13,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# Re-export for train.py callers.
from train_frs.utils.history_plot import plot_training_history as plot_training_history
