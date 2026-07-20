from __future__ import annotations

import pathlib
from collections import defaultdict

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pure_flow_decoder.utils.cache import CachedPairs
from pure_flow_decoder.utils.metrics import EvaluationResult
from pure_flow_decoder.utils.model import FlowSolver
from pure_flow_decoder.utils.model import SelfAttentionFlowDecoder
from pure_flow_decoder.utils.model import decode_actions


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


def _select_trajectory_positions(sample_mse: np.ndarray, count: int) -> list[int]:
    return _select_ranked_positions(sample_mse, count)


def _episode_mean_mse(result: EvaluationResult, pairs: CachedPairs) -> tuple[np.ndarray, np.ndarray]:
    episode_indices = pairs.arrays["episode_index"]
    grouped: dict[int, list[float]] = defaultdict(list)
    for cache_index, mse in zip(result.cache_indices, result.sample_mse):
        grouped[int(episode_indices[cache_index])].append(float(mse))

    episodes = np.asarray(sorted(grouped), dtype=np.int64)
    means = np.asarray([float(np.mean(grouped[int(episode)])) for episode in episodes], dtype=np.float32)
    return episodes, means


def _select_episode_indices(episode_means: np.ndarray, count: int) -> list[int]:
    return _select_ranked_positions(episode_means, count)


def _validation_episode_cache_indices(pairs: CachedPairs, episode_index: int) -> np.ndarray:
    val_indices = pairs.indices("val")
    episode_indices = pairs.arrays["episode_index"]
    dataset_indices = pairs.arrays["dataset_index"]
    mask = episode_indices[val_indices] == episode_index
    selected = val_indices[mask]
    return selected[np.argsort(dataset_indices[selected])]


def _decode_cache_indices(
    model: SelfAttentionFlowDecoder,
    pairs: CachedPairs,
    cache_indices: np.ndarray,
    *,
    num_steps: int,
    solver: FlowSolver = "euler",
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    if len(cache_indices) == 0:
        raise ValueError("cache_indices must not be empty.")

    targets: list[np.ndarray] = []
    decoded: list[np.ndarray] = []
    for start in range(0, len(cache_indices), batch_size):
        batch_indices = cache_indices[start : start + batch_size]
        x_base = jnp.asarray(pairs.arrays["x_base"][batch_indices])
        targets.append(np.asarray(pairs.arrays["target"][batch_indices], dtype=np.float32))
        decoded.append(
            np.asarray(decode_actions(model, x_base, num_steps=num_steps, solver=solver), dtype=np.float32)
        )
    return np.concatenate(targets, axis=0), np.concatenate(decoded, axis=0)


def _concatenate_action_horizons(actions: np.ndarray) -> np.ndarray:
    if actions.ndim != 3:
        raise ValueError(f"Expected actions with shape [N, T, A], got {actions.shape}.")
    return actions.reshape(-1, actions.shape[-1])


def write_evaluation_plots(
    *,
    output_dir: pathlib.Path,
    result: EvaluationResult,
    pairs: CachedPairs,
    model: SelfAttentionFlowDecoder,
    num_steps: int,
    solver: FlowSolver,
    num_trajectory_samples: int,
    num_episode_strips: int,
) -> list[pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []

    written.append(
        _plot_metric_histograms(
            output_dir / "metrics_histogram.png",
            result,
        )
    )
    written.append(
        _plot_metric_scatter(
            output_dir / "metrics_scatter.png",
            result,
        )
    )
    written.append(
        _plot_per_episode_mse(
            output_dir / "per_episode_mse.png",
            result,
            pairs,
        )
    )
    if num_trajectory_samples > 0:
        written.append(
            _plot_action_trajectories(
                path=output_dir / "action_trajectories.png",
                result=result,
                pairs=pairs,
                model=model,
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
                num_steps=num_steps,
                solver=solver,
                num_episodes=num_episode_strips,
            )
        )
    return written


def _plot_metric_histograms(path: pathlib.Path, result: EvaluationResult) -> pathlib.Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    metrics = [
        ("flow_loss", result.sample_flow_loss, "Flow loss"),
        ("mse", result.sample_mse, "MSE"),
        ("rmse", result.sample_rmse, "RMSE"),
        ("mae", result.sample_mae, "MAE"),
    ]
    for axis, (_, values, title) in zip(axes.flat, metrics):
        axis.hist(values, bins=min(30, max(5, len(values) // 3)), color="#4C72B0", edgecolor="white")
        axis.axvline(float(np.mean(values)), color="#C44E52", linestyle="--", linewidth=1.5, label="mean")
        axis.set_title(title)
        axis.set_xlabel("per-sample value")
        axis.set_ylabel("count")
        axis.legend()
    fig.suptitle("Validation metric distributions", fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_metric_scatter(path: pathlib.Path, result: EvaluationResult) -> pathlib.Path:
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    scatter = axis.scatter(
        result.sample_flow_loss,
        result.sample_mse,
        alpha=0.65,
        s=24,
        c=result.sample_mae,
        cmap="viridis",
    )
    fig.colorbar(scatter, ax=axis, label="MAE")
    axis.set_xlabel("flow loss (t=0.5)")
    axis.set_ylabel("reconstruction MSE")
    axis.set_title("Per-sample flow loss vs reconstruction error")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_per_episode_mse(
    path: pathlib.Path,
    result: EvaluationResult,
    pairs: CachedPairs,
) -> pathlib.Path:
    episode_indices = pairs.arrays["episode_index"]
    grouped: dict[int, list[float]] = defaultdict(list)
    for cache_index, mse in zip(result.cache_indices, result.sample_mse):
        grouped[int(episode_indices[cache_index])].append(float(mse))

    episodes = sorted(grouped)
    data = [grouped[episode] for episode in episodes]
    labels = [str(episode) for episode in episodes]

    fig, axis = plt.subplots(figsize=(max(8, len(episodes) * 0.45), 5), constrained_layout=True)
    if len(episodes) == 1:
        axis.boxplot(data, labels=labels)
    else:
        axis.boxplot(data, tick_labels=labels)
    axis.set_xlabel("episode index")
    axis.set_ylabel("reconstruction MSE")
    axis.set_title("Per-episode reconstruction error")
    if len(episodes) > 12:
        axis.tick_params(axis="x", rotation=45)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_action_trajectories(
    *,
    path: pathlib.Path,
    result: EvaluationResult,
    pairs: CachedPairs,
    model: SelfAttentionFlowDecoder,
    num_steps: int,
    solver: FlowSolver,
    num_samples: int,
) -> pathlib.Path:
    positions = _select_trajectory_positions(result.sample_mse, num_samples)
    if not positions:
        return path

    cache_indices = result.cache_indices[positions]
    x_base = jnp.asarray(pairs.arrays["x_base"][cache_indices])
    target = np.asarray(pairs.arrays["target"][cache_indices], dtype=np.float32)
    decoded = np.asarray(decode_actions(model, x_base, num_steps=num_steps, solver=solver), dtype=np.float32)
    action_horizon = target.shape[1]
    action_dim = target.shape[2]
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
            axis.plot(
                timesteps,
                target[row, :, dim],
                linestyle="-",
                linewidth=1.8,
                label=f"target dim {dim}",
            )
            axis.plot(
                timesteps,
                decoded[row, :, dim],
                linestyle="--",
                linewidth=1.8,
                label=f"decoded dim {dim}",
            )
        axis.set_xlabel("action horizon step")
        axis.set_ylabel("normalized action")
        axis.set_title(
            f"cache={cache_index} episode={int(episode_indices[cache_index])} "
            f"dataset={int(dataset_indices[cache_index])} mse={result.sample_mse[position]:.4f}"
        )
        axis.legend(loc="upper right", fontsize=8, ncol=2)

    fig.suptitle(
        f"Decoded vs target actions (best / median / worst samples, first {dims_to_plot} dims)",
        fontsize=13,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


_DIM_COLORS = ("#4C72B0", "#55A868", "#C44E52")


def _plot_episode_action_strips(
    *,
    path: pathlib.Path,
    result: EvaluationResult,
    pairs: CachedPairs,
    model: SelfAttentionFlowDecoder,
    num_steps: int,
    solver: FlowSolver,
    num_episodes: int,
) -> pathlib.Path:
    episodes, episode_means = _episode_mean_mse(result, pairs)
    selected_positions = _select_episode_indices(episode_means, num_episodes)
    if not selected_positions:
        return path

    selected_episodes = [int(episodes[position]) for position in selected_positions]
    action_horizon = int(pairs.manifest["action_horizon"])
    action_dim = int(pairs.manifest["action_dim"])
    dims_to_plot = min(3, action_dim)

    episode_series: list[tuple[int, float, np.ndarray, np.ndarray, int]] = []
    max_timesteps = 0
    for position, episode_index in zip(selected_positions, selected_episodes):
        cache_indices = _validation_episode_cache_indices(pairs, episode_index)
        target, decoded = _decode_cache_indices(
            model,
            pairs,
            cache_indices,
            num_steps=num_steps,
            solver=solver,
        )
        target_series = _concatenate_action_horizons(target)
        decoded_series = _concatenate_action_horizons(decoded)
        episode_series.append(
            (
                episode_index,
                float(episode_means[position]),
                target_series,
                decoded_series,
                len(cache_indices),
            )
        )
        max_timesteps = max(max_timesteps, len(target_series))

    fig_height = max(2.8, 2.4 * len(episode_series))
    fig_width = min(48.0, max(18.0, max_timesteps / 220.0))
    fig, axes = plt.subplots(
        len(episode_series),
        1,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        squeeze=False,
    )

    for row, (episode_index, mean_mse, target_series, decoded_series, sample_count) in enumerate(
        episode_series
    ):
        axis = axes[row, 0]
        timesteps = np.arange(len(target_series))
        for dim in range(dims_to_plot):
            color = _DIM_COLORS[dim]
            axis.plot(
                timesteps,
                target_series[:, dim],
                linestyle="-",
                linewidth=1.2,
                color=color,
                label=f"target dim {dim}",
            )
            axis.plot(
                timesteps,
                decoded_series[:, dim],
                linestyle="--",
                linewidth=1.2,
                color=color,
                label=f"decoded dim {dim}",
            )

        for sample_start in range(action_horizon, len(target_series), action_horizon):
            axis.axvline(sample_start, color="#BBBBBB", linewidth=0.6, alpha=0.8)

        axis.set_xlim(0, max(len(target_series) - 1, 1))
        axis.set_ylabel("normalized action")
        axis.set_title(
            f"episode={episode_index} samples={sample_count} "
            f"mean_mse={mean_mse:.4f} timesteps={len(target_series)}"
        )
        if row == len(episode_series) - 1:
            axis.set_xlabel("concatenated action horizon steps")
        if row == 0:
            axis.legend(loc="upper right", fontsize=8, ncol=3)

    fig.suptitle(
        f"Episode action strips (best / median / worst episodes, first {dims_to_plot} dims)",
        fontsize=13,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training_history(
    history_path: pathlib.Path,
    *,
    output_path: pathlib.Path | None = None,
) -> pathlib.Path:
    """Plot train/val flow loss and val MSE from a training history CSV."""
    import csv

    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found: {history_path}")

    epochs: list[int] = []
    train_flow_loss: list[float] = []
    val_flow_loss: list[float] = []
    val_mse: list[float] = []
    with history_path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            epochs.append(int(row["epoch"]))
            train_flow_loss.append(float(row["train_flow_loss"]))
            val_flow_loss.append(float(row["val_flow_loss"]))
            val_mse.append(float(row["val_mse"]))

    if not epochs:
        raise ValueError(f"No training history rows found in {history_path}.")

    destination = output_path or history_path.with_name("training_curves.png")
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot(epochs, train_flow_loss, label="train_flow_loss", linewidth=2.0, color="#4C72B0")
    axis.plot(epochs, val_flow_loss, label="val_flow_loss", linewidth=2.0, color="#55A868")
    axis.plot(epochs, val_mse, label="val_mse", linewidth=2.0, color="#C44E52")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss / MSE")
    axis.set_title("Training curves")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination
