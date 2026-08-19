#!/usr/bin/env python
"""Sample cache pairs and plot predicted-vs-GT action error.

Top: per-sample MSE (sorted ascending for readability).
Bottom: per-sample × per-dimension mean |pred-gt| heatmap
        (abs error averaged over the action horizon).

Example:
  uv run python tools/plot_pred_gt_error.py \\
    --cache-dir cache/tactile_test_05 \\
    --split all --num-samples 512 --seed 0 \\
    --output outputs/pred_gt_error.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.cache import CachedPairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "val", "all"),
        default="val",
        help="Which split to draw samples from (default: val).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="Number of samples to plot; <=0 means use the whole split.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <cache-dir>/pred_gt_error.png).",
    )
    return parser.parse_args()


def select_indices(pairs: CachedPairs, split: str, num_samples: int, seed: int) -> np.ndarray:
    if split == "all":
        indices = np.arange(int(pairs.manifest["sample_count"]), dtype=np.int64)
    else:
        indices = pairs.indices(split)  # type: ignore[arg-type]
    if indices.size == 0:
        raise ValueError(f"No samples found for split={split!r}.")
    if num_samples > 0 and num_samples < indices.size:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=num_samples, replace=False))
    return np.asarray(indices, dtype=np.int64)


def compute_errors(
    pred: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sample_mse [N], dim_mae [N, A])."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    if pred.ndim != 3:
        raise ValueError(f"Expected actions [N, H, A], got {pred.shape}")
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    sample_mse = np.mean(np.square(diff), axis=(1, 2))
    # Mean absolute deviation over horizon → one value per (sample, dim).
    dim_mae = np.mean(np.abs(diff), axis=1)
    return sample_mse.astype(np.float32), dim_mae.astype(np.float32)


def plot_errors(
    sample_mse: np.ndarray,
    dim_mae: np.ndarray,
    *,
    cache_indices: np.ndarray,
    split: str,
    output: Path,
) -> Path:
    order = np.argsort(sample_mse)
    mse_sorted = sample_mse[order]
    dim_mae_sorted = dim_mae[order]
    xs = np.arange(mse_sorted.shape[0])
    action_dim = dim_mae_sorted.shape[1]
    mean_mse = float(np.mean(sample_mse))
    median_mse = float(np.median(sample_mse))
    p90 = float(np.quantile(sample_mse, 0.9))
    p99 = float(np.quantile(sample_mse, 0.99))
    dim_means = np.mean(dim_mae, axis=0)
    top_dims = np.argsort(dim_means)[::-1][:5]
    top_summary = ", ".join(f"d{int(d)}={dim_means[d]:.3f}" for d in top_dims)

    fig = plt.figure(figsize=(12, 10.2))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.95, 2.4, 3.3],
        hspace=0.55,
        left=0.08,
        right=0.90,
        top=0.97,
        bottom=0.07,
    )
    ax_info = fig.add_subplot(grid[0, 0])
    ax0 = fig.add_subplot(grid[1, 0])
    ax1 = fig.add_subplot(grid[2, 0], sharex=ax0)

    ax_info.set_axis_off()
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    # Three evenly spaced lines inside a dedicated header band (no collision with axes titles).
    ax_info.text(
        0.5,
        0.86,
        f"Predicted vs GT action gap  |  cache samples={cache_indices.size}",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        transform=ax_info.transAxes,
        clip_on=False,
    )
    ax_info.text(
        0.5,
        0.48,
        (
            f"split={split}   N={len(sample_mse)}   "
            f"mean={mean_mse:.4f}   median={median_mse:.4f}   "
            f"p90={p90:.3f}   p99={p99:.3f}"
        ),
        ha="center",
        va="center",
        fontsize=10,
        transform=ax_info.transAxes,
        clip_on=False,
    )
    ax_info.text(
        0.5,
        0.12,
        f"top dims by mean |Δ|: {top_summary}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
        transform=ax_info.transAxes,
        clip_on=False,
    )

    ax0.plot(xs, mse_sorted, color="#4C72B0", linewidth=1.4, label="sample MSE")
    ax0.fill_between(xs, 0.0, mse_sorted, color="#4C72B0", alpha=0.25)
    ax0.axhline(mean_mse, color="#C44E52", linestyle="--", linewidth=1.3, label=f"mean={mean_mse:.4f}")
    ax0.axhline(
        median_mse, color="#DD8452", linestyle=":", linewidth=1.3, label=f"median={median_mse:.4f}"
    )
    ax0.set_ylabel("MSE(pred, gt)")
    ax0.set_title("Per-sample pred→gt MSE (sorted ascending)", fontsize=11, pad=10)
    ax0.set_xlim(0, max(len(sample_mse) - 1, 1))
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left", fontsize=9, framealpha=0.92)
    plt.setp(ax0.get_xticklabels(), visible=False)

    vmax = float(np.quantile(dim_mae_sorted, 0.99)) if dim_mae_sorted.size else 1.0
    vmax = max(vmax, 1e-6)
    image = ax1.imshow(
        dim_mae_sorted.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        extent=(-0.5, dim_mae_sorted.shape[0] - 0.5, -0.5, action_dim - 0.5),
    )
    ax1.set_xlabel("samples (sorted by MSE, same order as top)")
    ax1.set_ylabel("action dim")
    ax1.set_yticks(np.arange(action_dim))
    ax1.tick_params(axis="y", labelsize=8)
    ax1.set_title(
        f"Per-sample × per-dim mean |pred−gt| over horizon  (vmax=p99={vmax:.3f})",
        fontsize=11,
        pad=10,
    )
    cbar = fig.colorbar(image, ax=ax1, fraction=0.035, pad=0.02)
    cbar.set_label("mean |Δ|")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    pairs = CachedPairs(args.cache_dir)
    indices = select_indices(pairs, args.split, args.num_samples, args.seed)
    pred = np.asarray(pairs.arrays["target"][indices], dtype=np.float32)
    gt = np.asarray(pairs.arrays["gt_action"][indices], dtype=np.float32)
    sample_mse, dim_mae = compute_errors(pred, gt)

    output = args.output or (args.cache_dir / "pred_gt_error.png")
    written = plot_errors(
        sample_mse,
        dim_mae,
        cache_indices=indices,
        split=args.split,
        output=output,
    )

    print(f"cache={args.cache_dir.resolve()}")
    print(f"split={args.split} plotted={len(indices)}/{int(pairs.manifest['sample_count'])}")
    print(f"action_shape={tuple(pred.shape)}  # [N, horizon, action_dim]")
    print(
        f"mse: mean={float(np.mean(sample_mse)):.6f} "
        f"median={float(np.median(sample_mse)):.6f} "
        f"std={float(np.std(sample_mse)):.6f} "
        f"min={float(np.min(sample_mse)):.6f} "
        f"max={float(np.max(sample_mse)):.6f}"
    )
    dim_means = np.mean(dim_mae, axis=0)
    print("per-dim mean |Δ|:", np.array2string(dim_means, precision=4, separator=", "))
    print(f"saved={written.resolve()}")


if __name__ == "__main__":
    main()
