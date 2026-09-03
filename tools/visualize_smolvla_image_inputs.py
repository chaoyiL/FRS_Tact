#!/usr/bin/env python3
"""Visualize raw SmolVLA training and saved deployment RGB inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.analyze_smolvla_online_run import load_saved_observations, load_training_parquets


RGB_COLORS = ("#d62728", "#2ca02c", "#1f77b4")
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def select_even_indices(length: int, count: int) -> np.ndarray:
    """Return evenly spaced indices, including both endpoints."""
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    return np.rint(np.linspace(0, length - 1, min(length, count))).astype(int)


def _normalized_histograms(camera_arrays: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rgb_counts = np.zeros((3, 256), dtype=np.float64)
    luminance_counts = np.zeros(256, dtype=np.float64)
    for images in camera_arrays:
        for start in range(0, len(images), 16):
            batch = np.asarray(images[start : start + 16], dtype=np.uint8)
            for channel in range(3):
                rgb_counts[channel] += np.bincount(batch[..., channel].ravel(), minlength=256)
            luminance = np.rint(batch.astype(np.float32) @ LUMINANCE_WEIGHTS)
            luminance_counts += np.bincount(
                np.clip(luminance, 0, 255).astype(np.uint8).ravel(),
                minlength=256,
            )
    rgb_counts /= rgb_counts.sum(axis=1, keepdims=True)
    luminance_counts /= luminance_counts.sum()
    return rgb_counts, luminance_counts


def _show_image_row(
    axes: np.ndarray,
    images: np.ndarray,
    indices: np.ndarray,
    labels: list[str],
    row_label: str,
) -> None:
    for column, axis in enumerate(axes):
        axis.set_xticks([])
        axis.set_yticks([])
        if column >= len(indices):
            axis.axis("off")
            continue
        image = images[indices[column]]
        axis.imshow(image)
        axis.set_title(
            f"{labels[column]}\nmean Y={float((image @ LUMINANCE_WEIGHTS).mean()):.1f}",
            fontsize=8,
        )
        if column == 0:
            axis.set_ylabel(row_label, fontsize=10, fontweight="bold")


def write_image_input_comparison(
    training_root: Path | str,
    obs_dir: Path | str,
    output: Path | str,
    sample_count: int = 6,
) -> Path:
    """Write a four-row RGB sample grid and full-source histogram comparison."""
    training = load_training_parquets(Path(training_root))
    saved = load_saved_observations(Path(obs_dir))
    if not saved:
        raise ValueError("saved observation directory is empty")

    train_indices = select_even_indices(len(training.camera0_rgb), sample_count)
    deploy_indices = select_even_indices(len(saved), sample_count)
    deploy_camera0 = np.stack([item.camera0_rgb for item in saved])
    deploy_camera1 = np.stack([item.camera1_rgb for item in saved])

    figure = plt.figure(figsize=(max(12, 3 * sample_count), 10), constrained_layout=True)
    grid = figure.add_gridspec(5, sample_count, height_ratios=(1, 1, 1, 1, 1.25))
    image_axes = np.asarray(
        [[figure.add_subplot(grid[row, column]) for column in range(sample_count)] for row in range(4)]
    )

    train_labels = [
        f"ep {int(training.episode_indices[index])}, frame {int(training.frame_indices[index])}"
        for index in train_indices
    ]
    deploy_labels = [f"saved step {int(saved[index].step)}" for index in deploy_indices]
    _show_image_row(
        image_axes[0],
        training.camera0_rgb,
        train_indices,
        train_labels,
        f"Training camera0\n{training.camera0_rgb.shape[1]}x{training.camera0_rgb.shape[2]}",
    )
    _show_image_row(
        image_axes[1],
        training.camera1_rgb,
        train_indices,
        train_labels,
        f"Training camera1\n{training.camera1_rgb.shape[1]}x{training.camera1_rgb.shape[2]}",
    )
    _show_image_row(
        image_axes[2],
        deploy_camera0,
        deploy_indices,
        deploy_labels,
        f"Deployment camera0\n{deploy_camera0.shape[1]}x{deploy_camera0.shape[2]}",
    )
    _show_image_row(
        image_axes[3],
        deploy_camera1,
        deploy_indices,
        deploy_labels,
        f"Deployment camera1\n{deploy_camera1.shape[1]}x{deploy_camera1.shape[2]}",
    )

    split = max(1, sample_count // 2)
    rgb_axis = figure.add_subplot(grid[4, :split])
    luminance_axis = figure.add_subplot(grid[4, split:])
    train_rgb, train_luminance = _normalized_histograms(
        (training.camera0_rgb, training.camera1_rgb)
    )
    deploy_rgb, deploy_luminance = _normalized_histograms((deploy_camera0, deploy_camera1))
    intensity = np.arange(256)
    for channel, (name, color) in enumerate(zip(("R", "G", "B"), RGB_COLORS, strict=True)):
        rgb_axis.plot(intensity, train_rgb[channel], color=color, label=f"Training {name}")
        rgb_axis.plot(
            intensity,
            deploy_rgb[channel],
            color=color,
            linestyle="--",
            label=f"Deployment {name}",
        )
    rgb_axis.set(title="RGB intensity distributions", xlabel="pixel value", ylabel="probability")
    rgb_axis.legend(ncol=3, fontsize=8)
    rgb_axis.grid(alpha=0.2)

    train_mean = float(np.sum(intensity * train_luminance))
    deploy_mean = float(np.sum(intensity * deploy_luminance))
    luminance_axis.plot(intensity, train_luminance, label=f"Training (mean={train_mean:.1f})")
    luminance_axis.plot(
        intensity,
        deploy_luminance,
        linestyle="--",
        label=f"Deployment (mean={deploy_mean:.1f})",
    )
    luminance_axis.set(
        title=f"Luminance distribution (deployment - training = {deploy_mean - train_mean:+.1f})",
        xlabel="luminance",
        ylabel="probability",
    )
    luminance_axis.legend(fontsize=9)
    luminance_axis.grid(alpha=0.2)

    figure.suptitle(
        "SmolVLA raw RGB inputs: training vs real deployment\n"
        "Uniform temporal samples; no augmentation, normalization, or exposure correction",
        fontsize=14,
        fontweight="bold",
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--obs-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-count", default=6, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = write_image_input_comparison(
        training_root=args.training_root,
        obs_dir=args.obs_dir,
        output=args.output,
        sample_count=args.sample_count,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
