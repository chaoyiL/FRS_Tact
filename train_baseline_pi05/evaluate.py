"""CPU-friendly masked metrics for the direct tactile action decoder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from train_baseline_pi05.model import DirectTactileActionDecoder


def _quantiles(norm_stats: Mapping[str, Any], dimension: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    stats: Mapping[str, Any] = norm_stats.get("actions", norm_stats) if isinstance(norm_stats, Mapping) else {}
    try:
        low = torch.as_tensor(stats["q01"], dtype=torch.float32, device=device).reshape(-1)
        high = torch.as_tensor(stats["q99"], dtype=torch.float32, device=device).reshape(-1)
    except (KeyError, TypeError) as exc:
        raise ValueError("norm_stats must provide actions.q01 and actions.q99") from exc
    if low.numel() != dimension or high.numel() != dimension or not torch.isfinite(low).all() or not torch.isfinite(high).all():
        raise ValueError("quantile norm stats have an invalid action dimension or non-finite values")
    return low, high


def _inverse_quantile(values: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return low + (values + 1.0) * 0.5 * (high - low)


def _episode_shuffle(tactile: torch.Tensor, episodes: torch.Tensor) -> torch.Tensor:
    """Deterministically rotate batch tactile rows, avoiding same-episode rows when possible."""
    if tactile.shape[0] < 2:
        return tactile
    order = torch.roll(torch.arange(tactile.shape[0], device=tactile.device), shifts=1)
    if torch.any(episodes[order] == episodes):
        candidates = [index for index in range(tactile.shape[0]) if torch.all(episodes[index] != episodes)]
        if candidates:
            order = torch.tensor(candidates[: tactile.shape[0]], device=tactile.device)
    return tactile[order]


@torch.no_grad()
def evaluate_decoder(
    model: torch.nn.Module, loader: Any, norm_stats: Mapping[str, Any], *, shuffle_tactile: bool = False
) -> dict[str, float]:
    """Evaluate masked normalized and inverse-quantile physical action metrics."""
    was_training = model.training
    model.eval()
    totals = {name: 0.0 for name in ("decoder_smooth_l1", "coarse_smooth_l1", "decoder_mse", "coarse_mse", "delta_sq", "physical_abs", "physical_sq", "gripper_9", "gripper_19", "shuffled_decoder_mse")}
    count = physical_count = gripper_count = 0
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    for batch in loader:
        coarse = batch["coarse"].to(device=device, dtype=torch.float32)
        target = batch["target"].to(device=device, dtype=torch.float32)
        tactile = batch["tactile"].to(device=device, dtype=torch.float32)
        valid = batch["valid"].to(device=device, dtype=torch.bool)
        expanded = valid.unsqueeze(-1).expand_as(target)
        elements = int(expanded.sum().item())
        if elements == 0:
            continue
        predicted = model(coarse, tactile)
        error = predicted - target
        coarse_error = coarse - target
        totals["decoder_smooth_l1"] += float((F.smooth_l1_loss(predicted, target, reduction="none") * expanded).sum().item())
        totals["coarse_smooth_l1"] += float((F.smooth_l1_loss(coarse, target, reduction="none") * expanded).sum().item())
        totals["decoder_mse"] += float((error.square() * expanded).sum().item())
        totals["coarse_mse"] += float((coarse_error.square() * expanded).sum().item())
        totals["delta_sq"] += float(((predicted - coarse).square() * expanded).sum().item())
        low, high = _quantiles(norm_stats, target.shape[-1], device)
        physical_error = _inverse_quantile(predicted, low, high) - _inverse_quantile(target, low, high)
        totals["physical_abs"] += float((physical_error.abs() * expanded).sum().item())
        totals["physical_sq"] += float((physical_error.square() * expanded).sum().item())
        gripper_valid = valid.sum().item()
        totals["gripper_9"] += float((error[..., 9].abs() * valid).sum().item())
        totals["gripper_19"] += float((error[..., 19].abs() * valid).sum().item())
        count += elements; physical_count += elements; gripper_count += int(gripper_valid)
        if shuffle_tactile:
            shuffled = model(coarse, _episode_shuffle(tactile, batch["episode_index"].to(device)))
            totals["shuffled_decoder_mse"] += float(((shuffled - target).square() * expanded).sum().item())
    if was_training:
        model.train()
    if count == 0:
        raise ValueError("evaluation loader has no valid action elements")
    result = {
        "decoder_smooth_l1": totals["decoder_smooth_l1"] / count,
        "coarse_smooth_l1": totals["coarse_smooth_l1"] / count,
        "decoder_mse": totals["decoder_mse"] / count,
        "coarse_mse": totals["coarse_mse"] / count,
        "delta_rms": (totals["delta_sq"] / count) ** 0.5,
        "physical_mae": totals["physical_abs"] / physical_count,
        "physical_rmse": (totals["physical_sq"] / physical_count) ** 0.5,
        "normalized_gripper_mae_9": totals["gripper_9"] / gripper_count,
        "normalized_gripper_mae_19": totals["gripper_19"] / gripper_count,
    }
    result["relative_mse_reduction"] = 0.0 if result["coarse_mse"] == 0.0 else (result["coarse_mse"] - result["decoder_mse"]) / result["coarse_mse"]
    if shuffle_tactile:
        result["shuffled_decoder_mse"] = totals["shuffled_decoder_mse"] / count
    return result


def write_metrics(metrics: Mapping[str, float], path: Path) -> Path:
    """Write metrics only; this never changes caches or checkpoints."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(metrics)); writer.writeheader(); writer.writerow(metrics)
    else:
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True, type=Path); parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from train_baseline_pi05.checkpoint import load_decoder_checkpoint
    model, metadata = load_decoder_checkpoint(args.checkpoint)
    if args.output is not None:
        write_metrics({str(key): float(value) for key, value in metadata["metrics"].items()}, args.output)


if __name__ == "__main__":
    main()
