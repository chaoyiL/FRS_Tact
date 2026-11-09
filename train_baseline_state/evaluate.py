"""Offline metrics for state-only structured-noise Flow correction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from train_baseline_state.action_cache import ActionCache
from train_baseline_state.checkpoint import load_decoder_checkpoint
from train_baseline_state.config import load_config
from train_baseline_state.data import BaselineCacheDataset, make_loader
from torch.nn import functional as F


def _quantiles(
    norm_stats: Mapping[str, Any], dimension: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    stats: Mapping[str, Any] = (
        norm_stats.get("actions", norm_stats) if isinstance(norm_stats, Mapping) else {}
    )
    try:
        low = torch.as_tensor(stats["q01"], dtype=torch.float32, device=device).reshape(-1)
        high = torch.as_tensor(stats["q99"], dtype=torch.float32, device=device).reshape(-1)
    except (KeyError, TypeError) as exc:
        raise ValueError("norm_stats must provide actions.q01 and actions.q99") from exc
    if (
        low.numel() != dimension
        or high.numel() != dimension
        or not torch.isfinite(low).all()
        or not torch.isfinite(high).all()
    ):
        raise ValueError("quantile norm stats have an invalid action dimension or non-finite values")
    return low, high


def _inverse_quantile(values: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return low + (values + 1.0) * 0.5 * (high - low + 1e-6)


@torch.no_grad()
def evaluate_decoder(
    model: torch.nn.Module, loader: Any, norm_stats: Mapping[str, Any], *, use_quantile_norm: bool = True
) -> dict[str, float]:
    """Evaluate masked normalized and inverse-quantile physical action metrics."""
    was_training = model.training
    model.eval()
    totals = {
        name: 0.0
        for name in (
            "decoder_smooth_l1",
            "coarse_smooth_l1",
            "decoder_mse",
            "coarse_mse",
            "delta_sq",
            "physical_abs",
            "physical_sq",
        )
    }
    gripper_totals: dict[int, float] = {}
    count = physical_count = gripper_count = 0
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    for batch in loader:
        coarse = batch["coarse"].to(device=device, dtype=torch.float32)
        target = batch["target"].to(device=device, dtype=torch.float32)
        structured_noise = batch["structured_noise"].to(device=device, dtype=torch.float32)
        state = batch["state"].to(device=device, dtype=torch.float32)
        valid = batch["valid"].to(device=device, dtype=torch.bool)
        expanded = valid.unsqueeze(-1).expand_as(target)
        elements = int(expanded.sum().item())
        if elements == 0:
            continue
        predicted = model.decode(structured_noise, state=state, vla_action=coarse)
        error = predicted - target
        coarse_error = coarse - target
        totals["decoder_smooth_l1"] += float(
            (F.smooth_l1_loss(predicted, target, reduction="none") * expanded).sum().item()
        )
        totals["coarse_smooth_l1"] += float(
            (F.smooth_l1_loss(coarse, target, reduction="none") * expanded).sum().item()
        )
        totals["decoder_mse"] += float((error.square() * expanded).sum().item())
        totals["coarse_mse"] += float((coarse_error.square() * expanded).sum().item())
        totals["delta_sq"] += float(((predicted - coarse).square() * expanded).sum().item())
        if use_quantile_norm:
            low, high = _quantiles(norm_stats, target.shape[-1], device)
            physical_error = _inverse_quantile(predicted, low, high) - _inverse_quantile(target, low, high)
        else:
            stats = norm_stats.get("actions", norm_stats)
            scale = torch.as_tensor(stats["std"], dtype=torch.float32, device=device).reshape(-1)
            if scale.numel() != target.shape[-1] or not torch.isfinite(scale).all() or torch.any(scale < 0):
                raise ValueError(
                    "action standard deviations must be finite, nonnegative and match action width"
                )
            physical_error = (predicted - target) * (scale + 1e-6)
        totals["physical_abs"] += float((physical_error.abs() * expanded).sum().item())
        totals["physical_sq"] += float((physical_error.square() * expanded).sum().item())
        gripper_valid = valid.sum().item()
        for index in range(9, target.shape[-1], 10):
            gripper_totals[index] = gripper_totals.get(index, 0.0) + float(
                (error[..., index].abs() * valid).sum().item()
            )
        count += elements
        physical_count += elements
        gripper_count += int(gripper_valid)
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
    }
    result.update(
        {f"normalized_gripper_mae_{index}": total / gripper_count for index, total in gripper_totals.items()}
    )
    result["relative_mse_reduction"] = (
        0.0
        if result["coarse_mse"] == 0.0
        else (result["coarse_mse"] - result["decoder_mse"]) / result["coarse_mse"]
    )
    return result


def write_metrics(metrics: Mapping[str, float], path: Path) -> Path:
    """Write metrics only; this never changes caches or checkpoints."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(metrics))
            writer.writeheader()
            writer.writerow(metrics)
    else:
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_norm_stats(config: Any) -> Mapping[str, Any]:
    path = Path(config.source.norm_stats_dir) / str(config.source.norm_stats_asset_id) / "norm_stats.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("norm_stats", raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    evaluation = config.evaluation
    split = args.split or evaluation.split
    checkpoint = args.checkpoint or (Path(config.decoder.output) / "best.pt")
    output = args.output if args.output is not None else evaluation.output
    action = ActionCache.open(config.cache.action_root)
    loader = make_loader(
        BaselineCacheDataset(action, split),
        batch_size=evaluation.batch_size,
        shuffle=False,
        seed=config.decoder.seed,
        workers=config.decoder.workers,
        pin_memory=config.decoder.pin_memory,
    )
    model, _ = load_decoder_checkpoint(checkpoint, map_location=config.decoder.device)
    model.to(config.decoder.device)
    metrics = evaluate_decoder(
        model, loader, _load_norm_stats(config), use_quantile_norm=config.source.use_quantile_norm
    )
    if output is not None:
        write_metrics(metrics, output)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
