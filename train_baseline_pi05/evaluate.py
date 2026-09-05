"""CPU-friendly masked metrics for the direct tactile action decoder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from train_baseline_pi05.action_cache import ActionCache
from train_baseline_pi05.checkpoint import load_decoder_checkpoint
from train_baseline_pi05.config import load_config
from train_baseline_pi05.data import BaselineCacheDataset, make_loader
from train_baseline_pi05.tactile_cache import TactileEmbeddingCache
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
    return low + (values + 1.0) * 0.5 * (high - low + 1e-6)


def _cross_episode_order(episodes: torch.Tensor) -> torch.Tensor:
    """Return a deterministic complete permutation whose donors are from other episodes."""
    if episodes.ndim != 1:
        raise ValueError("episode indices must be one-dimensional")
    episode_values = episodes.detach().cpu().tolist()
    if not episode_values:
        raise ValueError("cross-episode tactile permutation is impossible for this evaluation split")
    targets = sorted(range(len(episode_values)), key=lambda index: (episode_values[index], index))
    largest_group = max(episode_values.count(episode) for episode in set(episode_values))
    if largest_group > len(episode_values) - largest_group:
        raise ValueError("cross-episode tactile permutation is impossible for this evaluation split")
    sources = targets[largest_group:] + targets[:largest_group]
    order = torch.empty(len(episode_values), dtype=torch.int64, device=episodes.device)
    order[torch.tensor(targets, device=episodes.device)] = torch.tensor(sources, device=episodes.device)
    if torch.any(episodes[order] == episodes):
        raise ValueError("cross-episode tactile permutation is impossible for this evaluation split")
    return order


def _episode_shuffle(tactile: torch.Tensor, episodes: torch.Tensor) -> torch.Tensor:
    """Apply a strict complete cross-episode permutation to tactile rows."""
    if tactile.shape[0] != episodes.shape[0]:
        raise ValueError("tactile rows and episode indices must have the same length")
    order = _cross_episode_order(episodes.to(tactile.device))
    return tactile[order]


def _tactile_donor_lookup(loader: Any) -> tuple[dict[int, int], Any]:
    """Plan global donors from cache indices while keeping tactile tokens on disk."""
    dataset = getattr(loader, "dataset", None)
    action_cache = getattr(dataset, "action_cache", None)
    tactile_cache = getattr(dataset, "tactile_cache", None)
    rows = getattr(dataset, "rows", None)
    if action_cache is None or tactile_cache is None or rows is None:
        raise ValueError("shuffled tactile evaluation requires a cache-backed evaluation dataset")
    rows_array = np.asarray(rows, dtype=np.int64)
    episodes = np.asarray(action_cache.episode_indices[rows_array], dtype=np.int64)
    frames = np.asarray(action_cache.dataset_indices[rows_array], dtype=np.int64)
    if len(set(frames.tolist())) != len(frames):
        raise ValueError("evaluation dataset contains duplicate frame indices")
    order = _cross_episode_order(torch.from_numpy(episodes)).cpu().numpy()
    return {int(frame): int(frames[source]) for frame, source in zip(frames, order, strict=True)}, tactile_cache


@torch.no_grad()
def evaluate_decoder(
    model: torch.nn.Module, loader: Any, norm_stats: Mapping[str, Any], *, shuffle_tactile: bool = False
) -> dict[str, float]:
    """Evaluate masked normalized and inverse-quantile physical action metrics."""
    donor_lookup, donor_cache = _tactile_donor_lookup(loader) if shuffle_tactile else ({}, None)
    was_training = model.training
    model.eval()
    totals = {name: 0.0 for name in ("decoder_smooth_l1", "coarse_smooth_l1", "decoder_mse", "coarse_mse", "delta_sq", "physical_abs", "physical_sq", "shuffled_decoder_mse")}
    gripper_totals: dict[int, float] = {}
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
        for index in range(9, target.shape[-1], 10):
            gripper_totals[index] = gripper_totals.get(index, 0.0) + float(
                (error[..., index].abs() * valid).sum().item()
            )
        count += elements; physical_count += elements; gripper_count += int(gripper_valid)
        if shuffle_tactile:
            try:
                donor_frames = [donor_lookup[int(frame)] for frame in batch["dataset_index"].tolist()]
            except KeyError as exc:
                raise ValueError("evaluation batch contains a frame outside the donor plan") from exc
            donor_tokens = np.array(donor_cache.get_many(donor_frames), copy=True)
            shuffled_tactile = torch.from_numpy(donor_tokens).to(device=device, dtype=torch.float32)
            shuffled = model(coarse, shuffled_tactile)
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
    }
    result.update({f"normalized_gripper_mae_{index}": total / gripper_count for index, total in gripper_totals.items()})
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


def _load_norm_stats(config: Any) -> Mapping[str, Any]:
    path = Path(config.source.norm_stats_dir) / str(config.source.norm_stats_asset_id) / "norm_stats.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("norm_stats", raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--shuffle-tactile", action="store_true", default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    evaluation = config.evaluation
    split = args.split or evaluation.split
    checkpoint = args.checkpoint or (Path(config.decoder.output) / "best.pt")
    output = args.output if args.output is not None else evaluation.output
    shuffle = evaluation.shuffle_tactile if args.shuffle_tactile is None else args.shuffle_tactile
    action = ActionCache.open(config.cache.action_root)
    tactile = TactileEmbeddingCache.open(
        config.cache.tactile_root,
        tactile_keys=config.decoder.tactile_keys,
        encoder_path=config.tactile.encoder_checkpoint,
    )
    loader = make_loader(BaselineCacheDataset(action, tactile, split), batch_size=evaluation.batch_size, shuffle=False, seed=config.decoder.seed, workers=config.decoder.workers, pin_memory=config.decoder.pin_memory)
    model, _ = load_decoder_checkpoint(checkpoint, map_location=config.decoder.device)
    model.to(config.decoder.device)
    metrics = evaluate_decoder(model, loader, _load_norm_stats(config), shuffle_tactile=shuffle)
    if output is not None:
        write_metrics(metrics, output)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
