"""Paired error rows and episode-clustered summary statistics."""

from collections.abc import Iterable
from collections.abc import Mapping

import numpy as np


_MEAN_METRICS = (
    "contribution",
    "gt_gain",
    "mse_gt",
    "mse_vla",
    "mse_vla_gt",
    "original_gate",
    "counterfactual_gate",
)


def sample_error_rows(*, full, conditions, vla, gt, metadata, original_gate, counterfactual_gates):
    """Return per-sample paired errors for every counterfactual condition."""

    full_mse_gt = np.mean(np.square(full - gt), axis=(1, 2))
    vla_mse_gt = np.mean(np.square(vla - gt), axis=(1, 2))
    rows = []
    for condition, prediction in conditions.items():
        mse_gt = np.mean(np.square(prediction - gt), axis=(1, 2))
        mse_vla = np.mean(np.square(prediction - vla), axis=(1, 2))
        for index in range(len(gt)):
            rows.append(
                {
                    **metadata[index],
                    "condition": condition,
                    "original_gate": float(original_gate[index]),
                    "counterfactual_gate": float(counterfactual_gates[condition][index]),
                    "mse_gt": float(mse_gt[index]),
                    "mse_vla": float(mse_vla[index]),
                    "mse_vla_gt": float(vla_mse_gt[index]),
                    "gt_gain": float(vla_mse_gt[index] - mse_gt[index]),
                    "repair_success": bool(mse_gt[index] < vla_mse_gt[index]),
                    "contribution": float(mse_gt[index] - full_mse_gt[index]),
                }
            )
    return rows


def _episode_clusters(rows: Iterable[Mapping[str, object]]):
    clusters: dict[object, list[Mapping[str, object]]] = {}
    for row in rows:
        if "episode_index" not in row:
            raise ValueError("each row must include episode_index for clustered bootstrap")
        source_index = row.get("source_index")
        source = row.get("source")
        if source_index is not None:
            cluster_key = ("source_index", source_index, row["episode_index"])
        elif source is not None:
            cluster_key = ("source", source, row["episode_index"])
        else:
            cluster_key = row["episode_index"]
        clusters.setdefault(cluster_key, []).append(row)
    return tuple(clusters.values())


def _cluster_aggregates(clusters: tuple[list[Mapping[str, object]], ...]):
    metrics = (*_MEAN_METRICS, "repair_success")
    counts = np.fromiter((len(cluster) for cluster in clusters), dtype=np.int64)
    sums = {
        metric: np.fromiter(
            (sum(float(row[metric]) for row in cluster) for cluster in clusters),
            dtype=np.float64,
            count=len(clusters),
        )
        for metric in metrics
    }
    return counts, sums


def _summary_for_rows(rows, *, bootstrap_samples: int, rng: np.random.Generator):
    clusters = _episode_clusters(rows)
    if not rows:
        empty_ci = {"lower": None, "upper": None}
        bootstrap = {
            **{f"mean_{metric}": dict(empty_ci) for metric in _MEAN_METRICS},
            "repair_success_rate": dict(empty_ci),
        }
        return {
            "sample_count": 0,
            "episode_count": 0,
            "repair_success_rate": None,
            "repair_success_rate_ci": dict(empty_ci),
            **{
                key: value
                for metric in _MEAN_METRICS
                for key, value in (
                    (f"mean_{metric}", None),
                    (f"mean_{metric}_ci", dict(empty_ci)),
                )
            },
            "bootstrap": bootstrap,
        }
    cluster_counts, cluster_sums = _cluster_aggregates(clusters)
    if clusters:
        draw_indices = rng.integers(
            0,
            len(clusters),
            size=(bootstrap_samples, len(clusters)),
        )
        draw_counts = cluster_counts[draw_indices].sum(axis=1)
        draws = {
            metric: sums[draw_indices].sum(axis=1) / draw_counts
            for metric, sums in cluster_sums.items()
        }
    summary = {
        "sample_count": len(rows),
        "episode_count": len(clusters),
        "repair_success_rate": float(
            cluster_sums["repair_success"].sum() / cluster_counts.sum()
        ),
    }
    bootstrap = {}
    for metric in _MEAN_METRICS:
        summary[f"mean_{metric}"] = float(
            cluster_sums[metric].sum() / cluster_counts.sum()
        )
        ci = _confidence_interval(draws[metric])
        summary[f"mean_{metric}_ci"] = ci
        bootstrap[f"mean_{metric}"] = ci
    repair_success_ci = _confidence_interval(draws["repair_success"])
    summary["repair_success_rate_ci"] = repair_success_ci
    bootstrap["repair_success_rate"] = repair_success_ci
    summary["bootstrap"] = bootstrap
    return summary


def _confidence_interval(values: np.ndarray):
    lower, upper = np.quantile(values, (0.025, 0.975))
    return {"lower": float(lower), "upper": float(upper)}


def summarize_rows(
    rows,
    bootstrap_samples=1000,
    bootstrap_seed=0,
    *,
    rank_low_gate_threshold=0.3,
    rank_high_gate_threshold=0.7,
):
    """Summarize paired rows, stratified by original gate regions.

    Bootstrap samples resample entire episodes, preserving the paired
    observations within each episode.
    """

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    low_threshold = float(rank_low_gate_threshold)
    high_threshold = float(rank_high_gate_threshold)
    if not 0.0 <= low_threshold < high_threshold <= 1.0:
        raise ValueError("gate thresholds must satisfy 0 <= low < high <= 1")
    by_condition: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    rng = np.random.default_rng(bootstrap_seed)
    summary = {}
    for condition, condition_rows in by_condition.items():
        low_rows = [row for row in condition_rows if float(row["original_gate"]) <= low_threshold]
        transition_rows = [
            row
            for row in condition_rows
            if low_threshold < float(row["original_gate"]) < high_threshold
        ]
        high_rows = [row for row in condition_rows if float(row["original_gate"]) >= high_threshold]
        condition_summary = _summary_for_rows(
            condition_rows, bootstrap_samples=bootstrap_samples, rng=rng
        )
        condition_summary["gate_thresholds"] = {
            "low": low_threshold,
            "high": high_threshold,
        }
        condition_summary["low"] = _summary_for_rows(
            low_rows, bootstrap_samples=bootstrap_samples, rng=rng
        )
        condition_summary["transition"] = _summary_for_rows(
            transition_rows, bootstrap_samples=bootstrap_samples, rng=rng
        )
        condition_summary["high"] = _summary_for_rows(
            high_rows, bootstrap_samples=bootstrap_samples, rng=rng
        )
        summary[condition] = condition_summary
    return summary
