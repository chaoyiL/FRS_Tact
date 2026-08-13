"""Artifact writing for FRS modality intervention evaluations."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from modalities_eval.frs.statistics import summarize_rows


PER_SAMPLE_COLUMNS = (
    "cache_index",
    "source",
    "source_index",
    "source_cache_index",
    "dataset_index",
    "episode_index",
    "condition",
    "original_gate",
    "counterfactual_gate",
    "mse_gt",
    "mse_vla",
    "mse_vla_gt",
    "gt_gain",
    "repair_success",
    "contribution",
)
PER_EPISODE_COLUMNS = (
    "condition",
    "source",
    "source_index",
    "episode_index",
    "sample_count",
    "original_gate",
    "counterfactual_gate",
    "mse_gt",
    "mse_vla",
    "mse_vla_gt",
    "gt_gain",
    "repair_success_rate",
    "contribution",
)
_EPISODE_MEAN_COLUMNS = (
    "original_gate",
    "counterfactual_gate",
    "mse_gt",
    "mse_vla",
    "mse_vla_gt",
    "gt_gain",
    "contribution",
)


def _write_csv(path: Path, *, fieldnames: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in fieldnames})
    return path


def write_per_sample_csv(rows: Iterable[Mapping[str, object]], path: Path) -> Path:
    """Write paired sample rows with a fixed, documented column order."""

    return _write_csv(path, fieldnames=PER_SAMPLE_COLUMNS, rows=rows)


def _episode_group_key(row: Mapping[str, object]) -> tuple[object, ...]:
    if "condition" not in row or "episode_index" not in row:
        raise ValueError("each report row must include condition and episode_index")
    source_index = row.get("source_index")
    source_key = ("source_index", source_index) if source_index is not None else ("source", row.get("source"))
    return str(row["condition"]), *source_key, row["episode_index"]


def _group_sort_key(group_key: tuple[object, ...]) -> tuple[str, ...]:
    return tuple(str(value) for value in group_key)


def aggregate_episode_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Average rows by condition, source identity, and episode index."""

    grouped: defaultdict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_episode_group_key(row)].append(row)

    episode_rows = []
    for key in sorted(grouped, key=_group_sort_key):
        group = grouped[key]
        first = group[0]
        aggregate: dict[str, object] = {
            "condition": first["condition"],
            "source": first.get("source"),
            "source_index": first.get("source_index"),
            "episode_index": first["episode_index"],
            "sample_count": len(group),
            "repair_success_rate": float(np.mean([bool(row["repair_success"]) for row in group])),
        }
        aggregate.update(
            {
                metric: float(np.mean([float(row[metric]) for row in group]))
                for metric in _EPISODE_MEAN_COLUMNS
            }
        )
        episode_rows.append(aggregate)
    return episode_rows


def write_per_episode_csv(rows: Iterable[Mapping[str, object]], path: Path) -> Path:
    """Write episode-level aggregates with a fixed column order."""

    return _write_csv(path, fieldnames=PER_EPISODE_COLUMNS, rows=rows)


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_summary_json(summary: Mapping[str, object], path: Path) -> Path:
    """Write standard JSON, representing undefined empty-stratum metrics as null."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(summary), file, allow_nan=False, indent=2, sort_keys=True)
        file.write("\n")
    return path


def _matplotlib_pyplot():
    cache_dir = Path("/tmp/modalities_eval_mpl")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    return plt


def write_contribution_plot(summary: Mapping[str, Mapping[str, object]], path: Path) -> Path:
    """Plot condition mean contribution with episode-cluster bootstrap intervals."""

    plt = _matplotlib_pyplot()
    conditions = sorted(summary)
    means = [float(summary[name]["mean_contribution"]) for name in conditions]
    intervals = [summary[name]["mean_contribution_ci"] for name in conditions]
    lower = [mean - float(interval["lower"]) for mean, interval in zip(means, intervals, strict=True)]
    upper = [float(interval["upper"]) - mean for mean, interval in zip(means, intervals, strict=True)]

    figure, axis = plt.subplots(figsize=(max(6.0, 0.9 * len(conditions)), 4.5))
    if conditions:
        positions = np.arange(len(conditions))
        axis.bar(positions, means, yerr=[lower, upper], capsize=4)
        axis.set_xticks(positions, conditions, rotation=35, ha="right")
    else:
        axis.text(0.5, 0.5, "No evaluation rows", ha="center", va="center", transform=axis.transAxes)
        axis.set_xticks([])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Contribution: MSE(condition) - MSE(full)")
    axis.set_title("FRS modality contribution (95% episode-cluster bootstrap CI)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def write_report(
    rows: Iterable[Mapping[str, object]],
    *,
    output_dir: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
    rank_low_gate_threshold: float,
    rank_high_gate_threshold: float,
    provenance: Mapping[str, object],
) -> dict[str, Path]:
    """Write the complete FRS modality report and return its artifact paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized_rows = list(rows)
    summary = summarize_rows(
        materialized_rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        rank_low_gate_threshold=rank_low_gate_threshold,
        rank_high_gate_threshold=rank_high_gate_threshold,
    )
    per_sample = write_per_sample_csv(materialized_rows, output_dir / "per_sample.csv")
    per_episode = write_per_episode_csv(
        aggregate_episode_rows(materialized_rows), output_dir / "per_episode.csv"
    )
    summary_document = {**summary, "provenance": dict(provenance)}
    summary_path = write_summary_json(summary_document, output_dir / "summary.json")
    plot_path = write_contribution_plot(summary, output_dir / "contribution.png")
    return {
        "per_sample": per_sample,
        "per_episode": per_episode,
        "summary": summary_path,
        "plot": plot_path,
    }
