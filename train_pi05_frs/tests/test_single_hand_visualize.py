from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from train_pi05_frs.utils.metrics import EvaluationResult
from train_pi05_frs.utils.single_hand_visualize import _HISTORY_FIELDS


def _write_history(path: Path) -> Path:
    fields = ["epoch", *_HISTORY_FIELDS]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in (5, 10):
            row = {field: 0.2 for field in fields}
            row.update(
                {
                    "epoch": epoch,
                    "val_gate_w_p10": 0.1,
                    "val_gate_w_p50": 0.5,
                    "val_gate_w_p90": 0.9,
                    "val_tactile_change_p10": 0.05,
                    "val_tactile_change_p50": 0.4,
                    "val_tactile_change_p90": 0.8,
                    "val_gate_n_low": 2,
                    "val_gate_n_mid": 1,
                    "val_gate_n_high": 2,
                    "checkpoint_selection_feasible": 1,
                }
            )
            writer.writerow(row)
    return path


def _result() -> EvaluationResult:
    gates = np.asarray((0.1, 0.2, 0.5, 0.8, 0.9), dtype=np.float64)
    samples, horizon, action_dim = len(gates), 4, 10
    gt = np.zeros((samples, horizon, action_dim), dtype=np.float32)
    vla = np.ones_like(gt)
    frs = np.asarray(
        [(1.0 - gate) * vla[index] for index, gate in enumerate(gates)],
        dtype=np.float32,
    )
    mse_gt = np.mean(np.square(frs - gt), axis=(1, 2))
    mse_vla = np.mean(np.square(frs - vla), axis=(1, 2))
    mse_vla_gt = np.mean(np.square(vla - gt), axis=(1, 2))
    zeros = np.zeros(samples, dtype=np.float64)
    return EvaluationResult(
        target="gt",
        flow_loss=0.2,
        mse=float(np.mean(mse_gt)),
        rmse=float(np.sqrt(np.mean(mse_gt))),
        mae=0.2,
        flow_loss_gt=0.2,
        mse_gt=float(np.mean(mse_gt)),
        rmse_gt=float(np.sqrt(np.mean(mse_gt))),
        mae_gt=0.2,
        flow_loss_pred=0.3,
        mse_pred=float(np.mean(mse_vla)),
        rmse_pred=float(np.sqrt(np.mean(mse_vla))),
        mae_pred=0.3,
        cache_indices=np.arange(samples),
        sample_flow_loss=zeros,
        sample_mse=mse_gt,
        sample_rmse=np.sqrt(mse_gt),
        sample_mae=zeros,
        sample_mse_gt=mse_gt,
        sample_mae_gt=zeros,
        sample_mse_pred=mse_vla,
        sample_mae_pred=zeros,
        predictions=frs,
        sample_gate_w=gates,
        sample_tactile_change=np.asarray((0.0, 0.1, 0.4, 0.7, 0.9)),
        sample_mse_vla_gt=mse_vla_gt,
        sample_gt_gain=mse_vla_gt - mse_gt,
        sample_relative_gt_error=mse_gt / mse_vla_gt,
        composite_fm=0.2,
        sample_composite_fm=np.full(samples, 0.2),
        gate_low_threshold=0.3,
        gate_high_threshold=0.7,
        gt_actions=gt,
        vla_actions=vla,
    )


def test_single_hand_plot_bundle_writes_stable_filenames(tmp_path: Path) -> None:
    from train_pi05_frs.utils.single_hand_visualize import (
        plot_single_hand_diagnostics,
    )

    history = _write_history(tmp_path / "history.csv")
    paths = plot_single_hand_diagnostics(history, _result(), output_dir=tmp_path)
    assert {path.name for path in paths} == {
        "training_overview.png",
        "single_hand_behavior.png",
        "gate_diagnostics.png",
        "single_hand_action_examples.png",
    }
    assert all(path.stat().st_size > 0 for path in paths)
