from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pytest
import torch


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from loglike_evaluate import (  # noqa: E402
    _run_euler_likelihood,
    _run_fireflow_likelihood,
    save_contribution_curve,
    standard_normal_log_prob,
    velocity_and_hutchinson_trace,
)


def _expand_time(time: torch.Tensor, ndim: int) -> torch.Tensor:
    return time.reshape((time.shape[0],) + (1,) * (ndim - 1))


def _toy_velocity(x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    return 0.25 * x + _expand_time(time, x.ndim)


def _toy_velocity_trace(
    x: torch.Tensor,
    time: torch.Tensor,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step
    event_size = math.prod(x.shape[1:])
    divergence = torch.full(
        (x.shape[0],),
        0.25 * event_size,
        dtype=torch.float32,
        device=x.device,
    )
    return _toy_velocity(x, time), divergence


@pytest.mark.parametrize("num_steps", [1, 2, 7])
def test_solver_shapes_and_nfe(num_steps: int) -> None:
    x = torch.ones(2, 3, 4)
    r_tot = torch.zeros(2)
    time = torch.zeros(2)
    dt = 1.0 / num_steps

    euler_x, euler_r, euler_nfe = _run_euler_likelihood(
        x=x,
        r_tot=r_tot,
        t=time,
        num_steps=num_steps,
        dt=dt,
        velocity_trace_fn=_toy_velocity_trace,
    )
    fireflow_x, fireflow_r, fireflow_nfe = _run_fireflow_likelihood(
        x=x,
        r_tot=r_tot,
        t=time,
        num_steps=num_steps,
        dt=dt,
        velocity_fn=_toy_velocity,
        velocity_trace_fn=_toy_velocity_trace,
    )

    assert euler_x.shape == x.shape
    assert fireflow_x.shape == x.shape
    assert euler_r.shape == r_tot.shape
    assert fireflow_r.shape == r_tot.shape
    assert euler_nfe == num_steps
    assert fireflow_nfe == num_steps + 1


def test_hutchinson_trace_is_exact_for_scaled_identity() -> None:
    x = torch.randn(3, 2, 5)
    velocity, trace = velocity_and_hutchinson_trace(
        lambda value: 0.25 * value,
        x,
        num_samples=2,
        seed=7,
    )
    expected = torch.full((3,), 0.25 * 10)

    torch.testing.assert_close(velocity, 0.25 * x)
    torch.testing.assert_close(trace, expected)


def test_hutchinson_ignores_padded_action_dimensions() -> None:
    x = torch.randn(2, 4, 6)
    action_dim = 3
    _, trace = velocity_and_hutchinson_trace(
        lambda value: 0.5 * value,
        x,
        num_samples=1,
        seed=0,
        action_dim=action_dim,
    )
    expected = torch.full((2,), 0.5 * 4 * action_dim)
    torch.testing.assert_close(trace, expected)


def test_standard_normal_log_prob() -> None:
    x = torch.zeros(2, 3, 4)
    log_prob = standard_normal_log_prob(x)
    expected = torch.full((2,), -0.5 * 12 * math.log(2.0 * math.pi))
    torch.testing.assert_close(log_prob, expected)

    sliced = standard_normal_log_prob(x, action_dim=2)
    sliced_expected = torch.full((2,), -0.5 * 6 * math.log(2.0 * math.pi))
    torch.testing.assert_close(sliced, sliced_expected)


def test_save_contribution_curve(tmp_path: Path) -> None:
    rows = [
        {
            "frame": 0,
            "dataset_index": 10,
            "original_log_likelihood": -1.0,
            "ablated_log_likelihood": -2.0,
            "original_r_tot": 0.2,
            "ablated_r_tot": 0.1,
            "delta_logp": 0.9,
            "delta_r_tot": 0.1,
            "contribution": 1.0,
        },
        {
            "frame": 3,
            "dataset_index": 13,
            "original_log_likelihood": -1.5,
            "ablated_log_likelihood": -2.25,
            "original_r_tot": 0.15,
            "ablated_r_tot": 0.05,
            "delta_logp": 0.65,
            "delta_r_tot": 0.1,
            "contribution": 0.75,
        },
    ]
    csv_path, plot_path = save_contribution_curve(
        rows,
        output_dir=tmp_path,
        modality="vision",
        episode_index="4",
    )

    assert csv_path.is_file()
    assert plot_path.is_file()
    with csv_path.open(newline="", encoding="utf-8") as file:
        saved = list(csv.DictReader(file))
    assert [int(row["frame"]) for row in saved] == [0, 3]
    assert [float(row["contribution"]) for row in saved] == [1.0, 0.75]

