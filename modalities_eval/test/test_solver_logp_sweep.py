from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVAL_SCRIPTS = ROOT / "modalities_eval"
if str(EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPTS))

import pytest
import solver_logp_sweep as sweep


def _curve(values: list[tuple[int, float]]) -> list[sweep.LogpRow]:
    return [
        sweep.LogpRow(k=k, log_likelihood=logp, log_p_base=0.0, r_tot=logp)
        for k, logp in values
    ]


def test_detect_convergence_finds_first_stable_window() -> None:
    curve = _curve(
        [
            (10, 100.0),
            (20, 110.0),
            (30, 110.4),
            (40, 110.7),
            (50, 110.8),
        ]
    )
    result = sweep.detect_convergence(curve, atol=1.0, patience=2)
    assert result.converged is True
    assert result.k_star == 40


def test_detect_convergence_falls_back_to_max_k() -> None:
    curve = _curve(
        [
            (10, 0.0),
            (20, 10.0),
            (30, 20.0),
            (40, 30.0),
        ]
    )
    result = sweep.detect_convergence(curve, atol=1.0, patience=2)
    assert result.converged is False
    assert result.k_star == 40


def test_detect_convergence_rejects_bad_args() -> None:
    curve = _curve([(10, 1.0), (20, 1.0)])
    with pytest.raises(ValueError, match="atol"):
        sweep.detect_convergence(curve, atol=-0.1, patience=1)
    with pytest.raises(ValueError, match="patience"):
        sweep.detect_convergence(curve, atol=1.0, patience=0)
    with pytest.raises(ValueError, match="non-empty"):
        sweep.detect_convergence([], atol=1.0, patience=1)


def test_parse_k_values_sorts_and_deduplicates() -> None:
    assert sweep.parse_k_values(["30", "10", "20", "10"]) == (10, 20, 30)


def test_default_config_exists() -> None:
    assert sweep.DEFAULT_CONFIG.is_file()
    cfg = sweep.load_yaml_config(sweep.DEFAULT_CONFIG)
    assert "data" in cfg and "experiment" in cfg
    assert "k_values" in cfg["experiment"]
