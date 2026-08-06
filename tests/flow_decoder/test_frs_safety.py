from __future__ import annotations

import pathlib

import numpy as np
import pytest

from prepare import _require_finite_cache_batch
from tactile_flow_steering.train import _existing_run_artifacts
from tactile_flow_steering.train import _validate_resume_cache
from tools.compare_frs_reverse_solvers import mean_ratio
from tools.compare_frs_reverse_solvers import summarize_inversion_mse
from tools.train_frs import resolve_resume_mode


def test_cache_batch_finite_check_rejects_nan() -> None:
    _require_finite_cache_batch(actions=np.ones((2, 3), dtype=np.float32))
    with pytest.raises(FloatingPointError, match=r"x_base.*\(1, 2\)"):
        values = np.ones((2, 3), dtype=np.float32)
        values[1, 2] = np.nan
        _require_finite_cache_batch(x_base=values)


def test_fresh_output_guard_ignores_logs_but_finds_training_state(
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / "pipeline_20260101.log").write_text("safe", encoding="utf-8")
    assert _existing_run_artifacts(tmp_path) == ()

    history = tmp_path / "history.csv"
    history.write_text("epoch\n", encoding="utf-8")
    assert _existing_run_artifacts(tmp_path) == (history,)


def test_resume_cache_provenance_must_match() -> None:
    manifest = {
        "records_sha256": "records",
        "configuration": {"reverse_solver": "slerpflow"},
    }
    metadata = {
        "extra_metadata": {
            "cache_records_sha256": "records",
            "cache_configuration": {"reverse_solver": "slerpflow"},
        }
    }
    _validate_resume_cache(metadata, manifest)

    bad = {
        "extra_metadata": {
            "cache_records_sha256": "records",
            "cache_configuration": {"reverse_solver": "fireflow"},
        }
    }
    with pytest.raises(ValueError, match="different action-cache configuration"):
        _validate_resume_cache(bad, manifest)


def test_solver_ab_summary_counts_nonfinite_and_computes_ratio() -> None:
    fire = summarize_inversion_mse(np.asarray([1.0, 2.0, 3.0]))
    slerp = summarize_inversion_mse(np.asarray([0.5, 1.0, np.nan]))
    assert fire["mean"] == 2.0
    assert slerp["nonfinite_count"] == 1
    assert mean_ratio(slerp, fire) == pytest.approx(0.375)


def test_resume_auto_uses_last_checkpoint_when_available(tmp_path: pathlib.Path) -> None:
    assert not resolve_resume_mode("auto", output_dir=tmp_path)
    last = tmp_path / "last"
    last.mkdir()
    (last / "checkpoint.json").write_text("{}", encoding="utf-8")
    assert resolve_resume_mode("auto", output_dir=tmp_path)
    assert not resolve_resume_mode("false", output_dir=tmp_path)
