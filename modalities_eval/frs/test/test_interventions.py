import subprocess
import sys

import numpy as np
import pytest

from modalities_eval.frs.interventions import DEFAULT_INTERVENTIONS
from modalities_eval.frs.interventions import Intervention
from modalities_eval.frs.interventions import apply_intervention
from modalities_eval.frs.interventions import gate_weights_from_change
from modalities_eval.frs.interventions import tactile_change_from_tokens


def test_baseline_recomputed_replaces_window_and_recomputes_gate():
    tactile = np.arange(2 * 3 * 4 * 2, dtype=np.float32).reshape(2, 3, 4, 2)
    baseline = np.ones((2, 4, 2), dtype=np.float32)

    result = apply_intervention(
        "baseline_recomputed",
        tactile,
        baseline,
        np.array([0.8, 0.9]),
        tau=0.4,
        temperature=0.1,
    )

    np.testing.assert_array_equal(result.tactile, np.repeat(baseline[:, None], 3, axis=1))
    assert result.recomputed_gate is True


def test_fixed_window_interventions_preserve_original_gate():
    tactile = np.arange(1 * 3 * 4 * 2, dtype=np.float32).reshape(1, 3, 4, 2)
    baseline = np.full((1, 4, 2), -1.0, dtype=np.float32)
    original_gate = np.array([0.8], dtype=np.float32)

    baseline_fixed = apply_intervention(
        "baseline_fixed", tactile, baseline, original_gate, tau=0.4, temperature=0.1
    )
    current_only = apply_intervention(
        "current_only", tactile, baseline, original_gate, tau=0.4, temperature=0.1
    )

    np.testing.assert_array_equal(
        baseline_fixed.tactile, np.repeat(baseline[:, None], 3, axis=1)
    )
    np.testing.assert_array_equal(current_only.tactile, np.repeat(tactile[:, -1:], 3, axis=1))
    np.testing.assert_array_equal(baseline_fixed.gate, original_gate)
    assert baseline_fixed.recomputed_gate is False
    assert current_only.recomputed_gate is False


def test_sensor_intervention_replaces_only_target_values():
    tactile = np.arange(1 * 2 * 4 * 2, dtype=np.float32).reshape(1, 2, 4, 2)
    baseline = np.full((1, 4, 2), -1.0, dtype=np.float32)
    original_gate = np.array([0.8], dtype=np.float32)

    dropped = apply_intervention(
        "drop_sensor_2", tactile, baseline, original_gate, tau=0.4, temperature=0.1
    )
    np.testing.assert_array_equal(
        dropped.tactile[:, :, 2, :], np.repeat(baseline[:, None, 2, :], 2, axis=1)
    )
    np.testing.assert_array_equal(dropped.tactile[:, :, :2, :], tactile[:, :, :2, :])
    np.testing.assert_array_equal(dropped.gate, original_gate)


def test_interventions_validate_input_and_expose_default_names():
    tactile = np.zeros((1, 2, 4, 2), dtype=np.float32)
    baseline = np.zeros((1, 4, 2), dtype=np.float32)
    gate = np.array([0.5], dtype=np.float32)

    assert all(isinstance(item, Intervention) for item in DEFAULT_INTERVENTIONS)
    assert {item.name for item in DEFAULT_INTERVENTIONS} == {
        "baseline_fixed",
        "baseline_recomputed",
        "current_only",
        "drop_sensor_0",
        "drop_sensor_1",
        "drop_sensor_2",
        "drop_sensor_3",
    }
    with pytest.raises(ValueError, match="expected tactile"):
        apply_intervention("baseline_fixed", tactile[:, 0], baseline, gate, tau=0.4, temperature=0.1)
    with pytest.raises(ValueError, match="sensor index out of range"):
        apply_intervention("drop_sensor_4", tactile, baseline, gate, tau=0.4, temperature=0.1)
    with pytest.raises(ValueError, match="unsupported intervention"):
        apply_intervention("unknown", tactile, baseline, gate, tau=0.4, temperature=0.1)
    with pytest.raises(ValueError, match="unsupported intervention"):
        apply_intervention("gate_0.5", tactile, baseline, gate, tau=0.4, temperature=0.1)


def test_interventions_validate_gate_batch_and_actual_sensor_dimension():
    tactile = np.zeros((2, 2, 2, 1), dtype=np.float32)
    baseline = np.zeros((2, 2, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="original_gate"):
        apply_intervention(
            "baseline_fixed",
            tactile,
            baseline,
            np.array([0.5], dtype=np.float32),
            tau=0.4,
            temperature=0.1,
            sensor_count=2,
        )
    with pytest.raises(ValueError, match="sensor index out of range"):
        apply_intervention(
            "drop_sensor_2",
            tactile,
            baseline,
            np.array([0.5, 0.5], dtype=np.float32),
            tau=0.4,
            temperature=0.1,
        )


def test_intervention_module_import_does_not_load_train_frs_dependencies():
    script = "\n".join(
        (
            "import sys",
            "import modalities_eval.frs.interventions",
            "assert 'train_frs.utils.data' not in sys.modules",
        )
    )

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_local_numpy_helpers_match_tactile_change_and_gate_semantics():
    baseline = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    current = np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float32)

    change = tactile_change_from_tokens(current, baseline)
    gate = gate_weights_from_change(change, tau=1.0, temperature=0.5)

    np.testing.assert_allclose(change, np.array([1.0], dtype=np.float32))
    np.testing.assert_allclose(gate, np.array([0.5], dtype=np.float32))
    with pytest.raises(ValueError, match="temperature"):
        gate_weights_from_change(change, tau=1.0, temperature=0.0)


def test_interventions_reject_empty_tactile_windows():
    tactile = np.zeros((1, 0, 4, 2), dtype=np.float32)
    baseline = np.zeros((1, 4, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="time"):
        apply_intervention(
            "baseline_fixed",
            tactile,
            baseline,
            np.array([0.5], dtype=np.float32),
            tau=0.4,
            temperature=0.1,
        )


@pytest.mark.parametrize("value", ["0.0", "0.5", "1.0", "nan", "inf", "-inf", "-0.1", "1.1"])
def test_gate_interventions_are_unsupported(value):
    tactile = np.zeros((1, 2, 4, 2), dtype=np.float32)
    baseline = np.zeros((1, 4, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="unsupported intervention"):
        apply_intervention(
            f"gate_{value}",
            tactile,
            baseline,
            np.array([0.5], dtype=np.float32),
            tau=0.4,
            temperature=0.1,
        )
