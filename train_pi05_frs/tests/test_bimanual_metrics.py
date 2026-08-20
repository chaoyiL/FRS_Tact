import numpy as np
import pytest

from train_pi05_frs.utils.bimanual_metrics import (
    bimanual_gate_region_counts,
    bimanual_quadrant_metrics,
)


def test_quadrants_keep_left_and_right_independent():
    result = bimanual_quadrant_metrics(
        mse_gt=np.asarray([[0.25, 4.0]]),
        mse_vla=np.asarray([[1.0, 0.04]]),
        mse_vla_gt=np.asarray([[1.0, 4.0]]),
        gate_weights=np.asarray([[0.8, 0.2]]),
        low_threshold=0.3,
        high_threshold=0.7,
    )
    assert result["high_low"]["n"] == 1
    assert result["high_low"]["left"]["relative_gt_error"] == pytest.approx(0.25)
    assert result["high_low"]["right"]["vla_preserve_ratio"] == pytest.approx(0.01)


def test_region_counts_include_mid_without_forcing_quadrant():
    counts = bimanual_gate_region_counts(
        np.asarray([[0.0, 0.0], [0.8, 0.2], [0.5, 0.9]]),
        low_threshold=0.3,
        high_threshold=0.7,
    )
    np.testing.assert_array_equal(counts, [[1, 0, 0], [0, 0, 1], [1, 0, 0]])
