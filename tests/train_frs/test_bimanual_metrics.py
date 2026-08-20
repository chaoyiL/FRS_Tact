import numpy as np
import pytest

from train_smolvla_frs.utils.bimanual_metrics import (
    bimanual_gate_region_counts,
    bimanual_quadrant_metrics,
    flatten_bimanual_quadrant_metrics,
)


def test_quadrants_keep_high_left_and_low_right_independent():
    result = bimanual_quadrant_metrics(
        mse_gt=np.asarray([[0.25, 4.0], [1.0, 1.0]]),
        mse_vla=np.asarray([[1.0, 0.04], [0.5, 0.5]]),
        mse_vla_gt=np.asarray([[1.0, 4.0], [1.0, 1.0]]),
        gate_weights=np.asarray([[0.7, 0.3], [0.5, 0.5]]),
        low_threshold=0.3,
        high_threshold=0.7,
    )
    assert result["high_low"]["n"] == 1
    assert result["high_low"]["left"]["relative_gt_error"] == pytest.approx(0.25)
    assert result["high_low"]["right"]["vla_preserve_ratio"] == pytest.approx(0.01)
    assert result["low_high"]["n"] == 0


def test_quadrant_ratios_use_group_means_across_different_baseline_scales():
    result = bimanual_quadrant_metrics(
        mse_gt=np.asarray([[1.0, 3.0], [25.0, 30.0]]),
        mse_vla=np.asarray([[2.0, 6.0], [25.0, 15.0]]),
        mse_vla_gt=np.asarray([[2.0, 6.0], [100.0, 60.0]]),
        gate_weights=np.asarray([[0.8, 0.2], [0.9, 0.1]]),
        low_threshold=0.3,
        high_threshold=0.7,
    )

    left = result["high_low"]["left"]
    assert left["mse_gt"] == pytest.approx(13.0)
    assert left["mse_vla_gt"] == pytest.approx(51.0)
    assert left["relative_gt_error"] == pytest.approx(13.0 / 51.0)
    assert left["relative_gt_error"] != pytest.approx(np.mean([1.0 / 2.0, 25.0 / 100.0]))


def test_quadrant_ratios_clamp_zero_and_near_zero_group_baselines():
    result = bimanual_quadrant_metrics(
        mse_gt=np.asarray([[2e-8, 4e-8], [4e-8, 8e-8]]),
        mse_vla=np.asarray([[1e-8, 2e-8], [3e-8, 6e-8]]),
        mse_vla_gt=np.asarray([[0.0, 0.0], [1e-12, 2e-12]]),
        gate_weights=np.asarray([[0.8, 0.2], [0.9, 0.1]]),
        low_threshold=0.3,
        high_threshold=0.7,
    )

    left = result["high_low"]["left"]
    assert left["mse_gt"] == pytest.approx(3e-8)
    assert left["mse_vla"] == pytest.approx(2e-8)
    assert left["mse_vla_gt"] == pytest.approx(5e-13)
    assert left["relative_gt_error"] == pytest.approx(3.0)
    assert left["vla_preserve_ratio"] == pytest.approx(2.0)


def test_joint_region_counts_include_mid_without_forcing_a_quadrant():
    counts = bimanual_gate_region_counts(
        np.asarray([[0.0, 0.0], [0.8, 0.2], [0.5, 0.9]]),
        low_threshold=0.3,
        high_threshold=0.7,
    )
    np.testing.assert_array_equal(counts, [[1, 0, 0], [0, 0, 1], [1, 0, 0]])


def test_quadrant_thresholds_are_inclusive_and_empty_values_are_nan():
    result = bimanual_quadrant_metrics(
        mse_gt=np.ones((3, 2)),
        mse_vla=np.ones((3, 2)),
        mse_vla_gt=np.ones((3, 2)),
        gate_weights=np.asarray([[0.3, 0.7], [0.7, 0.3], [0.5, 0.5]]),
        low_threshold=0.3,
        high_threshold=0.7,
    )
    assert result["low_high"]["n"] == 1
    assert result["high_low"]["n"] == 1
    assert result["low_low"]["n"] == 0
    assert np.isnan(result["low_low"]["left"]["mse_gt"])
    assert np.isnan(result["high_high"]["right"]["rank_satisfied_frac"])


def test_quadrant_metrics_reject_shape_mismatch_and_nonfinite_inputs():
    kwargs = {
        "mse_gt": np.ones((2, 2)),
        "mse_vla": np.ones((2, 2)),
        "mse_vla_gt": np.ones((2, 2)),
        "gate_weights": np.ones((2, 2)),
        "low_threshold": 0.3,
        "high_threshold": 0.7,
    }
    kwargs["mse_vla"] = np.ones((2, 1))
    with pytest.raises(ValueError, match=r"shape \[N, 2\]"):
        bimanual_quadrant_metrics(**kwargs)
    kwargs["mse_vla"] = np.full((2, 2), np.nan)
    with pytest.raises(ValueError, match="matching finite values"):
        bimanual_quadrant_metrics(**kwargs)


def test_flatten_quadrant_metrics_uses_stable_json_friendly_keys():
    metrics = bimanual_quadrant_metrics(
        mse_gt=np.asarray([[1.0, 2.0]]),
        mse_vla=np.asarray([[3.0, 4.0]]),
        mse_vla_gt=np.asarray([[2.0, 8.0]]),
        gate_weights=np.asarray([[0.8, 0.2]]),
        low_threshold=0.3,
        high_threshold=0.7,
    )
    flat = flatten_bimanual_quadrant_metrics(metrics, prefix="val_quadrant")
    assert flat["val_quadrant_high_low_n"] == 1
    assert flat["val_quadrant_high_low_vla_preserve_ratio_right"] == pytest.approx(0.5)
    assert isinstance(flat["val_quadrant_low_high_n"], int)


@pytest.mark.parametrize(
    ("low_threshold", "high_threshold"),
    [
        (0.3, 0.3),
        (0.7, 0.3),
        (np.nan, 0.7),
        (0.3, np.inf),
        (-0.1, 0.7),
        (0.3, 1.1),
    ],
)
def test_threshold_validation_is_shared_and_requires_strict_unit_interval(
    low_threshold, high_threshold
):
    metric_kwargs = {
        "mse_gt": np.ones((1, 2)),
        "mse_vla": np.ones((1, 2)),
        "mse_vla_gt": np.ones((1, 2)),
        "gate_weights": np.ones((1, 2)),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
    }
    with pytest.raises(ValueError, match="threshold"):
        bimanual_quadrant_metrics(**metric_kwargs)
    with pytest.raises(ValueError, match="threshold"):
        bimanual_gate_region_counts(
            metric_kwargs["gate_weights"],
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )


def test_ranking_margin_is_added_to_gt_error_before_satisfaction_check():
    kwargs = {
        "mse_gt": np.asarray([[1.0, 1.0], [1.0, 1.0]]),
        "mse_vla": np.asarray([[1.05, 1.05], [1.2, 1.2]]),
        "mse_vla_gt": np.ones((2, 2)),
        "gate_weights": np.asarray([[0.1, 0.1], [0.2, 0.2]]),
        "low_threshold": 0.3,
        "high_threshold": 0.7,
    }
    no_margin = bimanual_quadrant_metrics(**kwargs, ranking_margin=0.0)
    margin = bimanual_quadrant_metrics(**kwargs, ranking_margin=0.1)
    assert no_margin["low_low"]["left"]["rank_satisfied_frac"] == pytest.approx(1.0)
    assert margin["low_low"]["left"]["rank_satisfied_frac"] == pytest.approx(0.5)
