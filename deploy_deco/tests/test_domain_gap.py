import numpy as np
import pytest

from deploy_deco.domain_gap import (
    basic_image_metrics,
    cosine_knn_distances,
    monotonic_state_match,
    relative_bimanual_state,
    right_world_z_delta,
    rotation_geodesic_degrees,
    standardized_state_summary,
)


def test_rotation_geodesic_uses_so3_distance() -> None:
    identity = np.zeros(3)
    quarter_turn_z = np.array([0.0, 0.0, np.pi / 2.0])

    assert rotation_geodesic_degrees(identity, identity) == pytest.approx(0.0)
    assert rotation_geodesic_degrees(identity, quarter_turn_z) == pytest.approx(90.0)


def test_cosine_knn_distances_returns_mean_of_k_nearest() -> None:
    reference = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    query = np.array([[1.0, 0.0], [0.0, 1.0]])

    distances = cosine_knn_distances(query, reference, k=2)

    assert distances.tolist() == pytest.approx([0.1, 0.2])


def test_monotonic_state_match_preserves_order() -> None:
    reference = np.arange(6, dtype=float)[:, None]
    query = np.array([[0.2], [2.8], [4.9]])

    indices, costs = monotonic_state_match(query, reference)

    assert indices.tolist() == [0, 3, 5]
    assert costs.tolist() == pytest.approx([0.2, 0.2, 0.1])


def test_monotonic_state_match_can_limit_reference_jump() -> None:
    reference = np.arange(7, dtype=float)[:, None]
    query = np.array([[0.0], [6.0]])

    indices, _ = monotonic_state_match(query, reference, max_reference_step=2)

    assert indices[1] - indices[0] <= 2


def test_standardized_state_summary_reports_rms_and_threshold() -> None:
    reference = np.array([[-1.0, 0.0], [1.0, 2.0]])
    query = np.array([[3.0, 1.0], [0.0, 5.0]])

    result = standardized_state_summary(
        reference,
        query,
        mean=np.array([0.0, 1.0]),
        std=np.array([1.0, 1.0]),
        threshold=3.0,
    )

    assert result["max_abs_zscore"] == pytest.approx(4.0)
    assert result["values_over_threshold"] == 1
    assert result["rms_zscore"].tolist() == pytest.approx(
        [np.sqrt(4.5), np.sqrt(8.0)]
    )
    assert result["wasserstein_by_dimension"].tolist() == pytest.approx([1.5, 2.0])


def test_basic_image_metrics_uses_rgb_and_normalized_luma() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = 255

    result = basic_image_metrics(image)

    assert result["red_mean"] == pytest.approx(1.0)
    assert result["green_mean"] == pytest.approx(0.0)
    assert result["blue_mean"] == pytest.approx(0.0)
    assert result["luma_mean"] == pytest.approx(0.2126, abs=1e-4)
    assert result["saturation_mean"] == pytest.approx(1.0)
    assert 0.0 <= result["edge_density"] <= 1.0


def test_relative_bimanual_state_is_zero_at_start_except_geometry_and_grippers() -> None:
    left = np.array([0.2, 0.1, 0.3, 0.0, 0.0, 0.0])
    right = np.array([-0.2, 0.1, 0.3, 0.0, 0.0, 0.0])

    state = relative_bimanual_state(
        left,
        right,
        left_gripper=0.12,
        right_gripper=0.07,
        left_start=left,
        right_start=right,
    )

    assert state.shape == (20,)
    assert state[:6].tolist() == pytest.approx(np.zeros(6))
    assert state[7:13].tolist() == pytest.approx(np.zeros(6))
    assert state[6] == pytest.approx(0.12)
    assert state[13] == pytest.approx(0.07)
    assert state[14:17].tolist() == pytest.approx([0.4, 0.0, 0.0])


def test_right_world_z_delta_composes_tcp_local_actions() -> None:
    action = np.zeros((2, 20), dtype=float)
    action[:, 10:13] = [0.0, 0.0, 0.01]
    action[:, 13:19] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    delta_mm = right_world_z_delta(
        action,
        right_pose=np.zeros(6),
    )

    assert delta_mm == pytest.approx(20.0)


def test_right_world_z_delta_rotates_local_translation_and_ignores_left_arm() -> None:
    action = np.zeros((1, 20), dtype=float)
    action[0, :3] = [0.0, 0.0, 10.0]
    action[0, 10:13] = [0.01, 0.0, 0.0]
    action[0, 13:19] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    right_pose = np.array([0.0, 0.0, 0.0, 0.0, np.pi / 2.0, 0.0])

    delta_mm = right_world_z_delta(action, right_pose=right_pose)

    assert delta_mm == pytest.approx(-10.0)
