"""Reusable metrics for DECO deployment domain-gap diagnostics."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import wasserstein_distance


def _pose_matrix(pose6d) -> np.ndarray:
    pose = np.asarray(pose6d, dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("pose must be finite xyz+rotation-vector with shape (6,)")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def _matrix_pose(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate(
        [matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_rotvec()]
    )


def relative_bimanual_state(
    left_pose,
    right_pose,
    *,
    left_gripper: float,
    right_gripper: float,
    left_start,
    right_start,
) -> np.ndarray:
    """Build the exact 20D relative-start state used by the robot server."""
    left = _pose_matrix(left_pose)
    right = _pose_matrix(right_pose)
    left_relative_start = _matrix_pose(np.linalg.inv(_pose_matrix(left_start)) @ left)
    right_relative_start = _matrix_pose(np.linalg.inv(_pose_matrix(right_start)) @ right)
    left_relative_right = _matrix_pose(np.linalg.inv(right) @ left)
    return np.concatenate(
        [
            left_relative_start,
            [float(left_gripper)],
            right_relative_start,
            [float(right_gripper)],
            left_relative_right,
        ]
    ).astype(np.float32)


def _normalize_rows(values, floor: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), floor)


def _tcp_delta_matrix(action_pose9) -> np.ndarray:
    action_pose9 = np.asarray(action_pose9, dtype=np.float64)
    if action_pose9.shape != (9,):
        raise ValueError("TCP delta pose must contain xyz plus two rotation columns")
    first = _normalize_rows(action_pose9[3:6][None])[0]
    second_raw = action_pose9[6:9]
    second = _normalize_rows((second_raw - np.dot(first, second_raw) * first)[None])[0]
    third = np.cross(first, second)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.stack((first, second, third), axis=-1)
    matrix[:3, 3] = action_pose9[:3]
    return matrix


def right_world_z_delta(actions, *, right_pose) -> float:
    """Compose right-arm local TCP deltas and return base/world z change in mm."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 20 or not np.isfinite(actions).all():
        raise ValueError("actions must be finite [T,20]")
    current = _pose_matrix(right_pose)
    start_z = float(current[2, 3])
    for action in actions:
        current = current @ _tcp_delta_matrix(action[10:19])
    return float((current[2, 3] - start_z) * 1000.0)


def rotation_geodesic_degrees(first_rotvec, second_rotvec) -> float:
    """Return the shortest SO(3) angular distance between two rotation vectors."""
    first = Rotation.from_rotvec(np.asarray(first_rotvec, dtype=np.float64))
    second = Rotation.from_rotvec(np.asarray(second_rotvec, dtype=np.float64))
    relative = first.inv() * second
    return float(np.degrees(np.linalg.norm(relative.as_rotvec())))


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("features must be a finite rank-2 array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("feature rows must have nonzero norm")
    return values / norms


def cosine_knn_distances(query, reference, *, k: int = 5) -> np.ndarray:
    """Mean cosine distance from each query to its k closest references."""
    query = _unit_rows(np.asarray(query))
    reference = _unit_rows(np.asarray(reference))
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(reference):
        raise ValueError(f"k must be in [1, {len(reference)}], got {k!r}")
    distances = 1.0 - query @ reference.T
    nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    return nearest.mean(axis=1)


def monotonic_state_match(
    query,
    reference,
    *,
    weights=None,
    max_reference_step: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Globally match query rows to nondecreasing reference indices."""
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference must be rank-2 with matching dimensions")
    if len(query) == 0 or len(reference) == 0:
        raise ValueError("query and reference must be nonempty")
    if not np.isfinite(query).all() or not np.isfinite(reference).all():
        raise ValueError("query and reference must be finite")
    if max_reference_step is not None and (
        isinstance(max_reference_step, bool)
        or not isinstance(max_reference_step, int)
        or max_reference_step < 0
    ):
        raise ValueError("max_reference_step must be a nonnegative integer or None")
    if weights is None:
        weights_array = np.ones(query.shape[1], dtype=np.float64)
    else:
        weights_array = np.asarray(weights, dtype=np.float64)
        if (
            weights_array.shape != (query.shape[1],)
            or not np.isfinite(weights_array).all()
            or np.any(weights_array < 0)
        ):
            raise ValueError("weights must be finite, nonnegative, and match dimensions")
    pair_cost = np.sqrt(
        np.sum((query[:, None, :] - reference[None, :, :]) ** 2 * weights_array, axis=2)
    )
    rows, columns = pair_cost.shape
    accumulated = np.full((rows, columns), np.inf, dtype=np.float64)
    parents = np.zeros((rows, columns), dtype=np.int64)
    accumulated[0] = pair_cost[0]
    for row in range(1, rows):
        for column in range(columns):
            first = 0 if max_reference_step is None else max(0, column - max_reference_step)
            candidates = accumulated[row - 1, first : column + 1]
            relative_best = int(np.argmin(candidates))
            best_index = first + relative_best
            best_value = candidates[relative_best]
            accumulated[row, column] = pair_cost[row, column] + best_value
            parents[row, column] = best_index
    indices = np.empty(rows, dtype=np.int64)
    indices[-1] = int(np.argmin(accumulated[-1]))
    for row in range(rows - 1, 0, -1):
        indices[row - 1] = parents[row, indices[row]]
    return indices, pair_cost[np.arange(rows), indices]


def standardized_state_summary(
    reference,
    query,
    *,
    mean,
    std,
    threshold: float = 3.0,
    std_floor: float = 1e-4,
) -> dict:
    """Summarize query displacement in the model's normalized state space."""
    reference = np.asarray(reference, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.maximum(np.asarray(std, dtype=np.float64), float(std_floor))
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1:] != query.shape[1:]:
        raise ValueError("reference and query must be rank-2 with matching dimensions")
    if mean.shape != (reference.shape[1],) or std.shape != mean.shape:
        raise ValueError("mean/std must match the state dimension")
    reference_z = (reference - mean) / std
    query_z = (query - mean) / std
    abs_query = np.abs(query_z)
    wasserstein = np.array(
        [
            wasserstein_distance(reference_z[:, index], query_z[:, index])
            for index in range(reference.shape[1])
        ],
        dtype=np.float64,
    )
    return {
        "query_zscore": query_z,
        "rms_zscore": np.sqrt(np.mean(query_z**2, axis=1)),
        "max_abs_zscore": float(abs_query.max()),
        "values_over_threshold": int(np.sum(abs_query > threshold)),
        "fraction_over_threshold": float(np.mean(abs_query > threshold)),
        "median_abs_by_dimension": np.median(abs_query, axis=0),
        "p95_abs_by_dimension": np.quantile(abs_query, 0.95, axis=0),
        "max_abs_by_dimension": abs_query.max(axis=0),
        "wasserstein_by_dimension": wasserstein,
    }


def basic_image_metrics(image) -> dict[str, float]:
    """Compute exposure, color, sharpness, and edge metrics for an RGB image."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be HWC RGB uint8")
    unit = image.astype(np.float32) / 255.0
    luma = 0.2126 * unit[..., 0] + 0.7152 * unit[..., 1] + 0.0722 * unit[..., 2]
    hsv = cv2.cvtColor(unit, cv2.COLOR_RGB2HSV)
    gray_u8 = np.round(luma * 255.0).astype(np.uint8)
    laplacian = cv2.Laplacian(gray_u8, cv2.CV_64F)
    sobel_x = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.Canny(gray_u8, 80, 160)
    fft = np.fft.fftshift(np.fft.fft2(luma))
    height, width = luma.shape
    radius = max(1, min(height, width) // 16)
    center_y, center_x = height // 2, width // 2
    low = np.zeros_like(luma, dtype=bool)
    low[
        max(0, center_y - radius) : min(height, center_y + radius + 1),
        max(0, center_x - radius) : min(width, center_x + radius + 1),
    ] = True
    power = np.abs(fft) ** 2
    high_frequency_ratio = float(power[~low].sum() / max(power.sum(), 1e-12))
    return {
        "red_mean": float(unit[..., 0].mean()),
        "green_mean": float(unit[..., 1].mean()),
        "blue_mean": float(unit[..., 2].mean()),
        "luma_mean": float(luma.mean()),
        "luma_std": float(luma.std()),
        "luma_p05": float(np.quantile(luma, 0.05)),
        "luma_p50": float(np.quantile(luma, 0.50)),
        "luma_p95": float(np.quantile(luma, 0.95)),
        "black_clip_fraction": float(np.mean(luma <= 2.0 / 255.0)),
        "white_clip_fraction": float(np.mean(luma >= 253.0 / 255.0)),
        "saturation_mean": float(hsv[..., 1].mean()),
        "laplacian_variance": float(laplacian.var()),
        "tenengrad_mean": float(np.mean(sobel_x**2 + sobel_y**2)),
        "edge_density": float(np.mean(edges > 0)),
        "fft_high_frequency_ratio": high_frequency_ratio,
    }
