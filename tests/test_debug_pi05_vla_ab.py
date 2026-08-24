from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools.debug_pi05_vla_ab import (
    IMAGE_FILE_BY_KEY,
    ROOT,
    _paired_predictions,
    compare_artifacts,
    load_saved_observation,
    write_artifact,
)


def _metadata() -> dict[str, object]:
    return {
        "config_path": "/tmp/deploy_pi05_frs.yaml",
        "observation_dir": "/tmp/step_000001",
        "checkpoint": "/tmp/pi05",
        "assets_dir": "/tmp/pi05/assets",
        "asset_id": "pick_tube_all",
        "prompt": "pick up the tube",
        "seed": 0,
        "num_steps": 10,
        "warmup_runs": 1,
    }


def _actions() -> tuple[np.ndarray, np.ndarray]:
    normalized = np.zeros((1, 5, 20), dtype=np.float32)
    robot = np.zeros((1, 5, 20), dtype=np.float32)
    robot[..., [9, 19]] = 0.12
    robot[0, 3:, 9] = 0.08
    return normalized, robot


def test_module_bootstraps_pi05_source_import_path() -> None:
    assert str(ROOT) in sys.path
    assert str(ROOT / "deploy_pi05/src") in sys.path


def test_paired_predictions_replays_identical_policy_rng_for_frs_source() -> None:
    class FakePolicy:
        def __init__(self) -> None:
            self._rng = None
            self._rng_seed = None

        def predict_action_chunk(self, observation, prompt, *, seed, num_steps):
            del observation, prompt, num_steps
            if self._rng is None:
                self._rng = seed
                self._rng_seed = seed
            self._rng += 1
            return np.full((1, 5, 20), self._rng, dtype=np.float32)

        @staticmethod
        def unnormalize_actions(actions):
            return np.asarray(actions) * 2.0

    class FakeRuntime:
        def __init__(self, policy) -> None:
            self.policy = policy

        def reset_episode(self, observation) -> None:
            del observation

        def warmup(self, observation, prompt, *, seed, sample_steps) -> None:
            self.policy.predict_action_chunk(
                observation, prompt, seed=seed, num_steps=sample_steps
            )

        def begin_chunk(self, chunk_id, observation, prompt, *, seed, num_steps):
            del chunk_id
            normalized = self.policy.predict_action_chunk(
                observation, prompt, seed=seed, num_steps=num_steps
            )
            return SimpleNamespace(
                action_vla_normalized=normalized,
                action_vla=self.policy.unnormalize_actions(normalized),
            )

    policy = FakePolicy()
    runtime = FakeRuntime(policy)

    result = _paired_predictions(
        policy,
        runtime,
        {"observation.state": np.zeros(20, dtype=np.float32)},
        prompt="pick",
        seed=0,
        num_steps=10,
        warmup_runs=1,
    )

    np.testing.assert_array_equal(result["direct_normalized"], result["frs_normalized"])
    np.testing.assert_array_equal(result["direct_robot"], result["frs_robot"])


def test_load_saved_observation_restores_rgb_state_and_tactile_keys(tmp_path: Path) -> None:
    state = np.arange(20, dtype=np.float32)
    np.save(tmp_path / "observation_state.npy", state)
    for index, filename in enumerate(IMAGE_FILE_BY_KEY.values()):
        rgb = np.zeros((8, 9, 3), dtype=np.uint8)
        rgb[..., index % 3] = 220
        cv2.imwrite(str(tmp_path / filename), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    observation = load_saved_observation(tmp_path)

    np.testing.assert_array_equal(observation["observation.state"], state)
    assert set(observation) == {"observation.state", *IMAGE_FILE_BY_KEY}
    for index, key in enumerate(IMAGE_FILE_BY_KEY):
        image = observation[key]
        assert image.shape == (8, 9, 3)
        assert image.dtype == np.uint8
        assert float(image[..., index % 3].mean()) > 200


def test_compare_artifacts_reports_exact_match_and_close_indices(tmp_path: Path) -> None:
    normalized, robot = _actions()
    direct = tmp_path / "direct.npz"
    frs = tmp_path / "frs.npz"
    write_artifact(
        direct,
        mode="direct",
        normalized=normalized,
        robot_action=robot,
        metadata=_metadata(),
    )
    write_artifact(
        frs,
        mode="frs",
        normalized=normalized,
        robot_action=robot,
        metadata=_metadata(),
    )

    report = compare_artifacts(direct, frs, tolerance=1e-6)

    assert report["passed"] is True
    assert report["max_abs_diff"] == {
        "normalized_all": 0.0,
        "normalized_grippers": 0.0,
        "robot_action_all": 0.0,
        "robot_grippers": 0.0,
    }
    assert report["first_close_index"]["direct"] == {"left": 3, "right": None}
    assert report["first_close_index"]["frs"] == {"left": 3, "right": None}


def test_compare_artifacts_detects_gripper_mismatch(tmp_path: Path) -> None:
    normalized, robot = _actions()
    changed_normalized = normalized.copy()
    changed_robot = robot.copy()
    changed_normalized[0, 2, 19] = 0.02
    changed_robot[0, 2, 19] = 0.02
    direct = tmp_path / "direct.npz"
    frs = tmp_path / "frs.npz"
    write_artifact(
        direct,
        mode="direct",
        normalized=normalized,
        robot_action=robot,
        metadata=_metadata(),
    )
    write_artifact(
        frs,
        mode="frs",
        normalized=changed_normalized,
        robot_action=changed_robot,
        metadata=_metadata(),
    )

    report = compare_artifacts(direct, frs, tolerance=1e-6)

    assert report["passed"] is False
    assert report["max_abs_diff"]["normalized_grippers"] == pytest.approx(0.02)
    assert report["max_abs_diff"]["robot_grippers"] == pytest.approx(0.1)


def test_compare_artifacts_rejects_different_inference_metadata(tmp_path: Path) -> None:
    normalized, robot = _actions()
    direct = tmp_path / "direct.npz"
    frs = tmp_path / "frs.npz"
    write_artifact(
        direct,
        mode="direct",
        normalized=normalized,
        robot_action=robot,
        metadata=_metadata(),
    )
    changed = _metadata()
    changed["prompt"] = "different prompt"
    write_artifact(
        frs,
        mode="frs",
        normalized=normalized,
        robot_action=robot,
        metadata=changed,
    )

    with pytest.raises(ValueError, match="prompt"):
        compare_artifacts(direct, frs, tolerance=1e-6)
