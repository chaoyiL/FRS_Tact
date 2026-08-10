from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from deploy_smolvla import remote_client
from deploy_smolvla.frs_runtime import FRSRuntime, TactileHistory

ROOT = Path(__file__).resolve().parents[2]
FRS_CONFIG = ROOT / "deploy_smolvla" / "configs" / "deploy_frs.yaml"


def test_deploy_frs_config_preserves_training_time_scale() -> None:
    config = remote_client.load_config(FRS_CONFIG)

    assert config["observation"]["data_type"] == "vitac"
    assert config["control"]["control_frequency"] == 30.0
    assert config["control"]["steps_per_inference"] == 1
    assert config["frs"]["history_stride"] == 3
    assert config["frs"]["reverse_solver"] == "slerpflow"
    assert config["frs"]["decode_solver"] == "euler"


def test_tactile_history_matches_clamped_training_indices() -> None:
    history = TactileHistory(window=4, stride=2, token_shape=(1, 1))
    history.reset(np.asarray([[0.0]], dtype=np.float32))
    for value in range(1, 7):
        history.append(np.asarray([[value]], dtype=np.float32))

    # Current=6 with offsets [6, 4, 2, 0], returned oldest -> newest.
    np.testing.assert_array_equal(
        history.window_tokens()[:, 0, 0],
        np.asarray([0.0, 2.0, 4.0, 6.0], dtype=np.float32),
    )


def test_tactile_history_clamps_short_episode_to_first_frame() -> None:
    history = TactileHistory(window=4, stride=3, token_shape=(1, 1))
    history.reset(np.asarray([[10.0]], dtype=np.float32))
    history.append(np.asarray([[11.0]], dtype=np.float32))

    np.testing.assert_array_equal(
        history.window_tokens()[:, 0, 0],
        np.asarray([10.0, 10.0, 10.0, 11.0], dtype=np.float32),
    )


def test_predict_chunk_unnormalizes_frs_output() -> None:
    class Preprocessor:
        @staticmethod
        def unnormalize_actions(actions):
            return actions * 10.0

    class Policy:
        config = SimpleNamespace(chunk_size=2, action_dim=1)
        preprocessor = Preprocessor()

        @staticmethod
        def predict_action_chunk(*args, **kwargs):
            del args, kwargs
            return jnp.ones((1, 2, 1), dtype=jnp.float32)

    class FRS:
        @staticmethod
        def steer(policy, observation, task, actions, *, update_history):
            del policy, observation, task
            assert update_history is True
            return actions + 2.0

    action, normalized = remote_client._predict_chunk(
        Policy(),
        {},
        "task",
        seed=0,
        jit=False,
        num_steps=10,
        previous_chunk=None,
        inference_delay=None,
        execution_horizon=None,
        frs_runtime=FRS(),  # type: ignore[arg-type]
    )

    np.testing.assert_array_equal(normalized, np.full((2, 1), 3.0, dtype=np.float32))
    np.testing.assert_array_equal(action, np.full((2, 1), 30.0, dtype=np.float32))


def _contract_runtime() -> tuple[FRSRuntime, SimpleNamespace]:
    runtime = object.__new__(FRSRuntime)
    runtime.config = SimpleNamespace(
        tactile_keys=("left", "right", "left_1", "right_1"),
        history_stride=3,
        gate_tau=0.2,
        gate_temperature=0.1,
        decode_steps=10,
        decode_solver="euler",
        reverse_steps=20,
        reverse_solver="slerpflow",
        verify_source_checkpoint_fingerprint=False,
    )
    runtime.embedding_dim = 512
    runtime.model = SimpleNamespace(
        config=SimpleNamespace(
            action_dim=20,
            action_horizon=10,
            num_tactile_tokens=4,
            resnet_embedding_dim=512,
            tactile_window=10,
            gate_conditioning=True,
        )
    )
    runtime.metadata = {
        "extra_metadata": {
            "loss_mode": "gated",
            "loss_weighting_version": 4,
            "rank_low_gate_threshold": 0.3,
            "rank_high_gate_threshold": 0.7,
            "history_stride": 3,
            "tactile_window": 10,
            "gate_conditioning": True,
            "gate_tau": 0.2,
            "gate_temperature": 0.1,
            "validation_steps": 10,
            "validation_solver": "euler",
            "cache_configuration": {
                "model_sample_steps": 10,
                "reverse_steps": 20,
                "reverse_solver": "slerpflow",
                "normalization_source": "checkpoint",
                "reverse_integration_version": 1,
            },
        }
    }
    policy = SimpleNamespace(
        config=SimpleNamespace(action_dim=20, chunk_size=10),
        checkpoint=Path("unused"),
    )
    return runtime, policy


def test_frs_contract_accepts_matching_training_metadata() -> None:
    runtime, policy = _contract_runtime()

    runtime._validate_contract(policy, source_sample_steps=10)


@pytest.mark.parametrize("version", [2, 3, 4])
def test_frs_contract_accepts_supported_loss_versions(version: int) -> None:
    runtime, policy = _contract_runtime()
    runtime.metadata["extra_metadata"]["loss_weighting_version"] = version

    runtime._validate_contract(policy, source_sample_steps=10)


def test_frs_contract_rejects_different_source_sampling_steps() -> None:
    runtime, policy = _contract_runtime()

    with pytest.raises(ValueError, match="sample_steps"):
        runtime._validate_contract(policy, source_sample_steps=8)
