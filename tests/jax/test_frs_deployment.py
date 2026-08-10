from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from deploy_smolvla import remote_client
from deploy_smolvla.bridge_client import RobotBridgeClient
from deploy_smolvla.frs_runtime import FRSRuntime, TactileHistory

ROOT = Path(__file__).resolve().parents[2]
FRS_CONFIG = ROOT / "deploy_smolvla" / "configs" / "deploy_frs.yaml"


def test_deploy_frs_config_uses_project_local_downloads() -> None:
    config = remote_client.load_config(FRS_CONFIG)
    root = ROOT / "checkpoints"
    assert Path(config["checkpoint"]) == root / "model/pick_tube_02_3w_jax"
    assert Path(config["frs"]["checkpoint"]) == root / "frs/frs_0809_02"
    assert Path(config["frs"]["tactile_encoder_checkpoint"]) == (
        root / "encoder/encoder_ckpt_0809"
    )


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


def test_frs_runtime_retains_vla_and_refined_normalized_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(FRSRuntime)
    runtime.config = SimpleNamespace(
        tactile_keys=("left",),
        gate_tau=0.2,
        gate_temperature=0.1,
        reverse_steps=2,
        reverse_solver="euler",
        decode_steps=2,
        decode_solver="euler",
        max_normalized_action_abs=8.0,
        max_normalized_delta_rms=4.0,
    )
    runtime.baseline = np.zeros((1, 1), dtype=np.float32)
    runtime.history = SimpleNamespace(
        append=lambda tokens: None,
        window_tokens=lambda: np.zeros((1, 1, 1), dtype=np.float32),
    )
    runtime._encode_observation = lambda observation: np.ones((1, 1), dtype=np.float32)
    runtime._eval_observation = lambda policy, observation, task: object()
    runtime.model = object()

    monkeypatch.setattr(
        "deploy_smolvla.frs_runtime.tactile_change_from_tokens",
        lambda current, baseline: jnp.asarray([0.25], dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "deploy_smolvla.frs_runtime.gate_weights_from_change",
        lambda change, *, tau, temperature: jnp.asarray([0.75], dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "deploy_smolvla.frs_runtime.reverse_integrate_actions",
        lambda *args, **kwargs: args[2],
    )
    monkeypatch.setattr(
        "deploy_smolvla.frs_runtime.decode_actions",
        lambda *args, **kwargs: args[1] + 1.0,
    )

    vla = jnp.asarray([[[1.0], [2.0]]], dtype=jnp.float32)
    refined = runtime.steer(object(), {}, "task", vla)

    np.testing.assert_array_equal(runtime.last_vla_normalized, np.asarray(vla))
    np.testing.assert_array_equal(runtime.last_frs_normalized, np.asarray(refined))


def test_action_trace_contains_complete_vla_frs_chunks_timestamps_and_diagnostics() -> None:
    class Preprocessor:
        @staticmethod
        def unnormalize_actions(actions):
            return np.asarray(actions) * 10.0

    policy = SimpleNamespace(preprocessor=Preprocessor())
    frs_runtime = SimpleNamespace(
        last_vla_normalized=np.asarray([[[1.0], [2.0]]], dtype=np.float32),
        last_frs_normalized=np.asarray([[[3.0], [4.0]]], dtype=np.float32),
        last_diagnostics=SimpleNamespace(
            tactile_change=0.25,
            gate_weight=0.75,
            delta_rms=2.0,
            max_normalized_action_abs=4.0,
        ),
    )

    trace = remote_client._build_action_trace(
        policy,
        frs_runtime,
        inference_wall_start_s=100.25,
        inference_wall_end_s=100.75,
    )

    assert trace["version"] == 1
    np.testing.assert_array_equal(trace["vla_normalized"], [[1.0], [2.0]])
    np.testing.assert_array_equal(trace["vla_action"], [[10.0], [20.0]])
    np.testing.assert_array_equal(trace["frs_normalized"], [[3.0], [4.0]])
    np.testing.assert_array_equal(trace["frs_action"], [[30.0], [40.0]])
    assert trace["inference_started_at"] == 100.25
    assert trace["inference_finished_at"] == 100.75
    assert trace["frs_diagnostics"] == {
        "tactile_change": 0.25,
        "gate_weight": 0.75,
        "delta_rms": 2.0,
        "max_normalized_action_abs": 4.0,
    }


def test_action_trace_failure_is_omitted_without_raising() -> None:
    class Preprocessor:
        @staticmethod
        def unnormalize_actions(actions):
            del actions
            raise RuntimeError("trace-only unnormalization failed")

    policy = SimpleNamespace(preprocessor=Preprocessor())
    frs_runtime = SimpleNamespace(
        last_vla_normalized=np.asarray([[[1.0]]], dtype=np.float32),
        last_frs_normalized=np.asarray([[[2.0]]], dtype=np.float32),
        last_diagnostics=SimpleNamespace(
            tactile_change=0.25,
            gate_weight=0.75,
            delta_rms=1.0,
            max_normalized_action_abs=2.0,
        ),
    )

    assert (
        remote_client._build_action_trace_or_none(
            policy,
            frs_runtime,
            inference_wall_start_s=100.25,
            inference_wall_end_s=100.75,
        )
        is None
    )


def test_bridge_send_action_keeps_legacy_payload_without_trace_and_adds_keyword_trace() -> None:
    bridge = object.__new__(RobotBridgeClient)
    messages: list[dict[str, object]] = []
    bridge._send = messages.append
    action = np.asarray([[1.0, 2.0]], dtype=np.float32)

    bridge.send_action(action, 7)
    bridge.send_action(action, 8, trace={"version": 1})

    assert messages[0] == {"type": "action", "obs_seq": 7, "action": action}
    assert messages[1] == {
        "type": "action",
        "obs_seq": 8,
        "action": action,
        "trace": {"version": 1},
    }


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


@pytest.mark.parametrize("version", [2, 3, 4, 5])
def test_frs_contract_accepts_supported_loss_versions(version: int) -> None:
    runtime, policy = _contract_runtime()
    runtime.metadata["extra_metadata"]["loss_weighting_version"] = version

    runtime._validate_contract(policy, source_sample_steps=10)


def test_frs_contract_rejects_different_source_sampling_steps() -> None:
    runtime, policy = _contract_runtime()

    with pytest.raises(ValueError, match="sample_steps"):
        runtime._validate_contract(policy, source_sample_steps=8)
