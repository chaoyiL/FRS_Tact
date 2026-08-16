from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import yaml

from deploy_smolvla import direct_decoder as direct_decoder_module
from deploy_smolvla import remote_client
from deploy_smolvla.direct_decoder import (
    DIRECT_TACTILE_KEYS,
    DirectChunkReady,
    DirectDecoderRuntime,
    DirectDecoderSteeringRuntime,
    DirectSteerDiagnostics,
    DirectSteerResult,
    DirectTactileActionDecoder,
)
from deploy_smolvla.frs_protocol import (
    FRSChunkEnd,
    FRSChunkStart,
    FRSSteerAck,
    FRSSteerRequest,
)

ROOT = Path(__file__).resolve().parents[2]
ABLATION = ROOT / "checkpoints" / "ablation"


def _steering_observation(value: int) -> dict[str, np.ndarray]:
    return {
        key: np.full((1, 1, 3), value, dtype=np.uint8)
        for key in DIRECT_TACTILE_KEYS
    }


class _SteeringPreprocessor:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        self.calls.append(np.array(actions, copy=True))
        return np.asarray(actions, dtype=np.float32) * 10.0


class _SteeringPolicy:
    def __init__(self, *, chunk_size: int = 20, action_dim: int = 20) -> None:
        self.config = SimpleNamespace(chunk_size=chunk_size, action_dim=action_dim)
        self.preprocessor = _SteeringPreprocessor()
        self.predict_calls: list[SimpleNamespace] = []
        self.coarse = np.arange(
            chunk_size * action_dim, dtype=np.float32
        ).reshape(1, chunk_size, action_dim)

    def predict_action_chunk(self, observation, task, **kwargs):
        self.predict_calls.append(
            SimpleNamespace(
                observation_value=int(observation[DIRECT_TACTILE_KEYS[0]][0, 0, 0]),
                task=task,
                kwargs=kwargs,
            )
        )
        return jnp.asarray(self.coarse)


class _SteeringDecoder:
    tactile_keys = DIRECT_TACTILE_KEYS
    fixed_noise_jax = jnp.zeros((1, 20, 32), dtype=jnp.float32)

    def __init__(self) -> None:
        self.refine_calls: list[SimpleNamespace] = []
        self.returned: np.ndarray | None = None
        self.reset_calls = 0
        self.last_vla_normalized: np.ndarray | None = None
        self.last_direct_normalized: np.ndarray | None = None

    def refine(self, coarse_normalized: np.ndarray, observation) -> np.ndarray:
        self.refine_calls.append(
            SimpleNamespace(
                coarse_normalized=np.array(coarse_normalized, copy=True),
                observation_value=int(
                    observation[DIRECT_TACTILE_KEYS[0]][0, 0, 0]
                ),
            )
        )
        self.returned = (
            np.arange(400, dtype=np.float32).reshape(1, 20, 20)
            + self.refine_calls[-1].observation_value
        )
        self.last_vla_normalized = np.array(coarse_normalized, copy=True)
        self.last_direct_normalized = np.array(self.returned, copy=True)
        return self.returned

    def reset(self) -> None:
        self.reset_calls += 1
        self.last_vla_normalized = None
        self.last_direct_normalized = None


def _steering_runtime() -> tuple[
    DirectDecoderSteeringRuntime, _SteeringPolicy, _SteeringDecoder
]:
    policy = _SteeringPolicy()
    decoder = _SteeringDecoder()
    return DirectDecoderSteeringRuntime(policy=policy, decoder=decoder), policy, decoder


def test_direct_steering_runs_vla_once_and_refines_current_observation_per_action() -> None:
    steering, policy, decoder = _steering_runtime()

    ready = steering.begin_chunk(
        3, _steering_observation(10), "pick", seed=7, jit=False, num_steps=4
    )
    first = steering.steer_action(3, 11, _steering_observation(20), 0)
    second = steering.steer_action(3, 12, _steering_observation(30), 1)

    assert len(policy.predict_calls) == 1
    assert policy.predict_calls[0].kwargs == {
        "seed": 7,
        "noise": decoder.fixed_noise_jax,
        "jit": False,
        "normalized": True,
        "num_steps": 4,
        "previous_chunk": None,
        "inference_delay": None,
        "execution_horizon": None,
    }
    assert [call.observation_value for call in decoder.refine_calls] == [20, 30]
    np.testing.assert_array_equal(first.selected_normalized, first.decoded_normalized[0, 0])
    np.testing.assert_array_equal(second.selected_normalized, second.decoded_normalized[0, 1])
    assert ready.chunk_id == first.chunk_id == second.chunk_id == 3
    assert len(policy.preprocessor.calls) == 3
    assert first.selected_action.ndim == 1
    assert np.isfinite(first.selected_action).all()


def test_direct_steering_rejects_incompatible_policy_and_fixed_noise_contracts() -> None:
    decoder = _SteeringDecoder()
    with pytest.raises(ValueError, match="chunk_size=20 and action_dim=20"):
        DirectDecoderSteeringRuntime(
            policy=_SteeringPolicy(chunk_size=3, action_dim=2), decoder=decoder
        )

    decoder.fixed_noise_jax = jnp.zeros((1, 20, 31), dtype=jnp.float32)
    with pytest.raises(ValueError, match="fixed noise must be shaped \\[1,20,32\\]"):
        DirectDecoderSteeringRuntime(policy=_SteeringPolicy(), decoder=decoder)


def test_direct_steering_rejects_invalid_chunk_lifecycle_and_indices() -> None:
    steering, _, _ = _steering_runtime()

    with pytest.raises(ValueError):
        steering.steer_action(3, 10, _steering_observation(10), 0)

    steering.begin_chunk(3, _steering_observation(10), "pick", seed=7, jit=False, num_steps=4)
    with pytest.raises(ValueError):
        steering.begin_chunk(4, _steering_observation(10), "pick", seed=7, jit=False, num_steps=4)
    with pytest.raises(ValueError):
        steering.steer_action(4, 10, _steering_observation(10), 0)
    for index in (True, 1.5, -1, 20):
        with pytest.raises(ValueError):
            steering.steer_action(3, 10, _steering_observation(10), index)

    steering.steer_action(3, 11, _steering_observation(10), 1)
    for index in (0, 1):
        with pytest.raises(ValueError):
            steering.steer_action(3, 12 + index, _steering_observation(10), index)
    with pytest.raises(ValueError):
        steering.end_chunk(4)


def test_direct_steering_caches_identical_requests_and_rejects_conflicts() -> None:
    steering, _, decoder = _steering_runtime()
    steering.begin_chunk(3, _steering_observation(10), "pick", seed=7, jit=False, num_steps=4)

    first = steering.steer_action(3, 11, _steering_observation(20), 0)
    same_tactile = _steering_observation(20)
    same_tactile["non_tactile"] = np.array([999], dtype=np.int64)
    assert steering.steer_action(3, 11, same_tactile, 0) is first
    assert len(decoder.refine_calls) == 1

    with pytest.raises(ValueError):
        steering.steer_action(3, 11, _steering_observation(21), 0)
    with pytest.raises(ValueError):
        steering.steer_action(3, 11, _steering_observation(20), 1)


def test_direct_steering_end_chunk_and_reset_clear_chunk_local_state() -> None:
    steering, policy, _ = _steering_runtime()
    steering.begin_chunk(3, _steering_observation(10), "pick", seed=7, jit=False, num_steps=4)
    steering.steer_action(3, 11, _steering_observation(20), 0)
    steering.end_chunk(3)

    steering.begin_chunk(4, _steering_observation(30), "place", seed=8, jit=True, num_steps=5)
    steering.steer_action(4, 11, _steering_observation(40), 0)
    steering.reset()
    steering.begin_chunk(5, _steering_observation(50), "place", seed=9, jit=True, num_steps=6)

    assert len(policy.predict_calls) == 3


def test_direct_steering_reset_and_end_chunk_reset_decoder_snapshots() -> None:
    steering, policy, decoder = _steering_runtime()

    steering.begin_chunk(3, _steering_observation(10), "pick", seed=7, jit=False, num_steps=4)
    steering.steer_action(3, 11, _steering_observation(20), 0)
    assert decoder.last_vla_normalized is not None
    assert decoder.last_direct_normalized is not None

    steering.end_chunk(3)

    assert decoder.reset_calls == 1
    assert decoder.last_vla_normalized is None
    assert decoder.last_direct_normalized is None

    steering.begin_chunk(4, _steering_observation(30), "place", seed=8, jit=True, num_steps=5)
    steering.steer_action(4, 12, _steering_observation(40), 0)
    steering.reset()

    assert decoder.reset_calls == 2
    assert decoder.last_vla_normalized is None
    assert decoder.last_direct_normalized is None

    steering.begin_chunk(5, _steering_observation(50), "place", seed=9, jit=True, num_steps=6)
    assert len(policy.predict_calls) == 3


def test_direct_steering_uses_immutable_snapshot_copies() -> None:
    steering, policy, decoder = _steering_runtime()
    ready = steering.begin_chunk(
        3, _steering_observation(10), "pick", seed=7, jit=False, num_steps=4
    )
    result = steering.steer_action(3, 11, _steering_observation(20), 0)

    policy.coarse[:] = -100.0
    decoder.refine_calls[0].coarse_normalized[:] = -200.0
    assert decoder.returned is not None
    decoder.returned[:] = -300.0

    np.testing.assert_array_equal(
        ready.action_vla_normalized,
        np.arange(400, dtype=np.float32).reshape(1, 20, 20),
    )
    np.testing.assert_array_equal(
        result.decoded_normalized,
        np.arange(400, dtype=np.float32).reshape(1, 20, 20) + 20,
    )
    with pytest.raises(ValueError):
        result.selected_action[0] = 0.0


def _protocol_observation(value: int) -> dict[str, np.ndarray]:
    return {
        "observation.state": np.full((3,), value, dtype=np.float32),
        "camera": np.full((2, 2, 3), value, dtype=np.uint8),
        "tactile": np.full((2, 2, 3), value, dtype=np.uint8),
    }


class _ProtocolBridge:
    def __init__(self, inbound: list[object], events: list[tuple[Any, ...]]) -> None:
        self.inbound = deque(inbound)
        self.events = events
        self.sent: list[tuple[Any, ...]] = []

    def receive_frs_message(self, timeout: float) -> object:
        del timeout
        message = self.inbound.popleft()
        self.events.append(("receive", type(message).__name__))
        return message

    def send_frs_chunk_ready(
        self,
        obs_seq: int,
        chunk_id: int,
        prediction_trace: dict[str, Any] | None = None,
    ) -> None:
        sent = ("ready", obs_seq, chunk_id, prediction_trace)
        self.events.append(sent)
        self.sent.append(sent)

    def send_frs_steer_action(
        self,
        chunk_id: int,
        request_id: int,
        action_index: int,
        action: np.ndarray,
        *,
        trace: dict[str, Any] | None = None,
    ) -> None:
        sent = (
            "action",
            chunk_id,
            request_id,
            action_index,
            np.array(action, copy=True),
            trace,
        )
        self.events.append(sent)
        self.sent.append(sent)


class _DirectProtocolRuntime:
    tactile_keys = ("tactile",)

    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events
        self.policy = SimpleNamespace(
            config=SimpleNamespace(action_dim=2, chunk_size=3)
        )
        self.coarse = np.arange(6, dtype=np.float32).reshape(1, 3, 2)
        self.request_observations: list[int] = []
        self.results: list[DirectSteerResult] = []
        self.mismatch_result = False

    def begin_chunk(
        self,
        chunk_id: int,
        initial_observation: dict[str, Any],
        task: str,
        *,
        seed: int,
        jit: bool,
        num_steps: int | None,
    ) -> DirectChunkReady:
        del initial_observation, task, seed, jit, num_steps
        self.events.append(("begin", chunk_id))
        return DirectChunkReady(
            chunk_id=chunk_id,
            action_vla_normalized=self.coarse,
            action_vla=self.coarse + 100.0,
            prediction_started_at=1.0,
            prediction_finished_at=2.0,
        )

    def steer_action(
        self,
        chunk_id: int,
        request_id: int,
        observation: dict[str, Any],
        action_index: int,
    ) -> DirectSteerResult:
        value = int(observation["tactile"][0, 0, 0])
        self.request_observations.append(value)
        self.events.append(("steer", chunk_id, request_id, action_index))
        decoded = self.coarse + value
        selected_normalized = decoded[0, action_index]
        result = DirectSteerResult(
            chunk_id=chunk_id,
            request_id=request_id + int(self.mismatch_result),
            action_index=action_index,
            action_vla_normalized=self.coarse,
            decoded_normalized=decoded,
            selected_normalized=selected_normalized,
            selected_action=selected_normalized + 1000.0,
            diagnostics=DirectSteerDiagnostics(
                delta_rms=float(value),
                max_normalized_action_abs=float(np.max(np.abs(decoded))),
            ),
            decode_started_at=3.0 + request_id,
            decode_finished_at=4.0 + request_id,
        )
        self.results.append(result)
        return result

    def end_chunk(self, chunk_id: int) -> None:
        self.events.append(("end", chunk_id))


class _ProtocolSaver:
    def submit(self, iteration: int, obs_seq: int, observation: object) -> None:
        del iteration, obs_seq, observation


def _protocol_start(chunk_id: int = 1) -> FRSChunkStart:
    return FRSChunkStart(
        obs_seq=9,
        chunk_id=chunk_id,
        observation=_protocol_observation(1),
        observation_timestamp=100.0,
        control_dt=0.05,
        action_horizon=3,
        execution_mode="block",
        action_timestamps=None,
        nominal_chunk_end=None,
    )


def _protocol_request(request_id: int, action_index: int) -> FRSSteerRequest:
    return FRSSteerRequest(
        chunk_id=1,
        request_id=request_id,
        action_index=action_index,
        target_timestamp=123.5 + action_index,
        protection_applied=bool(action_index),
        observation=_protocol_observation(request_id),
    )


def _protocol_ack(
    request_id: int,
    action_index: int,
    *,
    chunk_id: int = 1,
) -> FRSSteerAck:
    return FRSSteerAck(
        chunk_id=chunk_id,
        request_id=request_id,
        action_index=action_index,
        status="scheduled",
        scheduled_timestamp=124.0,
    )


def _protocol_end() -> FRSChunkEnd:
    return FRSChunkEnd(
        chunk_id=1,
        reason="exhausted",
        scheduled_count=2,
        stale_count=0,
    )


def _run_direct_protocol(
    bridge: _ProtocolBridge,
    runtime: _DirectProtocolRuntime,
) -> None:
    remote_client._run_direct_decoder_protocol(
        bridge,
        runtime,
        task="pick",
        state_dim=3,
        image_keys=("camera", "tactile"),
        empty_cameras=0,
        observation_timeout_s=10.0,
        action_ack_timeout_s=2.0,
        seed=7,
        jit=True,
        num_steps=4,
        max_chunks=1,
        observation_saver=_ProtocolSaver(),
    )


def test_direct_protocol_orders_requests_and_sends_selected_rows() -> None:
    events: list[tuple[Any, ...]] = []
    bridge = _ProtocolBridge(
        [
            _protocol_start(),
            _protocol_request(4, 0),
            _protocol_ack(4, 0),
            _protocol_request(5, 1),
            _protocol_ack(5, 1),
            _protocol_end(),
        ],
        events,
    )
    runtime = _DirectProtocolRuntime(events)

    _run_direct_protocol(bridge, runtime)

    assert [event[0] for event in events] == [
        "receive",
        "begin",
        "ready",
        "receive",
        "steer",
        "action",
        "receive",
        "receive",
        "steer",
        "action",
        "receive",
        "receive",
        "end",
    ]
    actions = [message for message in bridge.sent if message[0] == "action"]
    np.testing.assert_array_equal(actions[0][4], runtime.results[0].selected_action)
    np.testing.assert_array_equal(actions[1][4], runtime.results[1].selected_action)
    np.testing.assert_array_equal(
        runtime.results[0].selected_normalized,
        runtime.results[0].decoded_normalized[0, 0],
    )
    np.testing.assert_array_equal(
        runtime.results[1].selected_normalized,
        runtime.results[1].decoded_normalized[0, 1],
    )
    assert runtime.request_observations == [4, 5]

    chunk_trace = bridge.sent[0][3]
    assert chunk_trace is not None
    assert set(chunk_trace) == {
        "version",
        "kind",
        "chunk_id",
        "action_vla_normalized",
        "action_vla",
        "x_base",
        "prediction_started_at",
        "prediction_finished_at",
    }
    assert chunk_trace["version"] == 2
    assert chunk_trace["kind"] == "frs_chunk"
    np.testing.assert_array_equal(
        chunk_trace["action_vla_normalized"], runtime.coarse
    )
    np.testing.assert_array_equal(chunk_trace["action_vla"], runtime.coarse + 100.0)
    np.testing.assert_array_equal(chunk_trace["x_base"], runtime.coarse)
    assert not chunk_trace["x_base"].flags.writeable

    steer_trace = actions[0][5]
    assert steer_trace is not None
    assert set(steer_trace) == {
        "version",
        "kind",
        "chunk_id",
        "request_id",
        "action_index",
        "target_timestamp",
        "protection_applied",
        "decoded_normalized",
        "selected_normalized",
        "selected_action",
        "tactile_sequence_length",
        "encode_started_at",
        "encode_finished_at",
        "decode_started_at",
        "decode_finished_at",
        "frs_diagnostics",
    }
    assert steer_trace["kind"] == "frs_steer"
    assert (
        steer_trace["chunk_id"],
        steer_trace["request_id"],
        steer_trace["action_index"],
    ) == (1, 4, 0)
    np.testing.assert_array_equal(
        steer_trace["decoded_normalized"], runtime.results[0].decoded_normalized
    )
    np.testing.assert_array_equal(
        steer_trace["selected_normalized"], runtime.results[0].selected_normalized
    )
    np.testing.assert_array_equal(
        steer_trace["selected_action"], runtime.results[0].selected_action
    )
    assert steer_trace["tactile_sequence_length"] == 1
    assert steer_trace["encode_started_at"] == runtime.results[0].decode_started_at
    assert steer_trace["encode_finished_at"] == runtime.results[0].decode_started_at
    assert steer_trace["decode_started_at"] == runtime.results[0].decode_started_at
    assert steer_trace["decode_finished_at"] == runtime.results[0].decode_finished_at
    assert steer_trace["frs_diagnostics"] == {
        "tactile_change": 0.0,
        "delta_rms": runtime.results[0].diagnostics.delta_rms,
        "max_normalized_action_abs": (
            runtime.results[0].diagnostics.max_normalized_action_abs
        ),
    }


def test_direct_protocol_checks_ack_identity() -> None:
    events: list[tuple[Any, ...]] = []
    bridge = _ProtocolBridge(
        [_protocol_start(), _protocol_request(4, 0), _protocol_ack(5, 0)],
        events,
    )

    with pytest.raises(RuntimeError, match="acknowledgement.*does not match"):
        _run_direct_protocol(bridge, _DirectProtocolRuntime(events))


def test_direct_protocol_rejects_mismatched_result_before_sending() -> None:
    events: list[tuple[Any, ...]] = []
    bridge = _ProtocolBridge([_protocol_start(), _protocol_request(4, 0)], events)
    runtime = _DirectProtocolRuntime(events)
    runtime.mismatch_result = True

    with pytest.raises(RuntimeError, match="steer result does not match"):
        _run_direct_protocol(bridge, runtime)

    assert [message for message in bridge.sent if message[0] == "action"] == []


def test_direct_protocol_trace_builder_exception_is_fail_open(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    bridge = _ProtocolBridge(
        [
            _protocol_start(),
            _protocol_request(4, 0),
            _protocol_ack(4, 0),
            _protocol_end(),
        ],
        events,
    )
    runtime = _DirectProtocolRuntime(events)

    def fail_builder(*args: Any) -> dict[str, Any]:
        del args
        raise RuntimeError("direct trace serialization failed")

    monkeypatch.setattr(remote_client, "_build_direct_steer_trace", fail_builder)

    _run_direct_protocol(bridge, runtime)

    actions = [message for message in bridge.sent if message[0] == "action"]
    assert len(actions) == 1
    np.testing.assert_array_equal(actions[0][4], runtime.results[0].selected_action)
    assert actions[0][5] is None
    assert "Omitting direct decoder steering trace after serialization failure: direct trace serialization failed" in caplog.text


def _direct_backend_config() -> dict[str, object]:
    config = yaml.safe_load(
        (ROOT / "deploy_smolvla/configs/deploy_smolvla_jax.yaml").read_text()
    )
    config["backend"] = "direct_tactile_decoder"
    config["direct_decoder"] = {"bundle": str(ABLATION), "device": "cpu"}
    config["observation"]["data_type"] = "vitac"
    return config


def test_direct_decoder_config_and_launcher() -> None:
    config_path = ROOT / "deploy_smolvla/configs/deploy_direct_decoder.yaml"
    launcher_path = ROOT / "deploy_smolvla/scripts/start_direct_decoder.sh"
    config = remote_client.load_config(config_path)
    assert config["backend"] == "direct_tactile_decoder"
    assert config["observation"]["data_type"] == "vitac"
    assert config["control"]["action_horizon"] == 20
    assert config["control"]["steps_per_inference"] == 20
    assert (
        config["control"]["steps_per_inference"]
        == config["control"]["action_horizon"]
    )
    launcher = launcher_path.read_text()
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in launcher
    assert "start_remote_client.sh" in launcher


def _direct_policy(*, use_tactile_encoder: bool, rtc_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            use_tactile_encoder=use_tactile_encoder,
            rtc_config=(
                SimpleNamespace(enabled=True, execution_horizon=20)
                if rtc_enabled
                else None
            ),
            chunk_size=20,
            action_dim=20,
            image_keys=("observation.images.camera0",),
            state_dim=14,
            empty_cameras=0,
        ),
        reset=lambda: None,
    )


def test_run_rejects_direct_backend_with_checkpoint_rtc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _direct_backend_config()
    monkeypatch.setattr(remote_client, "load_config", lambda path: config)
    monkeypatch.setattr(
        remote_client,
        "_load_validated_policy",
        lambda *args, **kwargs: _direct_policy(
            use_tactile_encoder=False, rtc_enabled=True
        ),
    )

    with pytest.raises(ValueError, match="does not support checkpoint RTC"):
        remote_client.run(tmp_path / "deploy.yaml")


def test_run_rejects_direct_backend_with_tactile_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _direct_backend_config()
    monkeypatch.setattr(remote_client, "load_config", lambda path: config)
    monkeypatch.setattr(
        remote_client,
        "_load_validated_policy",
        lambda *args, **kwargs: _direct_policy(
            use_tactile_encoder=True, rtc_enabled=False
        ),
    )

    with pytest.raises(ValueError, match="requires a visual-only JaxSmolVLAPolicy"):
        remote_client.run(tmp_path / "deploy.yaml")


def test_direct_backend_requires_vitac_horizon_and_bundle(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (ROOT / "deploy_smolvla/configs/deploy_smolvla_jax.yaml").read_text()
    )
    config["backend"] = "direct_tactile_decoder"
    config["direct_decoder"] = {"bundle": str(ABLATION), "device": "cpu"}
    config["observation"]["data_type"] = "vision"
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="requires observation.data_type='vitac'"):
        remote_client.load_config(path)


def test_direct_backend_config_rejects_partial_chunks(tmp_path: Path) -> None:
    config = _direct_backend_config()
    config["control"]["steps_per_inference"] = 19
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(
        ValueError,
        match="steps_per_inference to equal action_horizon",
    ):
        remote_client.load_config(path)


def test_direct_run_advertises_server_config_and_routes_per_action_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _direct_backend_config()
    config["control"]["steps_per_inference"] = 20
    config["runtime"].update(auto_start=True, warmup_runs=1, max_iterations=2)
    config["logging"]["save_observations"] = False
    config["connection"]["require_token"] = False
    events: list[tuple[Any, ...]] = []
    server_config: dict[str, Any] = {}
    routed: dict[str, Any] = {}
    frame = {
        "observation.state": np.zeros((14,), dtype=np.float32),
        "camera": np.zeros((2, 2, 3), dtype=np.uint8),
        **{
            key: np.zeros((2, 2, 3), dtype=np.uint8)
            for key in DIRECT_TACTILE_KEYS
        },
    }

    class Preprocessor:
        @staticmethod
        def unnormalize_actions(actions: np.ndarray) -> np.ndarray:
            return np.asarray(actions)

    class Policy:
        config = SimpleNamespace(
            state_dim=14,
            action_dim=20,
            chunk_size=20,
            image_keys=("camera",),
            tactile_keys=(),
            use_tactile_encoder=False,
            empty_cameras=0,
            rtc_config=None,
            adapt_to_pi_aloha=False,
        )
        preprocessor = Preprocessor()

        @staticmethod
        def reset() -> None:
            events.append(("policy_reset",))

        @staticmethod
        def predict_action_chunk(*args: Any, **kwargs: Any) -> jnp.ndarray:
            del args, kwargs
            events.append(("predict",))
            return jnp.zeros((1, 20, 20), dtype=jnp.float32)

    class Decoder:
        fixed_noise_jax = jnp.zeros((1, 20, 32), dtype=jnp.float32)
        tactile_keys = DIRECT_TACTILE_KEYS

        @staticmethod
        def reset() -> None:
            events.append(("decoder_reset",))

        @staticmethod
        def refine(coarse: np.ndarray, observation: object) -> np.ndarray:
            del observation
            events.append(("refine",))
            return np.asarray(coarse)

    decoder = Decoder()

    class Steering:
        def __init__(self, *, policy: Policy, decoder: Decoder) -> None:
            self.policy = policy
            self.decoder = decoder
            self.tactile_keys = tuple(decoder.tactile_keys)
            events.append(("steering_init",))

        @staticmethod
        def reset() -> None:
            events.append(("steering_reset",))

    class Bridge:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            events.append(("bridge",))

        @staticmethod
        def send_config(value: dict[str, Any]) -> None:
            server_config.update(value)
            events.append(("config",))

        @staticmethod
        def receive_observation(timeout: float) -> tuple[int, dict[str, Any]]:
            del timeout
            events.append(("receive_observation",))
            return 7, frame

        @staticmethod
        def send_state(state: str) -> None:
            events.append((state,))

        @staticmethod
        def send_action(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            pytest.fail("direct execution entered the legacy send_action loop")

        @staticmethod
        def close() -> None:
            events.append(("close",))

    def run_direct(bridge: Bridge, steering: Steering, **kwargs: Any) -> None:
        routed.update(bridge=bridge, steering=steering, **kwargs)
        events.append(("direct_protocol",))

    monkeypatch.setattr(remote_client, "load_config", lambda path: config)
    monkeypatch.setattr(
        remote_client, "_load_validated_policy", lambda *args, **kwargs: Policy()
    )
    monkeypatch.setattr(
        remote_client.DirectDecoderRuntime,
        "from_bundle",
        lambda *args, **kwargs: decoder,
    )
    monkeypatch.setattr(remote_client, "DirectDecoderSteeringRuntime", Steering, raising=False)
    monkeypatch.setattr(remote_client, "RobotBridgeClient", Bridge)
    monkeypatch.setattr(remote_client, "_run_direct_decoder_protocol", run_direct)

    remote_client.run(tmp_path / "deploy.yaml")

    assert server_config == {
        "data_type": "vitac",
        "language_prompt": config["observation"]["language_prompt"],
        "control_frequency": 20.0,
        "controller_frequency": 80.0,
        "single_arm_mode": False,
        "no_state_obs_mode": False,
        "steps_per_inference": 20,
        "action_horizon": 20,
        "execution_protocol": "frs_steering_v1",
        "steering_protection_interval_s": None,
        "frs_tactile_keys": list(DIRECT_TACTILE_KEYS),
    }
    assert routed["max_chunks"] == 2
    assert isinstance(routed["steering"], Steering)
    names = [event[0] for event in events]
    assert names.index("predict") < names.index("steering_reset")
    assert names.index("steering_reset") < names.index("start")
    assert names.index("start") < names.index("direct_protocol")


def test_predict_chunk_refines_normalized_actions_before_unnormalizing() -> None:
    coarse = jnp.ones((1, 2, 1), dtype=jnp.float32)
    refined = np.full((1, 2, 1), 4.0, dtype=np.float32)
    unnormalized: list[np.ndarray] = []
    refined_inputs: list[tuple[np.ndarray, object]] = []
    predict_kwargs: dict[str, object] = {}

    class Preprocessor:
        @staticmethod
        def unnormalize_actions(actions: np.ndarray) -> np.ndarray:
            unnormalized.append(np.asarray(actions))
            return np.asarray(actions) * 10.0

    class Policy:
        config = SimpleNamespace(chunk_size=2, action_dim=1)
        preprocessor = Preprocessor()

        @staticmethod
        def predict_action_chunk(observation, task, **kwargs):
            del observation, task
            predict_kwargs.update(kwargs)
            return coarse

    observation = {"tactile": np.zeros((1,), dtype=np.float32)}
    runtime = SimpleNamespace(
        fixed_noise_jax=jnp.zeros((1, 2, 1), dtype=jnp.float32),
        refine=lambda normalized, received_observation: (
            refined_inputs.append((np.asarray(normalized), received_observation)) or refined
        ),
    )

    action, normalized = remote_client._predict_chunk(
        Policy(),
        observation,
        "task",
        seed=7,
        jit=False,
        num_steps=3,
        previous_chunk=np.zeros((1, 1), dtype=np.float32),
        inference_delay=1,
        execution_horizon=2,
        direct_decoder=runtime,
    )

    assert predict_kwargs == {
        "seed": 7,
        "noise": runtime.fixed_noise_jax,
        "jit": False,
        "normalized": True,
        "num_steps": 3,
        "previous_chunk": None,
        "inference_delay": None,
        "execution_horizon": None,
    }
    assert len(refined_inputs) == 1
    np.testing.assert_array_equal(refined_inputs[0][0], np.asarray(coarse))
    assert refined_inputs[0][1] is observation
    assert len(unnormalized) == 1
    np.testing.assert_array_equal(unnormalized[0], refined)
    np.testing.assert_array_equal(action, np.full((2, 1), 40.0, dtype=np.float32))
    np.testing.assert_array_equal(normalized, np.full((2, 1), 4.0, dtype=np.float32))


def test_released_decoder_state_loads_strictly() -> None:
    checkpoint = torch.load(
        ABLATION / "decoder" / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = DirectTactileActionDecoder.from_config(checkpoint["decoder_config"])
    model.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    assert sum(parameter.numel() for parameter in model.parameters()) == 471_828


def test_fixed_noise_matches_training_contract() -> None:
    noise = np.load(ABLATION / "fixed_noise.npy", allow_pickle=False)
    assert noise.dtype == np.float32
    assert noise.shape == (1, 20, 32)
    assert np.isfinite(noise).all()
    np.testing.assert_array_equal(noise[:, :, 20:], 0.0)


def test_runtime_refine_records_copied_snapshots_and_reset_clears_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(DirectDecoderRuntime)
    runtime.device = torch.device("cpu")
    runtime.tactile_keys = DIRECT_TACTILE_KEYS
    runtime.encoder = lambda images: torch.ones((4, 512), dtype=torch.float32)
    runtime.decoder = lambda coarse, tactile: coarse + 2.0
    runtime.last_vla_normalized = None
    runtime.last_direct_normalized = None
    monkeypatch.setattr(
        direct_decoder_module,
        "_preprocess_image",
        lambda image: np.zeros((3, 224, 224), dtype=np.float32),
    )

    coarse = np.ones((1, 20, 20), dtype=np.float32)
    observation = {
        key: np.zeros((1, 1, 3), dtype=np.uint8) for key in DIRECT_TACTILE_KEYS
    }

    direct = runtime.refine(coarse, observation)

    np.testing.assert_array_equal(runtime.last_vla_normalized, coarse)
    np.testing.assert_array_equal(runtime.last_direct_normalized, direct)
    coarse[0, 0, 0] = -1.0
    direct[0, 0, 0] = -2.0
    assert runtime.last_vla_normalized[0, 0, 0] == 1.0
    assert runtime.last_direct_normalized[0, 0, 0] == 3.0

    runtime.reset()

    assert runtime.last_vla_normalized is None
    assert runtime.last_direct_normalized is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="deployment uses cuda:0")
def test_runtime_refine_returns_finite_normalized_chunk() -> None:
    runtime = DirectDecoderRuntime.from_bundle(ABLATION, device="cuda:0")
    observation = {
        key: np.zeros((240, 320, 3), dtype=np.uint8)
        for key in DIRECT_TACTILE_KEYS
    }
    result = runtime.refine(np.zeros((1, 20, 20), dtype=np.float32), observation)
    assert result.shape == (1, 20, 20)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
