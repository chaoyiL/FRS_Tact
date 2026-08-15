from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from deploy_smolvla import remote_client
from deploy_smolvla.frs_protocol import FRSChunkEnd, FRSChunkStart
from deploy_smolvla.frs_runtime import FRSChunkReady
from deploy_smolvla.remote_client import (
    _prepare_observation,
    _remaining_action_chunk,
    _robot_image_keys,
    _robot_tactile_keys,
    _validate_observation_mode,
)
from train_vtsmolvla.configuration import VTSmolVLAConfig
from train_vtsmolvla.modeling import VTJaxSmolVLA, normalize_tactile_embeddings
from train_vtsmolvla.policy import VTJaxSmolVLAPolicy
from train_vtsmolvla.validation import CheckpointValidationReport


def _write_remote_config(
    path: Path,
    checkpoint: str,
    *,
    revision: str | None = None,
    allow_download: bool = False,
) -> Path:
    config = {
        "checkpoint": checkpoint,
        "revision": revision,
        "allow_download": allow_download,
        "checkpoint_contract": {
            "state_dim": 20,
            "action_dim": 20,
            "chunk_size": 20,
            "image_keys": [
                "observation.images.camera1",
                "observation.images.camera2",
            ],
            "tactile_keys": [
                "observation.images.tactile_left_0",
                "observation.images.tactile_right_0",
                "observation.images.tactile_left_1",
                "observation.images.tactile_right_1",
            ],
            "tactile_embedding_dim": 512,
            "tactile_num_tokens": 4,
            "lora_rank": 16,
            "vlm_lora_target_modules": ["q_proj", "v_proj"],
        },
        "connection": {
            "address": "127.0.0.1",
            "port": 26421,
            "action_ack_timeout_s": 1.0,
            "require_token": False,
        },
        "observation": {
            "data_type": "vitac",
            "language_prompt": "test",
            "single_arm_mode": False,
            "no_state_obs_mode": False,
        },
        "control": {
            "control_frequency": 30.0,
            "controller_frequency": 80.0,
            "steps_per_inference": 10,
            "action_horizon": 20,
        },
        "runtime": {"warmup_runs": 1, "max_iterations": 0},
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _tactile_policy():
    config = SimpleNamespace(
        image_keys=("observation.images.camera1", "observation.images.camera2"),
        use_tactile_encoder=True,
        tactile_keys=(
            "observation.images.tactile_left_0",
            "observation.images.tactile_right_0",
        ),
    )
    return SimpleNamespace(config=config)


def test_invalid_checkpoint_fails_before_policy_or_robot_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "invalid-checkpoint"
    checkpoint.mkdir()
    config_path = _write_remote_config(tmp_path / "deploy.yaml", str(checkpoint))
    calls: list[str] = []

    def policy_loader(*args, **kwargs):
        del args, kwargs
        calls.append("policy")
        pytest.fail("policy must not load before checkpoint validation")

    def bridge_loader(*args, **kwargs):
        del args, kwargs
        calls.append("bridge")
        pytest.fail("robot bridge must not be constructed before checkpoint validation")

    monkeypatch.setattr(remote_client.VTJaxSmolVLAPolicy, "from_pretrained", policy_loader)
    monkeypatch.setattr(remote_client, "RobotBridgeClient", bridge_loader)

    with pytest.raises(ValueError, match="checkpoint validation failed"):
        remote_client.run(config_path)

    assert calls == []


def test_checkpoint_steps_override_is_sent_to_robot_server_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    revision = "a" * 40
    config_path = _write_remote_config(
        tmp_path / "deploy.yaml",
        "owner/vt-model",
        revision=revision,
        allow_download=True,
    )
    events: list[str] = []
    sent_server_config: dict[str, object] = {}

    def resolve_checkpoint(checkpoint, *, revision, local_files_only):
        events.append("resolve")
        assert checkpoint == "owner/vt-model"
        assert revision == "a" * 40
        assert local_files_only is False
        return snapshot

    def validate_checkpoint(path, *, expected, base_sidecars=None, require_weight=True):
        events.append("validate")
        assert path == snapshot
        assert expected.state_dim == 20
        assert expected.action_dim == 20
        assert expected.chunk_size == 20
        assert expected.tactile_num_tokens == 4
        assert expected.vlm_lora_target_modules == ("q_proj", "v_proj")
        assert base_sidecars is None
        assert require_weight is True
        return CheckpointValidationReport(snapshot, ())

    policy = SimpleNamespace(
        config=SimpleNamespace(
            state_dim=20,
            action_dim=20,
            chunk_size=20,
            n_action_steps=5,
            image_keys=("observation.images.camera1", "observation.images.camera2"),
            tactile_keys=(
                "observation.images.tactile_left_0",
                "observation.images.tactile_right_0",
                "observation.images.tactile_left_1",
                "observation.images.tactile_right_1",
            ),
            tactile_num_tokens=4,
            use_tactile_encoder=True,
            empty_cameras=0,
            rtc_config=None,
        ),
        reset=lambda: None,
    )

    def policy_loader(checkpoint, **kwargs):
        events.append("policy")
        assert checkpoint == snapshot
        assert kwargs["local_files_only"] is True
        assert kwargs["revision"] is None
        return policy

    class RecordingBridge:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            events.append("bridge")

        def send_config(self, server_config):
            events.append("send_config")
            sent_server_config.update(server_config)
            raise RuntimeError("stop after server config")

    monkeypatch.setattr(remote_client, "resolve_checkpoint", resolve_checkpoint, raising=False)
    monkeypatch.setattr(remote_client, "validate_checkpoint", validate_checkpoint, raising=False)
    monkeypatch.setattr(remote_client.VTJaxSmolVLAPolicy, "from_pretrained", policy_loader)
    monkeypatch.setattr(remote_client, "RobotBridgeClient", RecordingBridge)

    with pytest.raises(RuntimeError, match="stop after server config"):
        remote_client.run(config_path)

    assert events == ["resolve", "validate", "policy", "bridge", "send_config"]
    assert policy.config.n_action_steps == 5
    assert sent_server_config["steps_per_inference"] == 10
    assert sent_server_config["action_horizon"] == 20
    assert "execution_protocol" not in sent_server_config
    assert "steering_protection_interval_s" not in sent_server_config
    assert "frs_tactile_keys" not in sent_server_config


def test_offline_checkpoint_resolution_fails_before_policy_or_robot_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_remote_config(
        tmp_path / "deploy.yaml",
        "owner/not-cached",
        revision="b" * 40,
        allow_download=False,
    )
    events: list[str] = []

    def resolve_checkpoint(checkpoint, *, revision, local_files_only):
        events.append("resolve")
        assert checkpoint == "owner/not-cached"
        assert revision == "b" * 40
        assert local_files_only is True
        raise FileNotFoundError("offline snapshot is unavailable")

    monkeypatch.setattr(remote_client, "resolve_checkpoint", resolve_checkpoint, raising=False)
    monkeypatch.setattr(
        remote_client.VTJaxSmolVLAPolicy,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("policy must not load"),
    )
    monkeypatch.setattr(
        remote_client,
        "RobotBridgeClient",
        lambda *args, **kwargs: pytest.fail("robot bridge must not be constructed"),
    )

    with pytest.raises(FileNotFoundError, match="offline snapshot is unavailable"):
        remote_client.run(config_path)

    assert events == ["resolve"]


def test_default_deployment_config_infers_contract_from_checkpoint() -> None:
    from train_vtsmolvla.validation import contract_from_checkpoint

    config = remote_client.load_config(remote_client.DEFAULT_CONFIG)
    checkpoint = Path(str(config["checkpoint"])).expanduser()

    assert isinstance(config["checkpoint"], str) and config["checkpoint"]
    assert config["revision"] is None
    assert config["allow_download"] is False
    assert config["connection"]["token"] is None
    assert config["connection"]["token_env"] == "VB_ROBOT_TOKEN"
    assert config["connection"]["require_token"] is True
    assert config.get("checkpoint_contract") in (None, {})
    assert config["rename_map"] == {
        "observation.images.camera0": "observation.images.camera1",
        "observation.images.camera1": "observation.images.camera2",
    }
    assert config["observation"]["data_type"] == "vision"
    assert config["observation"]["single_arm_mode"] is False
    assert config["control"]["action_horizon"] == 10

    if not checkpoint.is_dir():
        pytest.skip(f"default deployment checkpoint unavailable: {checkpoint}")

    inferred = contract_from_checkpoint(checkpoint)
    contract = remote_client._checkpoint_contract(
        config,
        config["control"],
        inferred=inferred,
    )
    assert contract == inferred
    assert contract.state_dim == 20
    assert contract.action_dim == 20
    assert contract.chunk_size == 10
    assert contract.image_keys == (
        "observation.images.camera1",
        "observation.images.camera2",
    )
    assert contract.tactile_keys == ()
    assert contract.tactile_num_tokens == 0
    assert contract.tactile_proj_mode == "frozen"
    assert contract.lora_rank == 0
    assert contract.vlm_lora_target_modules == ()


def test_checkpoint_contract_overrides_only_provided_fields() -> None:
    from train_vtsmolvla.validation import CheckpointContract

    inferred = CheckpointContract(
        state_dim=20,
        action_dim=20,
        chunk_size=10,
        image_keys=("observation.images.camera1", "observation.images.camera2"),
        tactile_keys=(),
        tactile_embedding_dim=512,
        tactile_num_tokens=0,
        lora_rank=0,
        vlm_lora_target_modules=(),
    )
    config = {
        "checkpoint_contract": {
            "lora_rank": 8,
            "vlm_lora_target_modules": ["q_proj"],
        }
    }
    contract = remote_client._checkpoint_contract(
        config,
        {"action_horizon": 10},
        inferred=inferred,
    )
    assert contract.state_dim == 20
    assert contract.chunk_size == 10
    assert contract.lora_rank == 8
    assert contract.vlm_lora_target_modules == ("q_proj",)
    assert contract.image_keys == inferred.image_keys


def test_checkpoint_contract_can_be_omitted_when_inferred() -> None:
    from train_vtsmolvla.validation import CheckpointContract

    inferred = CheckpointContract(
        state_dim=4,
        action_dim=2,
        chunk_size=5,
        image_keys=("rgb",),
        tactile_keys=(),
        tactile_embedding_dim=512,
        tactile_num_tokens=0,
        lora_rank=0,
        vlm_lora_target_modules=(),
    )
    contract = remote_client._checkpoint_contract(
        {},
        {"action_horizon": 5},
        inferred=inferred,
    )
    assert contract == inferred

@pytest.mark.parametrize("steps_per_inference", [True, 10.5, "10"])
def test_load_config_rejects_non_integer_steps_per_inference(
    tmp_path: Path,
    steps_per_inference: object,
) -> None:
    config_path = _write_remote_config(tmp_path / "deploy.yaml", "owner/vt-model")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["control"]["steps_per_inference"] = steps_per_inference
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="steps_per_inference must be an integer"):
        remote_client.load_config(config_path)


@pytest.mark.parametrize("steps_per_inference", [0, 21])
def test_load_config_rejects_out_of_range_steps_per_inference(
    tmp_path: Path,
    steps_per_inference: int,
) -> None:
    config_path = _write_remote_config(tmp_path / "deploy.yaml", "owner/vt-model")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["control"]["steps_per_inference"] = steps_per_inference
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="steps_per_inference must be between 1 and action_horizon",
    ):
        remote_client.load_config(config_path)


def test_tactile_embedding_normalization_has_unit_rms() -> None:
    embeddings = jnp.asarray([[[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]]])
    normalized = normalize_tactile_embeddings(embeddings)
    rms = jnp.sqrt(jnp.mean(jnp.square(normalized), axis=-1))
    np.testing.assert_allclose(rms, np.ones((1, 2)), rtol=1e-5, atol=1e-5)


def test_cached_tactile_embeddings_keep_trainable_projection() -> None:
    config = VTSmolVLAConfig(
        use_tactile_encoder=True,
        tactile_encoder_path="unused-for-cached-input",
        tactile_keys=("left", "right"),
        tactile_num_tokens=2,
        tactile_embedding_dim=3,
        text_hidden_size=4,
    )
    model = VTJaxSmolVLA(config)
    params = {
        "model.tactile_proj.weight": jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10,
        "model.tactile_proj.bias": jnp.arange(4, dtype=jnp.float32),
    }
    cached = jnp.asarray([[[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]]], dtype=jnp.float16)
    actual = model.embed_tactile(params, tactile_embeddings=cached)
    expected = (
        normalize_tactile_embeddings(cached) @ params["model.tactile_proj.weight"].T
        + params["model.tactile_proj.bias"]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_remote_observation_keeps_and_requires_tactile_images() -> None:
    policy = _tactile_policy()
    rename_map = {
        "observation.images.camera0": "observation.images.camera1",
        "observation.images.camera1": "observation.images.camera2",
    }
    image_keys = _robot_image_keys(policy, rename_map)
    tactile_keys = _robot_tactile_keys(policy, rename_map)
    observation = {
        "observation.state": np.zeros((20,), dtype=np.float32),
        **{key: np.zeros((8, 8, 3), dtype=np.uint8) for key in image_keys},
    }

    prepared = _prepare_observation(
        observation,
        state_dim=20,
        image_keys=image_keys,
        empty_cameras=0,
        required_image_keys=tactile_keys,
    )

    assert image_keys == (
        "observation.images.camera0",
        "observation.images.camera1",
        "observation.images.tactile_left_0",
        "observation.images.tactile_right_0",
    )
    assert set(tactile_keys).issubset(prepared)

    observation.pop(tactile_keys[0])
    with pytest.raises(ValueError, match="required tactile"):
        _prepare_observation(
            observation,
            state_dim=20,
            image_keys=image_keys,
            empty_cameras=0,
            required_image_keys=tactile_keys,
        )


def test_remote_observation_mode_must_match_checkpoint() -> None:
    _validate_observation_mode("vitac", use_tactile_encoder=True)
    _validate_observation_mode("vision", use_tactile_encoder=False)

    with pytest.raises(ValueError, match="requires observation.data_type='vitac'"):
        _validate_observation_mode("vision", use_tactile_encoder=True)
    with pytest.raises(ValueError, match="requires observation.data_type='vision'"):
        _validate_observation_mode("vitac", use_tactile_encoder=False)


def test_rtc_previous_chunk_is_shifted_by_executed_steps() -> None:
    chunk = np.arange(20, dtype=np.float32).reshape(10, 2)
    remaining = _remaining_action_chunk(chunk, 3)

    np.testing.assert_array_equal(remaining, chunk[3:])
    assert remaining.shape == (7, 2)
    assert not np.shares_memory(remaining, chunk)


def test_policy_inference_accepts_cached_tactile_embeddings() -> None:
    captured = {}

    class FakeModel:
        def sample_actions(self, params, *args, **kwargs):
            del params, args
            captured.update(kwargs)
            return jnp.zeros((1, 2, 1), dtype=jnp.float32)

    class FakePreprocessor:
        def prepare(self, observation, task):
            del observation, task
            return {
                "images": jnp.zeros((1, 1, 2, 2, 3)),
                "image_masks": jnp.ones((1, 1), dtype=bool),
                "language_tokens": jnp.ones((1, 2), dtype=jnp.int32),
                "language_masks": jnp.ones((1, 2), dtype=bool),
                "state": jnp.zeros((1, 1)),
                "tactile_embeddings": jnp.ones((1, 2, 3)),
                "tactile_masks": jnp.ones((1, 2), dtype=bool),
            }

    policy = object.__new__(VTJaxSmolVLAPolicy)
    policy.config = SimpleNamespace(
        chunk_size=2,
        max_action_dim=1,
        num_steps=2,
        rtc_config=None,
        adapt_to_pi_aloha=False,
    )
    policy.params = {}
    policy.model = FakeModel()
    policy.preprocessor = FakePreprocessor()
    policy._compiled_samples = {}

    policy.predict_action_chunk({}, "task", jit=False, normalized=True)

    assert captured["tactile_images"] is None
    assert captured["tactile_embeddings"].shape == (1, 2, 3)


def test_select_action_advances_chunk_seed() -> None:
    policy = object.__new__(VTJaxSmolVLAPolicy)
    policy.config = SimpleNamespace(n_action_steps=1)
    seeds = []

    def predict(observation, task, *, seed, jit, **kwargs):
        del observation, task, jit, kwargs
        seeds.append(seed)
        return jnp.asarray([[seed]], dtype=jnp.float32)

    policy.predict_action_chunk = predict
    policy.reset()
    policy.select_action({}, "task", seed=10, jit=False)
    policy.select_action({}, "task", seed=10, jit=False)

    assert seeds == [10, 11]


def _write_protocol_run_config(path: Path, *, frs_enabled: bool) -> Path:
    config = {
        "checkpoint": "unused",
        "allow_download": False,
        "checkpoint_contract": {
            "state_dim": 3,
            "action_dim": 2,
            "chunk_size": 3,
            "image_keys": ["camera"],
            "tactile_keys": [],
            "tactile_embedding_dim": 4,
            "tactile_num_tokens": 0,
            "lora_rank": 0,
            "vlm_lora_target_modules": [],
        },
        "connection": {
            "address": "127.0.0.1",
            "port": 26421,
            "action_ack_timeout_s": 2.0,
            "observation_timeout_s": 10.0,
            "require_token": False,
        },
        "observation": {
            "data_type": "vitac" if frs_enabled else "vision",
            "language_prompt": "pick",
            "single_arm_mode": False,
            "no_state_obs_mode": False,
        },
        "control": {
            "control_frequency": 20.0,
            "controller_frequency": 80.0,
            "steps_per_inference": 3,
            "action_horizon": 3,
        },
        "runtime": {
            "auto_start": True,
            "warmup_runs": 1,
            "max_iterations": 1,
        },
        "logging": {"save_observations": False},
    }
    if frs_enabled:
        config["frs"] = {
            "enabled": True,
            "checkpoint": "unused-frs",
            "tactile_encoder_checkpoint": "unused-tactile",
            "tactile_keys": ["tactile"],
            "tactile_window_divisor": 1,
            "reverse_steps": 2,
            "reverse_solver": "euler",
            "decode_steps": 2,
            "decode_solver": "euler",
        }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _run_protocol_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    frs_enabled: bool,
    fail_frs_receive: bool = False,
    fail_begin_chunk: bool = False,
    fail_stop_send: bool = False,
) -> tuple[Path, list[tuple[object, ...]], list[np.ndarray]]:
    events: list[tuple[object, ...]] = []
    sent_actions: list[np.ndarray] = []
    normalized = jnp.asarray(
        [[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]], dtype=jnp.float32
    )

    class Preprocessor:
        @staticmethod
        def unnormalize_actions(action: object) -> jnp.ndarray:
            return jnp.asarray(action) + 10.0

    class Policy:
        config = SimpleNamespace(
            state_dim=3,
            action_dim=2,
            chunk_size=3,
            n_action_steps=3,
            image_keys=("camera",),
            tactile_keys=(),
            tactile_num_tokens=0,
            use_tactile_encoder=False,
            empty_cameras=0,
            rtc_config=None,
            num_steps=4,
            adapt_to_pi_aloha=False,
        )
        preprocessor = Preprocessor()

        @staticmethod
        def reset() -> None:
            events.append(("policy_reset",))

        @staticmethod
        def predict_action_chunk(*args: object, **kwargs: object) -> jnp.ndarray:
            del args, kwargs
            events.append(("predict",))
            return normalized

    class SteeringPolicy:
        tactile_keys = ("tactile",)

        def __init__(self, *args: object, policy: Policy, **kwargs: object) -> None:
            del args, kwargs
            self.policy = policy
            self.config = SimpleNamespace(
                checkpoint="unused-frs",
                tactile_window_divisor=1,
                steering_protection_interval_s=None,
            )
            self.model = SimpleNamespace(config=SimpleNamespace(tactile_window=3))

        def resolved_tactile_window(self) -> int:
            return int(self.policy.config.chunk_size) // int(self.config.tactile_window_divisor)
            self.last_diagnostics = None
            events.append(("frs_init",))

        @staticmethod
        def reset_episode(initial_observation: object) -> None:
            del initial_observation
            events.append(("reset_episode",))

        @staticmethod
        def warmup_all_tactile_lengths() -> None:
            events.append(("warmup_all",))

        @staticmethod
        def begin_chunk(
            chunk_id: int,
            initial_observation: object,
            task: str,
            **kwargs: object,
        ) -> FRSChunkReady:
            del initial_observation, task, kwargs
            events.append(("begin_chunk", chunk_id))
            if fail_begin_chunk:
                raise RuntimeError("source model failed")
            chunk = np.zeros((1, 3, 2), dtype=np.float32)
            return FRSChunkReady(chunk_id, chunk, chunk, chunk, 1.0, 2.0)

        @staticmethod
        def end_chunk(chunk_id: int) -> None:
            events.append(("end_chunk", chunk_id))

    warmup_observation = {
        "observation.state": np.zeros((3,), dtype=np.float32),
        "camera": np.zeros((4, 4, 3), dtype=np.uint8),
        "tactile": np.zeros((4, 4, 3), dtype=np.uint8),
    }

    class Bridge:
        observation_calls = 0
        frs_calls = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            events.append(("bridge",))

        @staticmethod
        def send_config(config: object) -> None:
            del config
            events.append(("config",))

        @classmethod
        def receive_observation(cls, timeout: float) -> tuple[int, dict[str, np.ndarray]]:
            del timeout
            cls.observation_calls += 1
            events.append(("receive_observation", cls.observation_calls))
            return cls.observation_calls, warmup_observation

        @staticmethod
        def send_state(state: str) -> None:
            events.append((state,))
            if state == "stop" and fail_stop_send:
                raise ConnectionError("bridge already closed")

        @staticmethod
        def send_action(action: np.ndarray, obs_seq: int, *, trace: object) -> None:
            del trace
            events.append(("legacy_action", obs_seq))
            sent_actions.append(np.array(action, copy=True))

        @staticmethod
        def receive_action_ack(obs_seq: int, timeout: float) -> None:
            del timeout
            events.append(("legacy_ack", obs_seq))

        @classmethod
        def receive_frs_message(cls, timeout: float) -> FRSChunkStart | FRSChunkEnd:
            del timeout
            cls.frs_calls += 1
            events.append(("receive_frs", cls.frs_calls))
            if fail_frs_receive:
                raise TimeoutError("FRS receive timed out")
            if cls.frs_calls == 1:
                return FRSChunkStart(
                    obs_seq=11,
                    chunk_id=5,
                    observation=warmup_observation,
                    observation_timestamp=100.0,
                    control_dt=0.05,
                    action_horizon=3,
                    execution_mode="block",
                    action_timestamps=None,
                    nominal_chunk_end=None,
                )
            return FRSChunkEnd(5, "exhausted", 0, 0)

        @staticmethod
        def send_frs_chunk_ready(
            obs_seq: int,
            chunk_id: int,
            prediction_trace: object,
        ) -> None:
            del prediction_trace
            events.append(("ready", obs_seq, chunk_id))

        @staticmethod
        def close() -> None:
            events.append(("close",))

    monkeypatch.setattr(remote_client, "_load_validated_policy", lambda *a, **k: Policy())
    monkeypatch.setattr(remote_client, "FRSRuntime", SteeringPolicy)
    monkeypatch.setattr(remote_client, "RobotBridgeClient", Bridge)
    config_path = _write_protocol_run_config(
        tmp_path / ("frs.yaml" if frs_enabled else "legacy.yaml"),
        frs_enabled=frs_enabled,
    )
    return config_path, events, sent_actions


def test_frs_run_warms_all_tactile_lengths_before_start_and_uses_typed_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, events, sent_actions = _run_protocol_fixture(
        tmp_path, monkeypatch, frs_enabled=True
    )

    remote_client.run(config_path)

    names = [event[0] for event in events]
    assert names.index("receive_observation") < names.index("reset_episode")
    assert names.index("reset_episode") < names.index("warmup_all") < names.index("start")
    assert names.index("start") < names.index("receive_frs") < names.index("begin_chunk")
    assert names[-2:] == ["stop", "close"]
    assert sent_actions == []


def test_frs_receive_failure_sends_stop_and_closes_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, events, _ = _run_protocol_fixture(
        tmp_path,
        monkeypatch,
        frs_enabled=True,
        fail_frs_receive=True,
    )

    with pytest.raises(TimeoutError, match="FRS receive timed out"):
        remote_client.run(config_path)

    assert ("receive_frs", 1) in events
    assert events[-2:] == [("stop",), ("close",)]


def test_frs_disconnect_during_stop_does_not_mask_receive_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, events, _ = _run_protocol_fixture(
        tmp_path,
        monkeypatch,
        frs_enabled=True,
        fail_frs_receive=True,
        fail_stop_send=True,
    )

    with pytest.raises(TimeoutError, match="FRS receive timed out"):
        remote_client.run(config_path)

    assert events[-2:] == [("stop",), ("close",)]


def test_frs_model_failure_sends_stop_and_closes_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, events, _ = _run_protocol_fixture(
        tmp_path,
        monkeypatch,
        frs_enabled=True,
        fail_begin_chunk=True,
    )

    with pytest.raises(RuntimeError, match="source model failed"):
        remote_client.run(config_path)

    assert ("begin_chunk", 5) in events
    assert events[-2:] == [("stop",), ("close",)]


def test_legacy_run_sends_the_exact_full_unnormalized_chunk_and_acknowledges_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, events, sent_actions = _run_protocol_fixture(
        tmp_path, monkeypatch, frs_enabled=False
    )

    remote_client.run(config_path)

    assert [event for event in events if event[0] == "legacy_action"] == [
        ("legacy_action", 2)
    ]
    assert ("legacy_ack", 2) in events
    assert len(sent_actions) == 1
    np.testing.assert_array_equal(
        sent_actions[0],
        np.asarray([[[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]], dtype=np.float32)[0],
    )
