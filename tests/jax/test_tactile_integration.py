from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from deploy_smolvla import remote_client
from deploy_smolvla.remote_client import (
    _prepare_observation,
    _remaining_action_chunk,
    _robot_image_keys,
    _robot_tactile_keys,
    _validate_observation_mode,
)
from lerobot.policies.smolvla_jax.configuration import JaxSmolVLAConfig
from lerobot.policies.smolvla_jax.modeling import (
    JaxSmolVLA,
    _repeat_tactile_tokens_and_masks,
    normalize_tactile_embeddings,
)
from lerobot.policies.smolvla_jax.policy import JaxSmolVLAPolicy
from lerobot.policies.smolvla_jax.validation import CheckpointValidationReport


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

    monkeypatch.setattr(remote_client.JaxSmolVLAPolicy, "from_pretrained", policy_loader)
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
    monkeypatch.setattr(remote_client.JaxSmolVLAPolicy, "from_pretrained", policy_loader)
    monkeypatch.setattr(remote_client, "RobotBridgeClient", RecordingBridge)

    with pytest.raises(RuntimeError, match="stop after server config"):
        remote_client.run(config_path)

    assert events == ["resolve", "validate", "policy", "bridge", "send_config"]
    assert policy.config.n_action_steps == 5
    assert sent_server_config["steps_per_inference"] == 10
    assert sent_server_config["action_horizon"] == 20


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
        remote_client.JaxSmolVLAPolicy,
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


def test_default_deployment_config_pins_the_bimanual_vt_contract() -> None:
    config = remote_client.load_config(remote_client.DEFAULT_CONFIG)
    contract = remote_client._checkpoint_contract(config, config["control"])

    assert config["checkpoint"] == "KaiyueChen/vtsmolvla_01_4w"
    assert config["revision"] == "0b5cc8208ef118f505b1f736b0ec604b598f9424"
    assert config["allow_download"] is True
    assert config["connection"]["token"] is None
    assert config["connection"]["token_env"] == "VB_ROBOT_TOKEN"
    assert config["connection"]["require_token"] is True
    assert contract.state_dim == 20
    assert contract.action_dim == 20
    assert contract.chunk_size == 20
    assert contract.image_keys == (
        "observation.images.camera1",
        "observation.images.camera2",
    )
    assert contract.tactile_keys == (
        "observation.images.tactile_left_0",
        "observation.images.tactile_right_0",
        "observation.images.tactile_left_1",
        "observation.images.tactile_right_1",
    )
    assert contract.tactile_embedding_dim == 512
    assert contract.tactile_num_tokens == 4
    assert contract.lora_rank == 16
    assert contract.vlm_lora_target_modules == ("q_proj", "v_proj")
    assert config["rename_map"] == {
        "observation.images.camera0": "observation.images.camera1",
        "observation.images.camera1": "observation.images.camera2",
    }
    assert config["observation"]["data_type"] == "vitac"
    assert config["observation"]["single_arm_mode"] is False
    assert config["control"]["action_horizon"] == 20
    assert config["control"]["steps_per_inference"] == 10


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


@pytest.mark.parametrize("factor,expected_tokens", [(1, 4), (8, 32), (21, 84)])
def test_repeat_tactile_tokens_and_masks_is_key_major(
    factor: int, expected_tokens: int
) -> None:
    tokens = jnp.arange(4, dtype=jnp.float32).reshape(1, 4, 1)
    masks = jnp.asarray([[True, False, True, False]])

    expanded, expanded_masks = _repeat_tactile_tokens_and_masks(tokens, masks, factor)

    assert expanded.shape == (1, expected_tokens, 1)
    assert expanded_masks.shape == (1, expected_tokens)
    np.testing.assert_array_equal(
        expanded[0, :, 0], np.repeat(np.arange(4), factor)
    )
    np.testing.assert_array_equal(
        expanded_masks[0], np.repeat([True, False, True, False], factor)
    )


def test_repeat_factor_one_is_value_preserving() -> None:
    tokens = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3)
    masks = jnp.asarray([[True] * 4, [True, False, True, False]])
    expanded, expanded_masks = _repeat_tactile_tokens_and_masks(tokens, masks, 1)
    np.testing.assert_array_equal(expanded, tokens)
    np.testing.assert_array_equal(expanded_masks, masks)


def test_embed_prefix_repeats_tactile_tokens_and_ablation_mask() -> None:
    hidden_size = 3

    class StubPrefixModel(JaxSmolVLA):
        def embed_image(self, params, image):
            del params
            return jnp.ones((image.shape[0], 2, hidden_size), dtype=jnp.float32)

        def embed_language(self, params, tokens):
            del params
            return jnp.full(
                (tokens.shape[0], tokens.shape[1], hidden_size),
                7.0,
                dtype=jnp.float32,
            )

        def embed_tactile(self, params, tactile_images=None, *, tactile_embeddings=None):
            del params, tactile_images
            return jnp.asarray(tactile_embeddings, dtype=jnp.float32)

        def _linear(self, params, name, value, *, bias=False, **kwargs):
            del params, bias, kwargs
            assert name == "model.state_proj"
            return jnp.zeros((value.shape[0], hidden_size), dtype=jnp.float32)

    config = JaxSmolVLAConfig(
        use_tactile_encoder=True,
        tactile_encoder_path="unused",
        tactile_keys=("t0", "t1", "t2", "t3"),
        tactile_num_tokens=4,
        tactile_token_repeat_factor=8,
        text_hidden_size=hidden_size,
        max_state_dim=2,
    )
    model = StubPrefixModel(config)
    tactile = jnp.arange(12, dtype=jnp.float32).reshape(1, 4, hidden_size)
    common = dict(
        params={},
        images=jnp.zeros((1, 2, 3, 2, 2), dtype=jnp.float32),
        image_masks=jnp.ones((1, 2), dtype=jnp.bool_),
        language_tokens=jnp.ones((1, 3), dtype=jnp.int32),
        language_masks=jnp.ones((1, 3), dtype=jnp.bool_),
        state=jnp.zeros((1, 2), dtype=jnp.float32),
        tactile_embeddings=tactile,
    )

    prefix, pad_mask, _ = model.embed_prefix(
        **common,
        tactile_masks=jnp.ones((1, 4), dtype=jnp.bool_),
    )
    _, ablated_pad_mask, _ = model.embed_prefix(
        **common,
        tactile_masks=jnp.zeros((1, 4), dtype=jnp.bool_),
    )

    tactile_start = 4  # two image slots x two stub tokens
    tactile_stop = tactile_start + 32
    assert prefix.shape == (1, 4 + 32 + 3 + 1, hidden_size)
    np.testing.assert_array_equal(
        prefix[:, tactile_start:tactile_stop],
        jnp.repeat(tactile, 8, axis=1),
    )
    assert bool(jnp.all(pad_mask[:, tactile_start:tactile_stop]))
    assert bool(jnp.all(~ablated_pad_mask[:, tactile_start:tactile_stop]))


def test_cached_tactile_embeddings_keep_trainable_projection() -> None:
    config = JaxSmolVLAConfig(
        use_tactile_encoder=True,
        tactile_encoder_path="unused-for-cached-input",
        tactile_keys=("left", "right"),
        tactile_num_tokens=2,
        tactile_embedding_dim=3,
        text_hidden_size=4,
    )
    model = JaxSmolVLA(config)
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

    policy = object.__new__(JaxSmolVLAPolicy)
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
    policy = object.__new__(JaxSmolVLAPolicy)
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
