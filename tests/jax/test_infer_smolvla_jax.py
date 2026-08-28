import json
from types import SimpleNamespace

import pytest

from deploy_smolvla import remote_client
from tools import infer_smolvla_jax as infer


def test_policy_type_is_selected_from_checkpoint_config(tmp_path) -> None:
    from train_smolvla import JaxSmolVLAPolicy

    visual = tmp_path / "visual"
    visual.mkdir()
    (visual / "config.json").write_text(json.dumps({"use_tactile_encoder": False}))
    tactile = tmp_path / "tactile"
    tactile.mkdir()
    (tactile / "config.json").write_text(json.dumps({"use_tactile_encoder": True}))

    assert infer._policy_type_from_snapshot(visual) is JaxSmolVLAPolicy
    with pytest.raises(ValueError, match="no longer supported"):
        infer._policy_type_from_snapshot(tactile)


def test_pure_vision_server_config_requests_smolvla_256_profile() -> None:
    observation = {
        "data_type": "vision",
        "language_prompt": "test",
        "single_arm_mode": False,
        "no_state_obs_mode": False,
    }
    control = {
        "control_frequency": 20.0,
        "controller_frequency": 80.0,
        "steps_per_inference": 5,
        "action_horizon": 20,
    }

    config = remote_client._build_server_config(observation, control, frs_policy=None)

    assert config["observation_profile"] == "smolvla_vision_256"


def test_smolvla_right_arm_profile_requires_7d_state_and_10d_action() -> None:
    observation = {
        "state_action_profile": "single-right-arm-7x10",
        "single_arm_mode": True,
        "controlled_arm": "right",
    }
    policy = SimpleNamespace(config=SimpleNamespace(state_dim=7, action_dim=10))

    assert remote_client._validate_state_action_profile(observation, policy) == (
        "single-right-arm-7x10"
    )

    policy.config.action_dim = 20
    with pytest.raises(ValueError, match="does not match"):
        remote_client._validate_state_action_profile(observation, policy)
