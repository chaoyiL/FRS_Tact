import json

from deploy_smolvla import remote_client
from tools import infer_smolvla_jax as infer


def test_policy_type_is_selected_from_checkpoint_config(tmp_path) -> None:
    from train_smolvla import JaxSmolVLAPolicy
    from train_vtsmolvla import VTJaxSmolVLAPolicy

    visual = tmp_path / "visual"
    visual.mkdir()
    (visual / "config.json").write_text(json.dumps({"use_tactile_encoder": False}))
    tactile = tmp_path / "tactile"
    tactile.mkdir()
    (tactile / "config.json").write_text(json.dumps({"use_tactile_encoder": True}))

    assert infer._policy_type_from_snapshot(visual) is JaxSmolVLAPolicy
    assert infer._policy_type_from_snapshot(tactile) is VTJaxSmolVLAPolicy


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
