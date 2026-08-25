import pytest

from types import SimpleNamespace

from train_deco.train import (
    action_mode_config_fields,
    resolve_dataset_action_mode,
    training_dataset_source,
)


def test_accepts_precomputed_tcp_delta_with_absolute_gripper():
    contract = {"action_mode": "tcp_delta_absolute_gripper"}
    assert resolve_dataset_action_mode(contract) == "tcp_delta_absolute_gripper"


def test_legacy_delta_flag_must_not_describe_mixed_action_mode():
    contract = {
        "action_mode": "tcp_delta_absolute_gripper",
        "use_delta_action": True,
    }
    with pytest.raises(ValueError, match="use_delta_action"):
        resolve_dataset_action_mode(contract)


def test_mixed_action_checkpoint_omits_legacy_delta_flag():
    assert action_mode_config_fields("tcp_delta_absolute_gripper") == {}
    assert action_mode_config_fields("delta") == {"use_delta_action": True}
    assert action_mode_config_fields("absolute") == {"use_delta_action": False}


def test_lerobot_training_accepts_multiroot_manifest():
    args = SimpleNamespace(
        dataset_format="lerobot-v21",
        dataset_dir=None,
        dataset_manifest="/tmp/pick-tube-01-06.json",
    )
    assert training_dataset_source(args) == "/tmp/pick-tube-01-06.json"


def test_preprocessed_training_rejects_multiroot_manifest():
    args = SimpleNamespace(
        dataset_format="preprocessed",
        dataset_dir="/tmp/preprocessed",
        dataset_manifest="/tmp/pick-tube-01-06.json",
    )
    with pytest.raises(ValueError, match="dataset-manifest"):
        training_dataset_source(args)
