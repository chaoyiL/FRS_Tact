import hashlib
import json

import pytest

from deploy_deco.artifact import load_sidecar, validate_metadata


def metadata(payload: bytes) -> dict:
    return {
        "format": "sudo-upstream-deco-stage1-torchscript-v1",
        "camera_names": [
            "observation.images.camera0",
            "observation.images.camera1",
        ],
        "input": {
            "images": [1, 2, 3, 224, 224],
            "images_dtype": "float32",
            "images_range": [0.0, 1.0],
            "observation": [1, 20],
            "state_layout": "relative_start_pose6d_gripper_plus_left_relative_right",
        },
        "output": {
            "action": [1, 32, 20],
            "action_mode": "tcp_delta_absolute_gripper",
            "rotation_representation": "rotation_6d_matrix_columns",
            "gripper_mode": "absolute",
        },
        "normalization": {"embedded": True},
        "expected_sample_hz": 30.0,
        "stochastic": True,
        "torchscript_sha256": hashlib.sha256(payload).hexdigest(),
    }


def right_metadata(payload: bytes) -> dict:
    contract = metadata(payload)
    contract["state_action_profile"] = "single-right-arm-7x10"
    contract["controlled_arms"] = ["right"]
    contract["input"]["observation"] = [1, 7]
    contract["input"]["state_layout"] = "single_right_relative_start_pose6d_gripper"
    contract["output"]["action"] = [1, 32, 10]
    return contract


def test_sidecar_and_hash_are_required(tmp_path):
    payload = b"fake torchscript archive"
    artifact = tmp_path / "policy.ts"
    artifact.write_bytes(payload)
    artifact.with_suffix(".ts.json").write_text(json.dumps(metadata(payload)))
    loaded = load_sidecar(artifact)
    assert loaded["output"]["action"] == [1, 32, 20]


def test_hash_mismatch_is_rejected(tmp_path):
    artifact = tmp_path / "policy.ts"
    artifact.write_bytes(b"changed")
    artifact.with_suffix(".ts.json").write_text(json.dumps(metadata(b"original")))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_sidecar(artifact)


def test_wrong_action_semantics_are_rejected():
    contract = metadata(b"")
    contract["output"]["action_mode"] = "absolute"
    with pytest.raises(ValueError, match="action_mode"):
        validate_metadata(contract)


def test_single_right_arm_contract_is_accepted():
    assert validate_metadata(right_metadata(b""))["controlled_arms"] == ["right"]


def test_single_right_arm_contract_rejects_bimanual_action_width():
    contract = right_metadata(b"")
    contract["output"]["action"] = [1, 32, 20]
    with pytest.raises(ValueError, match="single-right-arm"):
        validate_metadata(contract)
