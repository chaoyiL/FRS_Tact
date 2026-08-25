import json

import torch

from train_deco.export_torchscript import (
    EXPORT_FORMAT,
    export_checkpoint,
    export_policy,
)
from train_deco.model_factory import MODEL_TYPE, build_model


def config(camera_names=None):
    return {
        "model_type": MODEL_TYPE,
        "source_obs_dim": 3,
        "obs_dim": 2,
        "action_dim": 2,
        "chunk_size": 4,
        "observation_indices": [0, 2],
        "hidden_dim": 32,
        "layers": 1,
        "heads": 4,
        "image_size": 32,
        "inference_steps": 2,
        "rope_height": 32,
        "rope_width": 32,
        "use_task_condition": False,
        "num_tasks": 1,
        "dataset_id": "test-dataset",
        "action_mode": "delta",
        "expected_sample_hz": 20.0,
        "camera_names": camera_names or ["left", "right"],
    }


def stats():
    return {
        "observation_mean": [1.0, 2.0, 3.0],
        "observation_std": [2.0, 2.0, 2.0],
        "action_mean": [0.5, -0.5],
        "action_std": [0.25, 0.5],
    }


def test_export_contains_upstream_graph_weights_stats_and_metadata(tmp_path):
    model = build_model(config(), load_backbone=False)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({
        "model": model.state_dict(), "config": config(), "stats": stats(),
        "epoch": 7, "val_loss": 0.125,
    }, checkpoint)
    output = tmp_path / "policy.ts"
    metadata = export_checkpoint(checkpoint, output, 32, 32)
    assert output.is_file()
    assert metadata["format"] == EXPORT_FORMAT
    assert metadata["upstream_model"] == "train_deco.models.deco.deco.DECO"
    extra = {"deco_metadata.json": ""}
    exported = torch.jit.load(str(output), _extra_files=extra).eval()
    embedded = json.loads(extra["deco_metadata.json"])
    assert embedded["input"]["observation"] == [1, 3]
    assert embedded["output"]["action_mode"] == "delta"
    assert embedded["output"]["action_space"] == "denormalized delta robot action (target-current)"
    assert embedded["expected_sample_hz"] == 20.0
    with torch.inference_mode():
        action = exported(torch.zeros(1, 2, 3, 32, 32), torch.zeros(1, 3))
    assert action.shape == (1, 4, 2)
    assert torch.isfinite(action).all()


def test_live_policy_exports_without_intermediate_checkpoint(tmp_path):
    model = build_model(config(), load_backbone=False)
    output = tmp_path / "direct.ts"
    metadata = export_policy(model, stats(), config(), output, 32, 32, 3, 0.25)
    assert output.is_file()
    assert metadata["source"] == "training_loop"
    assert not list(tmp_path.glob("*.pt"))


def test_three_camera_torchscript_accepts_three_views(tmp_path):
    three_camera_config = config(
        ["left_hand_center", "right_hand_center", "chest_left"]
    )
    model = build_model(three_camera_config, load_backbone=False)
    output = tmp_path / "three-camera-policy.ts"
    export_policy(
        model, stats(), three_camera_config, output, 32, 32, 3, 0.25
    )
    exported = torch.jit.load(str(output)).eval()
    with torch.inference_mode():
        action = exported(
            torch.zeros(1, 3, 3, 32, 32), torch.zeros(1, 3)
        )
    assert action.shape == (1, 4, 2)
    assert torch.isfinite(action).all()


def test_camera_id_buffer_preserves_checkpoint_contract_and_follows_device():
    model = build_model(
        config(["left_hand_center", "right_hand_center", "chest_left"]),
        load_backbone=False,
    )
    buffers = dict(model.named_buffers())
    assert "_camera_ids" in buffers
    assert buffers["_camera_ids"].device == next(model.parameters()).device
    assert "_camera_ids" not in model.state_dict()


def test_pick_tube_export_describes_mixed_action_semantics(tmp_path):
    pick_tube_config = config()
    pick_tube_config |= {
        "source_obs_dim": 20,
        "obs_dim": 20,
        "action_dim": 20,
        "observation_indices": list(range(20)),
        "action_mode": "tcp_delta_absolute_gripper",
        "state_layout": "relative_start_pose6d_gripper_plus_left_relative_right",
        "rotation_representation": "rotation_6d_matrix_columns",
        "terminal_action_policy": "excluded",
        "state_columns": [f"state-{index}" for index in range(20)],
        "action_columns": [f"action-{index}" for index in range(20)],
        "gripper_mode": "absolute",
        "statistics_source": "train_episodes_nonterminal_rows_once",
    }
    pick_tube_stats = {
        "observation_mean": [0.0] * 20,
        "observation_std": [1.0] * 20,
        "action_mean": [0.0] * 20,
        "action_std": [1.0] * 20,
    }
    model = build_model(pick_tube_config, load_backbone=False)
    metadata = export_policy(
        model,
        pick_tube_stats,
        pick_tube_config,
        tmp_path / "pick-tube.ts",
        32,
        32,
        1,
        0.5,
    )
    assert metadata["output"]["action_mode"] == "tcp_delta_absolute_gripper"
    assert "TCP-frame" in metadata["output"]["action_space"]
    assert "absolute gripper" in metadata["output"]["action_space"]
    assert metadata["input"]["state_layout"] == pick_tube_config["state_layout"]
    assert metadata["input"]["state_columns"] == pick_tube_config["state_columns"]
    assert metadata["output"]["rotation_representation"] == "rotation_6d_matrix_columns"
    assert metadata["output"]["terminal_action_policy"] == "excluded"
    assert metadata["output"]["action_columns"] == pick_tube_config["action_columns"]
    assert metadata["output"]["gripper_mode"] == "absolute"
    assert metadata["normalization"]["statistics_source"] == "train_episodes_nonterminal_rows_once"


@torch.no_grad()
def test_three_camera_torchscript_exports_on_cuda_when_available(tmp_path):
    if not torch.cuda.is_available():
        return
    three_camera_config = config(
        ["left_hand_center", "right_hand_center", "chest_left"]
    )
    model = build_model(three_camera_config, load_backbone=False).cuda()
    output = tmp_path / "three-camera-policy-cuda.ts"
    export_policy(
        model, stats(), three_camera_config, output, 32, 32, 3, 0.25
    )
    extra = {"deco_metadata.json": ""}
    exported = torch.jit.load(
        str(output), _extra_files=extra, map_location="cuda"
    ).eval()
    action = exported(
        torch.zeros(1, 3, 3, 32, 32, device="cuda"),
        torch.zeros(1, 3, device="cuda"),
    )
    assert action.shape == (1, 4, 2)
    assert torch.isfinite(action).all()
