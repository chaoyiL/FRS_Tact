import json

import pytest
import torch
from torch import nn

import train_deco.export_torchscript as export_module
from train_deco.export_torchscript import (
    EXPORT_FORMAT,
    STAGE2_EXPORT_FORMAT,
    Stage2DECODeployment,
    export_checkpoint,
    export_policy,
)
from train_deco.lerobot_vision_dataset import TACTILE_NAMES
from train_deco.model_factory import (
    MODEL_TYPE,
    STAGE2_MODEL_TYPE,
    build_model,
    build_stage2_model,
)


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


class TinyTactileEncoder(nn.Module):
    """Fast 512-D image encoder preserving the Stage2 state-dict boundary."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 512, bias=False)

    def forward(self, images):
        return self.projection(images.mean(dim=(-2, -1)))


def stage2_config():
    return config() | {
        "model_type": STAGE2_MODEL_TYPE,
        "stage": 2,
        "tactile_adapter_rank": 4,
        "tactile_field_order": list(TACTILE_NAMES),
    }


def stage2_checkpoint(model):
    gate_values = {
        name: float(parameter.detach())
        for name, parameter in model.named_parameters()
        if name.endswith(".tactile_gate")
    }
    return {
        "model": model.state_dict(),
        "config": stage2_config(),
        "stats": stats(),
        "epoch": 11,
        "val_loss": 0.0625,
        "stage": 2,
        "model_type": STAGE2_MODEL_TYPE,
        "checkpoint_schema_version": 1,
        "stage2_metadata": {
            "model_type": STAGE2_MODEL_TYPE,
            "tactile_field_order": list(TACTILE_NAMES),
            "tactile_encoder": {
                "source_sha256": "1" * 64,
                "artifact_sha256": "2" * 64,
                "artifact_path": "/cache/tactile.safetensors",
                "metadata_path": "/cache/metadata.json",
                "architecture": "resnet18",
                "embedding_dim": 512,
            },
            "tactile_adapter_rank": 4,
            "gate_values": gate_values,
            "parameter_categories": {"trainable": {}, "frozen": {}},
            "parameter_counts": {"total": 2, "trainable": 1},
            "stage1_checkpoint": {"path": "/stage1.pt", "sha256": "3" * 64},
        },
    }


def patch_tiny_stage2(monkeypatch):
    def build(config, load_backbone=False):
        return build_stage2_model(
            config,
            load_backbone=load_backbone,
            tactile_encoder=TinyTactileEncoder(),
        )

    monkeypatch.setattr(export_module, "build_stage2_model", build)


def make_stage2_model():
    model = build_stage2_model(
        stage2_config(),
        load_backbone=False,
        tactile_encoder=TinyTactileEncoder(),
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".tactile_gate"):
                parameter.fill_(0.5)
    return model.eval()


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config(), load_backbone=False).to(device).train()
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    output = tmp_path / "direct.ts"
    metadata = export_policy(model, stats(), config(), output, 32, 32, 3, 0.25)
    assert output.is_file()
    assert metadata["source"] == "training_loop"
    assert not list(tmp_path.glob("*.pt"))
    assert model.training
    assert next(model.parameters()).device.type == device.type
    assert all(
        torch.equal(before[name], value) for name, value in model.state_dict().items()
    )
    exported = torch.jit.load(str(output), map_location="cpu").eval()
    with torch.inference_mode():
        action = exported(torch.zeros(1, 2, 3, 32, 32), torch.zeros(1, 3))
    assert action.shape == (1, 4, 2)
    assert all(value.device.type == "cpu" for value in exported.parameters())


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
def test_three_camera_cuda_policy_exports_for_cpu_when_available(tmp_path):
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
        str(output), _extra_files=extra, map_location="cpu"
    ).eval()
    action = exported(
        torch.zeros(1, 3, 3, 32, 32),
        torch.zeros(1, 3),
    )
    assert action.shape == (1, 4, 2)
    assert torch.isfinite(action).all()


def test_stage2_checkpoint_exports_three_input_six_stream_cpu_contract(
    tmp_path, monkeypatch
):
    patch_tiny_stage2(monkeypatch)
    model = make_stage2_model()
    checkpoint = tmp_path / "stage2.pt"
    torch.save(stage2_checkpoint(model), checkpoint)

    output = tmp_path / "stage2.ts"
    metadata = export_checkpoint(checkpoint, output, 32, 32, device="cuda")

    extra = {"deco_metadata.json": ""}
    exported = torch.jit.load(
        str(output), _extra_files=extra, map_location="cpu"
    ).eval()
    embedded = json.loads(extra["deco_metadata.json"])
    visual = torch.rand(1, 2, 3, 32, 32)
    tactile = torch.rand(1, 4, 3, 18, 24)
    observation = torch.rand(1, 3)
    with torch.inference_mode():
        action = exported(visual, tactile, observation)

    assert action.shape == (1, 4, 2)
    assert torch.isfinite(action).all()
    assert metadata["format"] == STAGE2_EXPORT_FORMAT
    assert embedded["input"]["images"] == [1, 2, 3, 32, 32]
    assert embedded["input"]["tactile_images"] == [1, 4, 3, 32, 32]
    assert embedded["input"]["stream_order"] == [
        "left",
        "right",
        *TACTILE_NAMES,
    ]
    assert embedded["preprocessing"]["visual"]["normalization"] == "imagenet"
    assert embedded["preprocessing"]["tactile"]["normalization"] is None
    assert embedded["preprocessing"]["tactile"]["padding_value"] == 0.0
    assert embedded["preprocessing"]["tactile"]["target_size"] == [224, 224]
    assert embedded["checkpoint_schema_version"] == 1
    assert embedded["tactile_field_order"] == list(TACTILE_NAMES)
    assert embedded["tactile_encoder"]["source_sha256"] == "1" * 64
    assert embedded["tactile_encoder"]["artifact_sha256"] == "2" * 64
    assert embedded["tactile_adapter_rank"] == 4
    assert embedded["gate_values"] == (
        stage2_checkpoint(model)["stage2_metadata"]["gate_values"]
    )
    assert embedded["normalization"]["statistics"] == stats()
    assert all(value.device.type == "cpu" for value in exported.parameters())


def test_stage2_export_matches_eager_and_keeps_tactile_in_graph(
    tmp_path, monkeypatch
):
    patch_tiny_stage2(monkeypatch)
    torch.manual_seed(7)
    model = make_stage2_model()
    checkpoint = tmp_path / "stage2.pt"
    torch.save(stage2_checkpoint(model), checkpoint)
    output = tmp_path / "stage2.ts"
    export_checkpoint(checkpoint, output, 32, 32)
    exported = torch.jit.load(str(output), map_location="cpu").eval()

    eager = Stage2DECODeployment(model, stats(), stage2_config()).eval()
    generator = torch.Generator().manual_seed(19)
    visual = torch.rand(1, 2, 3, 32, 32, generator=generator)
    tactile = torch.rand(1, 4, 3, 20, 28, generator=generator)
    observation = torch.rand(1, 3, generator=generator)
    with torch.inference_mode():
        torch.manual_seed(23)
        eager_action = eager(visual, tactile, observation)
        torch.manual_seed(23)
        exported_action = exported(visual, tactile, observation)
        torch.manual_seed(29)
        tactile_action = exported(visual, tactile, observation)
        torch.manual_seed(29)
        zero_tactile_action = exported(
            visual, torch.zeros_like(tactile), observation
        )

    # Trace and eager execute the same float32 graph; allow only accumulated
    # floating-point roundoff through the iterative denoising loop.
    torch.testing.assert_close(
        exported_action, eager_action, rtol=1e-5, atol=1e-6
    )
    assert not torch.allclose(
        tactile_action, zero_tactile_action, rtol=1e-6, atol=1e-7
    )


def test_stage2_export_strictly_rejects_missing_model_state(
    tmp_path, monkeypatch
):
    patch_tiny_stage2(monkeypatch)
    payload = stage2_checkpoint(make_stage2_model())
    payload["model"].pop(next(iter(payload["model"])))
    checkpoint = tmp_path / "missing-state.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        export_checkpoint(checkpoint, tmp_path / "policy.ts", 32, 32)


def test_stage2_export_rejects_inconsistent_checkpoint_schema(
    tmp_path, monkeypatch
):
    patch_tiny_stage2(monkeypatch)
    payload = stage2_checkpoint(make_stage2_model())
    payload["stage2_metadata"]["tactile_adapter_rank"] = 8
    checkpoint = tmp_path / "bad-schema.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="adapter rank"):
        export_checkpoint(checkpoint, tmp_path / "policy.ts", 32, 32)


def test_stage2_export_rejects_incomplete_checkpoint_metadata(
    tmp_path, monkeypatch
):
    patch_tiny_stage2(monkeypatch)
    payload = stage2_checkpoint(make_stage2_model())
    payload["stage2_metadata"].pop("parameter_counts")
    checkpoint = tmp_path / "incomplete-schema.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="parameter_counts"):
        export_checkpoint(checkpoint, tmp_path / "policy.ts", 32, 32)
