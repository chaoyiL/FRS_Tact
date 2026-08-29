import json

import pytest
import torch

from train_deco.export_torchscript import export_policy
from train_deco.model_factory import MODEL_TYPE, build_model


def test_bread_phase_export_helper_is_available():
    from train_deco.bread_phase.export import export_bread_phase_checkpoint

    assert callable(export_bread_phase_checkpoint)


def _config():
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
        "use_task_condition": True,
        "num_tasks": 2,
        "bread_phase_version": "bread-phase-v1",
        "dataset_id": "bread",
        "action_mode": "delta",
        "expected_sample_hz": 20.0,
        "camera_names": ["left", "right"],
    }


def _stats():
    return {
        "observation_mean": [0.0, 0.0, 0.0],
        "observation_std": [1.0, 1.0, 1.0],
        "action_mean": [0.0, 0.0],
        "action_std": [1.0, 1.0],
    }


def _export(tmp_path):
    config = _config()
    output = tmp_path / "bread_phase.ts"
    model = build_model(config, load_backbone=False)
    # DECO intentionally starts with a zero output projection.  Seed this tiny
    # contract model as if it had trained so task conditioning reaches outputs.
    with torch.no_grad():
        model.linear.weight.fill_(0.01)
    metadata = export_policy(
        model,
        _stats(),
        config,
        output,
        32,
        32,
        1,
        0.5,
    )
    return torch.jit.load(str(output)).eval(), metadata, output


def test_bread_phase_export_accepts_phase_input_and_writes_contract(tmp_path):
    exported, metadata, output = _export(tmp_path)
    extra = {"deco_metadata.json": ""}
    torch.jit.load(str(output), _extra_files=extra).eval()
    embedded = json.loads(extra["deco_metadata.json"])

    action = exported(
        torch.zeros(1, 2, 3, 32, 32),
        torch.zeros(1, 3),
        torch.tensor([0], dtype=torch.long),
    )

    assert action.shape == (1, 4, 2)
    assert metadata["input"]["phase_id"] == [1]
    assert metadata["phase_count"] == 2
    assert metadata["phase_labels"] == {"0": "right_bread", "1": "left_ketchup"}
    assert embedded["phase_count"] == 2


def test_phase_id_changes_prediction_for_the_same_exported_weight(tmp_path):
    exported, _, _ = _export(tmp_path)
    images = torch.zeros(1, 2, 3, 32, 32)
    observation = torch.zeros(1, 3)

    torch.manual_seed(7)
    phase_zero = exported(images, observation, torch.tensor([0], dtype=torch.long))
    torch.manual_seed(7)
    phase_one = exported(images, observation, torch.tensor([1], dtype=torch.long))

    assert not torch.equal(phase_zero, phase_one)


def test_non_bread_task_conditioned_export_remains_rejected(tmp_path):
    config = _config() | {"bread_phase_version": None}
    with pytest.raises(ValueError, match="task-conditioned"):
        export_policy(
            build_model(config, load_backbone=False),
            _stats(),
            config,
            tmp_path / "not-bread.ts",
            32,
            32,
            1,
            0.5,
        )
