from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from train_deco.model_factory import build_stage2_model
from train_deco.models.deco.deco import DECO
from train_deco.models.tactile_resnet import TactileResNet18
from train_deco.stage2_initialization import (
    initialize_stage2_from_stage1,
    validate_stage1_checkpoint_contract,
    verify_stage2_stage1_parity,
)


class _TinyImageEncoder(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(1, 2, 3), keepdim=True)
        return pooled.reshape(-1, 1, 1, 1).expand(-1, 512, 2, 2)


class _TinyTactileEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = images.mean(dim=(1, 2, 3))[:, None]
        return values.expand(-1, 512) * self.scale


def _models() -> tuple[DECO, DECO]:
    stage1 = DECO(
        act_dim=4,
        chunk_size=3,
        num_attn_blocks=2,
        heads=4,
        dim=32,
        rope_axes_dim=(4, 4),
        num_cameras=2,
    )
    stage2 = DECO(
        act_dim=4,
        chunk_size=3,
        use_tactile=True,
        tactile_image_mode=True,
        tactile_encoder=_TinyTactileEncoder(),
        plugin=True,
        plugin_rank=7,
        num_attn_blocks=2,
        heads=4,
        dim=32,
        rope_axes_dim=(4, 4),
        num_cameras=2,
    )
    stage1.img_encoder = _TinyImageEncoder()
    stage2.img_encoder = _TinyImageEncoder()
    with torch.no_grad():
        stage1.linear.weight.fill_(0.1)
        stage1.linear.bias.fill_(0.05)
    return stage1, stage2


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return {
        "img1": torch.randn(2, 3, 8, 8, generator=generator),
        "img2": torch.randn(2, 3, 8, 8, generator=generator),
        "obs": torch.randn(2, 4, generator=generator),
        "act": torch.randn(2, 3, 4, generator=generator),
    }


def _save_checkpoint(path: Path, state: dict[str, torch.Tensor]) -> Path:
    torch.save({"model": state, "config": {"model_type": "upstream-deco-stage1"}}, path)
    return path

def _stage_contract() -> dict:
    return {
        "model_type": "upstream-deco-stage1",
        "action_dim": 4,
        "obs_dim": 4,
        "source_obs_dim": 6,
        "chunk_size": 3,
        "camera_names": ["camera0", "camera1"],
        "hidden_dim": 32,
        "layers": 2,
        "heads": 4,
        "image_size": 8,
        "inference_steps": 2,
        "rope_height": 4,
        "rope_width": 4,
        "use_task_condition": False,
        "num_tasks": 2,
        "task_ids": ["pick", "place"],
        "action_mode": "tcp_delta_absolute_gripper",
        "objective_version": "masked-flow-mse-v1",
        "dataset_id": "pick-tube-fixture",
        "observation_indices": [0, 1, 2, 3],
        "state_columns": ["s0", "s1", "s2", "s3", "s4", "s5"],
        "action_columns": ["s0", "s1", "s2", "s3"],
    }


def _normalization_stats() -> dict:
    return {
        "observation_mean": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "observation_std": [1.0] * 6,
        "action_mean": [0.0] * 4,
        "action_std": [1.0] * 4,
    }


@pytest.mark.parametrize(
    "key",
    [
        "model_type", "action_dim", "obs_dim", "source_obs_dim",
        "chunk_size", "camera_names", "hidden_dim", "layers", "heads",
        "image_size", "inference_steps", "rope_height", "rope_width",
        "use_task_condition", "num_tasks", "task_ids", "action_mode",
        "objective_version", "dataset_id", "observation_indices",
        "state_columns", "action_columns",
    ],
)
def test_fresh_stage1_checkpoint_rejects_every_state_contract_mismatch(key: str) -> None:
    config = _stage_contract()
    checkpoint = {
        "model": {}, "config": dict(config), "stats": _normalization_stats()
    }
    current = {**config, "model_type": "upstream-deco-stage2-tactile-image"}
    checkpoint["config"][key] = "wrong"

    with pytest.raises(ValueError, match=key):
        validate_stage1_checkpoint_contract(
            checkpoint, current_config=current, current_stats=_normalization_stats()
        )


@pytest.mark.parametrize(
    "saved_task_ids",
    [
        ["pick", "insert"],
        ["place", "pick"],
    ],
)
def test_fresh_stage1_checkpoint_rejects_different_task_id_mapping_with_same_count(
    saved_task_ids: list[str],
) -> None:
    config = _stage_contract()
    checkpoint = {
        "model": {},
        "config": {**config, "task_ids": saved_task_ids},
        "stats": _normalization_stats(),
    }

    with pytest.raises(ValueError, match="task_ids"):
        validate_stage1_checkpoint_contract(
            checkpoint,
            current_config={
                **config,
                "model_type": "upstream-deco-stage2-tactile-image",
            },
            current_stats=_normalization_stats(),
        )


def test_fresh_stage1_checkpoint_rejects_normalization_stats_mismatch() -> None:
    config = _stage_contract()
    checkpoint = {
        "model": {}, "config": config, "stats": _normalization_stats()
    }
    current = _normalization_stats()
    current["action_std"] = [2.0] * 4

    with pytest.raises(ValueError, match="normalization.*action_std"):
        validate_stage1_checkpoint_contract(
            checkpoint,
            current_config={**config, "model_type": "upstream-deco-stage2-tactile-image"},
            current_stats=current,
        )


def test_production_parity_check_aborts_if_zero_adapter_contract_is_broken() -> None:
    stage1, stage2 = _models()
    stage2.load_state_dict({**stage2.state_dict(), **stage1.state_dict()}, strict=True)
    with torch.no_grad():
        stage2.mmattn[0].img_qkv_pi.up.weight.fill_(0.5)

    with pytest.raises(ValueError, match="zero-initialized PI adapter"):
        verify_stage2_stage1_parity(
            stage1, stage2, inputs=_inputs(),
            tactile_images=torch.rand(2, 4, 3, 224, 224), seed=101,
        )


def test_production_parity_check_uses_fixed_noise_and_accepts_zero_gate_adapter() -> None:
    stage1, stage2 = _models()
    stage2.load_state_dict({**stage2.state_dict(), **stage1.state_dict()}, strict=True)

    report = verify_stage2_stage1_parity(
        stage1, stage2, inputs=_inputs(),
        tactile_images=torch.rand(2, 4, 3, 224, 224), seed=101,
    )

    assert report["seed"] == 101
    assert report["max_abs_prediction"] <= 1e-6



def test_factory_builds_explicit_stage2_with_shared_encoder_and_rank32_default() -> None:
    config = {
        "action_dim": 4,
        "chunk_size": 3,
        "inference_steps": 2,
        "layers": 2,
        "heads": 4,
        "hidden_dim": 32,
        "rope_height": 4,
        "rope_width": 4,
        "camera_names": ["camera0", "camera1"],
    }

    model = build_stage2_model(config, load_backbone=False)

    assert model.tactile_image_mode is True
    assert isinstance(model.tactile_encoder, TactileResNet18)
    assert all(block.img_qkv_pi.down.weight.shape[0] == 32 for block in model.mmattn)
    assert not any(
        isinstance(child, TactileResNet18)
        for block in model.mmattn
        for child in block.modules()
    )


def test_strict_initialization_loads_all_stage1_keys_and_reports_stage2_superset(
    tmp_path: Path,
) -> None:
    stage1, stage2 = _models()
    prefixed = {f"module.{name}": value.clone() for name, value in stage1.state_dict().items()}
    checkpoint = _save_checkpoint(tmp_path / "stage1.pt", prefixed)

    report = initialize_stage2_from_stage1(stage2, checkpoint)

    assert report.loaded_stage1_keys == tuple(stage1.state_dict())
    assert set(report.stage2_only_keys) == set(stage2.state_dict()) - set(stage1.state_dict())
    for name, expected in stage1.state_dict().items():
        assert torch.equal(stage2.state_dict()[name], expected)
    assert set(report.parameters.trainable_by_category) == {
        "sensor_embeddings",
        "tactile_attention",
        "tactile_gates",
        "pi_adapters",
    }
    assert report.parameters.frozen_by_category["tactile_encoder"] == (
        "tactile_encoder.scale",
    )


def test_zero_gate_initialized_stage2_exactly_matches_stage1_for_any_tactile_pixels(
    tmp_path: Path,
) -> None:
    stage1, stage2 = _models()
    initialize_stage2_from_stage1(
        stage2,
        _save_checkpoint(tmp_path / "stage1.pt", stage1.state_dict()),
    )
    stage1.eval()
    stage2.eval()
    inputs = _inputs()
    random_tactile = torch.rand(
        2,
        4,
        3,
        224,
        224,
        generator=torch.Generator().manual_seed(202),
    )

    torch.manual_seed(101)
    stage1_prediction, stage1_noise = stage1(**inputs, training=True)
    torch.manual_seed(101)
    zero_prediction, zero_noise = stage2(
        **inputs,
        tactile_images=torch.zeros(2, 4, 3, 224, 224),
        training=True,
    )
    torch.manual_seed(101)
    random_prediction, random_noise = stage2(
        **inputs,
        tactile_images=random_tactile,
        training=True,
    )

    assert torch.equal(zero_noise, stage1_noise)
    assert torch.equal(random_noise, stage1_noise)
    assert torch.allclose(zero_prediction, stage1_prediction, rtol=1e-6, atol=1e-7)
    assert torch.allclose(random_prediction, stage1_prediction, rtol=1e-6, atol=1e-7)


def test_initialization_freezes_exactly_stage1_and_encoder_parameters(
    tmp_path: Path,
) -> None:
    stage1, stage2 = _models()
    report = initialize_stage2_from_stage1(
        stage2,
        _save_checkpoint(tmp_path / "stage1.pt", stage1.state_dict()),
    )
    trainable = {name for name, parameter in stage2.named_parameters() if parameter.requires_grad}
    reported = {
        name
        for names in report.parameters.trainable_by_category.values()
        for name in names
    }

    assert trainable == reported
    assert "sensor_embeddings.weight" in trainable
    assert all("tactile_encoder" not in name for name in trainable)
    assert all(
        name in trainable
        for name, _ in stage2.named_parameters()
        if ".tactile_key." in name
        or ".tactile_value." in name
        or name.endswith(".tactile_gate")
        or "_pi." in name
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in stage2.named_parameters()
        if name in stage1.state_dict() or name.startswith("tactile_encoder.")
    )


def test_initialized_gradient_boundary_matches_the_trainable_allowlist(
    tmp_path: Path,
) -> None:
    stage1, stage2 = _models()
    initialize_stage2_from_stage1(
        stage2,
        _save_checkpoint(tmp_path / "stage1.pt", stage1.state_dict()),
    )
    stage2.train()

    torch.manual_seed(41)
    prediction, _ = stage2(
        **_inputs(),
        tactile_images=torch.rand(2, 4, 3, 224, 224),
        training=True,
    )
    prediction.square().mean().backward()

    assert any(
        block.tactile_gate.grad is not None
        and block.tactile_gate.grad.abs().item() > 0
        for block in stage2.mmattn
    )
    assert any(
        adapter.up.weight.grad is not None
        and torch.count_nonzero(adapter.up.weight.grad) > 0
        for block in stage2.mmattn
        for adapter in (block.img_qkv_pi, block.act_qkv_pi)
    )
    assert all(
        parameter.grad is None
        for parameter in stage2.parameters()
        if not parameter.requires_grad
    )

    stage2.zero_grad(set_to_none=True)
    with torch.no_grad():
        for block in stage2.mmattn:
            block.tactile_gate.fill_(0.25)
    torch.manual_seed(41)
    prediction, _ = stage2(
        **_inputs(),
        tactile_images=torch.rand(2, 4, 3, 224, 224),
        training=True,
    )
    prediction.square().mean().backward()

    assert any(
        block.tactile_key.weight.grad is not None
        and torch.count_nonzero(block.tactile_key.weight.grad) > 0
        for block in stage2.mmattn
    )
    assert all(
        parameter.grad is None
        for parameter in stage2.parameters()
        if not parameter.requires_grad
    )


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_strict_initialization_rejects_invalid_stage1_state(
    tmp_path: Path,
    corruption: str,
) -> None:
    stage1, stage2 = _models()
    state = {name: value.clone() for name, value in stage1.state_dict().items()}
    if corruption == "missing":
        state.pop(next(iter(state)))
        message = "missing Stage1 keys"
    elif corruption == "unexpected":
        state["legacy.tactile_encoder.weight"] = torch.zeros(1)
        message = "unexpected Stage1 keys"
    else:
        name = next(name for name, value in state.items() if value.ndim > 0)
        state[name] = torch.zeros(state[name].numel() + 1)
        message = "shape mismatch"

    with pytest.raises(ValueError, match=message):
        initialize_stage2_from_stage1(
            stage2,
            _save_checkpoint(tmp_path / f"{corruption}.pt", state),
        )


def test_strict_initialization_does_not_strip_unknown_prefix(tmp_path: Path) -> None:
    stage1, stage2 = _models()
    state = {f"model.{name}": value.clone() for name, value in stage1.state_dict().items()}

    with pytest.raises(ValueError, match="missing Stage1 keys.*unexpected Stage1 keys"):
        initialize_stage2_from_stage1(
            stage2,
            _save_checkpoint(tmp_path / "unknown-prefix.pt", state),
        )


def test_strict_initialization_requires_training_checkpoint_model_wrapper(
    tmp_path: Path,
) -> None:
    stage1, stage2 = _models()
    path = tmp_path / "raw-state.pt"
    torch.save(stage1.state_dict(), path)

    with pytest.raises(ValueError, match="training checkpoint.*'model'"):
        initialize_stage2_from_stage1(stage2, path)
