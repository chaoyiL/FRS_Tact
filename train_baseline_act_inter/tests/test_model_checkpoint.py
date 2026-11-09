"""Behavioral coverage for the standalone direct tactile decoder."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest
import torch

from train_baseline_act_inter.config import DecoderTrainConfig, RIGHT_TACTILE_KEYS, TACTILE_KEYS
from train_baseline_act_inter.model import (
    DirectDecoderConfig,
    DirectTactileActionDecoder,
    masked_smooth_l1,
)
from train_baseline_act_inter.checkpoint import (
    load_decoder_checkpoint,
    save_best_checkpoint,
    save_last_checkpoint,
)


def _config(tmp_path: Path) -> DirectDecoderConfig:
    train_config = DecoderTrainConfig(output=tmp_path / "outputs")
    return DirectDecoderConfig.from_train_config(train_config)


def test_decoder_matches_tacfrs_flow_architecture(tmp_path):
    model = DirectTactileActionDecoder(_config(tmp_path))
    assert len(model.blocks) == 6
    assert model.action_in.out_features == 256
    assert model.blocks[0].self_attention.num_heads == 4
    assert model.blocks[0].fc1.out_features == 1024
    assert model.blocks[0].self_attention.dropout == 0.0
    assert model.tactile_gru.dense_i.in_features == 512
    assert model.tactile_gru.hidden_dim == 256
    assert model.state_fc1.in_features == 20


def test_decoder_velocity_and_integrated_action_interfaces(tmp_path):
    model = DirectTactileActionDecoder(_config(tmp_path)).eval()
    coarse = torch.randn(1,50,20)
    tactile = torch.randn(1,10,4,512)
    state = torch.randn(1,20)
    with torch.no_grad():
        velocity = model(coarse, torch.tensor([0.3]), tactile, state=state)
        decoded = model.decode(coarse,tactile,state=state)
    assert velocity.shape == decoded.shape == coarse.shape
    assert torch.isfinite(decoded).all()
    torch.testing.assert_close(decoded[...,9::10], coarse[...,9::10], rtol=0,atol=0)


def test_zero_velocity_keeps_vla_origin_instead_of_returning_zero(tmp_path):
    model = DirectTactileActionDecoder(_config(tmp_path)).eval()
    with torch.no_grad():
        model.action_out.weight.zero_(); model.action_out.bias.zero_()
        coarse = torch.randn(1,50,20)
        decoded = model.decode(coarse,torch.randn(1,10,4,512),state=torch.randn(1,20))
    torch.testing.assert_close(decoded,coarse,rtol=0,atol=0)


def test_masked_smooth_l1_ignores_invalid_action_tail():
    predicted = torch.zeros(1, 3, 1)
    target = torch.tensor([[[1.0], [1.0], [1000.0]]])
    valid_mask = torch.tensor([[True, True, False]])

    loss = masked_smooth_l1(predicted, target, valid_mask)

    assert loss == pytest.approx(0.5)


def _source_contract() -> dict[str, object]:
    return {"checkpoint": "reference/pi05", "seed": 0, "sample_steps": 10}


def test_best_checkpoint_round_trips_with_weights_only_and_strict_state(tmp_path: Path):
    config = _config(tmp_path)
    model = DirectTactileActionDecoder(config)
    path = save_best_checkpoint(
        tmp_path, model, config, epoch=3, global_step=12,
        metrics={"validation_loss": 0.25}, source_contract=_source_contract(),
    )

    raw = torch.load(path, weights_only=True)
    loaded, metadata = load_decoder_checkpoint(path)

    assert path == tmp_path / "best.pt"
    assert raw["run_kind"] == "formal"
    assert raw["mode"] == "action_origin_flow"
    assert raw["decoder_config"]["tactile_keys"] == list(TACTILE_KEYS)
    assert metadata["epoch"] == 3
    assert metadata["global_step"] == 12
    for key, value in model.state_dict().items():
        assert torch.equal(value.cpu(), loaded.state_dict()[key].cpu())


def test_single_arm_right_tactile_checkpoint_preserves_predictions(tmp_path: Path):
    keys = ("observation.images.tactile_left_1", "observation.images.tactile_right_1")
    config = DirectDecoderConfig(action_dim=10, tactile_keys=keys)
    model = DirectTactileActionDecoder(config).eval()
    coarse, tactile = torch.randn(2, 50, 10), torch.randn(2, 10, 2, 512)
    with torch.no_grad():
        state = torch.randn(2,7)
        expected = model.decode(coarse, tactile, state=state)
    path = save_best_checkpoint(
        tmp_path, model, config, epoch=0, global_step=0,
        metrics={"validation_loss": 1.0}, source_contract=_source_contract(),
    )

    loaded, metadata = load_decoder_checkpoint(path)
    loaded.eval()
    with torch.no_grad():
        actual = loaded.decode(coarse, tactile, state=state)

    assert actual.shape == (2, 50, 10)
    assert loaded.config.tactile_keys == keys
    assert metadata["decoder_config"]["action_dim"] == 10
    torch.testing.assert_close(actual, expected)


def test_checkpoint_loader_rejects_old_cross_arm_sensor_contract(tmp_path):
    config = DirectDecoderConfig(action_dim=10, tactile_keys=RIGHT_TACTILE_KEYS)
    path = save_best_checkpoint(
        tmp_path, DirectTactileActionDecoder(config), config, epoch=0, global_step=0,
        metrics={"validation_loss": 1.0}, source_contract=_source_contract(),
    )
    raw = torch.load(path, weights_only=True)
    raw["decoder_config"]["tactile_keys"] = [
        "observation.images.tactile_right_0", "observation.images.tactile_right_1",
    ]
    torch.save(raw, path)
    with pytest.raises(ValueError, match="tactile_keys"):
        load_decoder_checkpoint(path)


def test_best_checkpoint_canonicalizes_nested_source_contract_containers(tmp_path: Path):
    config = _config(tmp_path)
    source_contract = {
        "checkpoint": "reference/pi05",
        "nested": defaultdict(int, {"count": 1}),
    }
    path = save_best_checkpoint(
        tmp_path, DirectTactileActionDecoder(config), config, epoch=0, global_step=0,
        metrics={"validation_loss": 1.0}, source_contract=source_contract,
    )

    raw = torch.load(path, weights_only=True)

    assert raw["source_contract"] == {"checkpoint": "reference/pi05", "nested": {"count": 1}}


@pytest.mark.parametrize("mutation", ["schema", "config", "state"])
def test_loader_rejects_bad_checkpoint_contract(tmp_path: Path, mutation: str):
    config = _config(tmp_path)
    path = save_best_checkpoint(
        tmp_path, DirectTactileActionDecoder(config), config, epoch=0, global_step=0,
        metrics={"validation_loss": 1.0}, source_contract=_source_contract(),
    )
    raw = torch.load(path, weights_only=True)
    if mutation == "schema":
        raw["run_kind"] = "informal"
    elif mutation == "config":
        raw["decoder_config"]["num_layers"] = 3
    else:
        raw["decoder_state"].pop(next(iter(raw["decoder_state"])))
    torch.save(raw, path)

    with pytest.raises(ValueError):
        load_decoder_checkpoint(path)


def test_loader_rejects_wrong_type_in_fixed_decoder_config(tmp_path: Path):
    config = _config(tmp_path)
    path = save_best_checkpoint(
        tmp_path, DirectTactileActionDecoder(config), config, epoch=0, global_step=0,
        metrics={"validation_loss": 1.0}, source_contract=_source_contract(),
    )
    raw = torch.load(path, weights_only=True)
    raw["decoder_config"]["num_layers"] = 2.0
    torch.save(raw, path)

    with pytest.raises(ValueError, match="num_layers"):
        load_decoder_checkpoint(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_horizon", 50.0),
        ("num_layers", 2.0),
        ("nhead", True),
        ("tactile_keys", list(TACTILE_KEYS)),
    ],
)
def test_direct_decoder_config_rejects_wrong_fixed_field_types(field: str, value: object):
    config = DirectDecoderConfig(**{field: value})

    with pytest.raises(ValueError, match=field):
        DirectTactileActionDecoder(config)


def test_last_checkpoint_keeps_optimizer_and_resume_state(tmp_path: Path):
    config = _config(tmp_path)
    model = DirectTactileActionDecoder(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_last_checkpoint(
        tmp_path, model, config, epoch=2, global_step=9,
        metrics={"validation_loss": 0.75}, source_contract=_source_contract(),
        optimizer=optimizer, scheduler_state={"last_epoch": 2},
        rng_state={"torch": torch.get_rng_state()}, best_state={"validation_loss": 0.25},
    )

    raw = torch.load(path, weights_only=True)
    assert path == tmp_path / "last.pt"
    assert raw["optimizer_state"] == optimizer.state_dict()
    assert raw["scheduler_state"] == {"last_epoch": 2}
    assert raw["best_state"] == {"validation_loss": 0.25}


@pytest.mark.parametrize("field", ["scheduler_state", "rng_state", "best_state"])
def test_last_checkpoint_rejects_nested_non_weights_only_payloads(tmp_path: Path, field: str):
    config = _config(tmp_path)
    keyword = {field: {"nested": {"path": tmp_path}}}

    with pytest.raises(ValueError, match=field):
        save_last_checkpoint(
            tmp_path, DirectTactileActionDecoder(config), config, epoch=2, global_step=9,
            metrics={"validation_loss": 0.75}, source_contract=_source_contract(), **keyword,
        )
