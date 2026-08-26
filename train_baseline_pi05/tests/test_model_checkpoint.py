"""Behavioral coverage for the standalone direct tactile decoder."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest
import torch

from train_baseline_pi05.config import DecoderTrainConfig, TACTILE_KEYS
from train_baseline_pi05.model import (
    DirectDecoderConfig,
    DirectTactileActionDecoder,
    masked_smooth_l1,
)
from train_baseline_pi05.checkpoint import (
    load_decoder_checkpoint,
    save_best_checkpoint,
    save_last_checkpoint,
)


def _config(tmp_path: Path) -> DirectDecoderConfig:
    train_config = DecoderTrainConfig(output=tmp_path / "outputs")
    return DirectDecoderConfig.from_train_config(train_config)


def test_decoder_has_exactly_two_required_transformer_layers(tmp_path: Path):
    model = DirectTactileActionDecoder(_config(tmp_path))

    assert len(model.decoder.layers) == 2
    for layer in model.decoder.layers:
        assert isinstance(layer, torch.nn.TransformerDecoderLayer)
        assert layer.self_attn.embed_dim == 128
        assert layer.self_attn.num_heads == 4
        assert layer.linear1.out_features == 256
        assert layer.dropout.p == pytest.approx(0.1)
        assert layer.activation is torch.nn.functional.relu
        assert layer.self_attn.batch_first is True
        assert layer.norm_first is True


def test_decoder_predicts_full_finite_action_sequence(tmp_path: Path):
    model = DirectTactileActionDecoder(_config(tmp_path))

    result = model(torch.randn(3, 50, 20), torch.randn(3, 4, 512))

    assert result.shape == (3, 50, 20)
    assert torch.isfinite(result).all()


def test_decoder_accepts_float16_zero_tactile_tokens_with_float32_parameters(tmp_path: Path):
    model = DirectTactileActionDecoder(_config(tmp_path))

    result = model(
        torch.zeros(1, 50, 20, dtype=torch.float16),
        torch.zeros(1, 4, 512, dtype=torch.float16),
    )

    assert result.shape == (1, 50, 20)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize(
    ("coarse_shape", "tactile_shape", "error"),
    [
        ((3, 49, 20), (3, 4, 512), "coarse"),
        ((3, 50, 19), (3, 4, 512), "coarse"),
        ((3, 50, 20), (3, 3, 512), "tactile"),
        ((3, 50, 20), (2, 4, 512), "batch"),
    ],
)
def test_decoder_rejects_contract_shape_violations(
    tmp_path: Path, coarse_shape: tuple[int, ...], tactile_shape: tuple[int, ...], error: str
):
    model = DirectTactileActionDecoder(_config(tmp_path))

    with pytest.raises(ValueError, match=error):
        model(torch.randn(coarse_shape), torch.randn(tactile_shape))


def test_tactile_rms_normalization_is_per_token(tmp_path: Path):
    model = DirectTactileActionDecoder(_config(tmp_path))
    tactile = torch.tensor([[[3.0, 4.0], [5.0, 12.0]]])

    normalized = model.normalize_tactile(tactile)

    assert torch.allclose(
        torch.sqrt(normalized.square().mean(dim=-1)), torch.ones(1, 2), atol=1e-6
    )


def test_decoder_does_not_add_prediction_residual_to_coarse_actions(tmp_path: Path):
    model = DirectTactileActionDecoder(_config(tmp_path))
    coarse = torch.randn(1, 50, 20)
    with torch.no_grad():
        model.output_head.weight.zero_()
        model.output_head.bias.zero_()

    predicted = model(coarse, torch.randn(1, 4, 512))

    assert torch.equal(predicted, torch.zeros_like(predicted))
    assert not torch.equal(predicted, coarse)


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
    assert raw["mode"] == "action_tactile"
    assert raw["decoder_config"]["tactile_keys"] == list(TACTILE_KEYS)
    assert metadata["epoch"] == 3
    assert metadata["global_step"] == 12
    for key, value in model.state_dict().items():
        assert torch.equal(value.cpu(), loaded.state_dict()[key].cpu())


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
