from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn

from train_deco.models.deco.deco import DECO


class _TinyImageEncoder(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(-2, -1), keepdim=True)
        pooled = pooled.mean(dim=1, keepdim=True)
        return pooled.expand(-1, 512, 2, 2)


class _RecordingTactileEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.calls: list[tuple[int, ...]] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(images.shape))
        value = images.mean(dim=(1, 2, 3), keepdim=False)[:, None]
        return (value + 1.0).expand(-1, 512) * self.scale


def _stage1_model() -> DECO:
    model = DECO(
        act_dim=4,
        chunk_size=3,
        num_attn_blocks=2,
        heads=4,
        dim=32,
        rope_axes_dim=(4, 4),
        num_cameras=2,
    )
    model.img_encoder = _TinyImageEncoder()
    return model


def _stage2_model(*, rank: int = 7) -> tuple[DECO, _RecordingTactileEncoder]:
    encoder = _RecordingTactileEncoder()
    model = DECO(
        act_dim=4,
        chunk_size=3,
        use_tactile=True,
        tactile_image_mode=True,
        tactile_encoder=encoder,
        plugin=True,
        plugin_rank=rank,
        num_attn_blocks=2,
        heads=4,
        dim=32,
        rope_axes_dim=(4, 4),
        num_cameras=2,
    )
    model.img_encoder = _TinyImageEncoder()
    return model, encoder


def _inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(123)
    return {
        "img1": torch.randn(batch_size, 3, 8, 8, generator=generator),
        "img2": torch.randn(batch_size, 3, 8, 8, generator=generator),
        "obs": torch.randn(batch_size, 4, generator=generator),
        "act": torch.randn(batch_size, 3, 4, generator=generator),
        "tactile_images": torch.zeros(batch_size, 4, 3, 224, 224),
    }


def test_stage1_state_dict_key_and_shape_contract_is_unchanged() -> None:
    model = DECO(
        act_dim=4,
        chunk_size=3,
        num_attn_blocks=2,
        heads=4,
        dim=32,
        rope_axes_dim=(4, 4),
        num_cameras=2,
    )
    inventory = "\n".join(
        f"{name}:{tuple(tensor.shape)}" for name, tensor in model.state_dict().items()
    )

    assert len(model.state_dict()) == 282
    assert hashlib.sha256(inventory.encode()).hexdigest() == (
        "05410bd8efc85b3178059ff1cef0f3693527d84cc6b717cdb486700fbe2de10b"
    )


def test_stage2_encodes_four_images_once_then_normalizes_and_adds_sensor_ids() -> None:
    model, encoder = _stage2_model()
    with torch.no_grad():
        model.sensor_embeddings.weight.fill_(0.25)
    model.train()

    tokens = model.encode_tactile_images(torch.zeros(2, 4, 3, 224, 224))

    assert encoder.calls == [(8, 3, 224, 224)]
    assert tokens.shape == (2, 4, 512)
    assert model.sensor_embeddings.weight.shape == (4, 512)
    assert torch.allclose(tokens, torch.full_like(tokens, 1.25), atol=1e-5)
    assert encoder.training is False


def test_every_stage2_block_receives_exactly_four_tactile_kv_tokens() -> None:
    model, _ = _stage2_model()
    seen: list[tuple[int, ...]] = []
    hooks = [
        block.tactile_key.register_forward_pre_hook(
            lambda _module, args: seen.append(tuple(args[0].shape))
        )
        for block in model.mmattn
    ]
    try:
        prediction, _ = model(**_inputs(), training=True)
    finally:
        for hook in hooks:
            hook.remove()

    assert prediction.shape == (2, 3, 4)
    assert seen == [(2, 4, 512), (2, 4, 512)]


@pytest.mark.parametrize(
    ("tactile_images", "message"),
    [
        (None, "requires tactile_images"),
        (torch.zeros(2, 3, 3, 224, 224), "exactly 4 tactile sensors"),
        (torch.zeros(2, 4, 3, 128, 128), r"\[B, 4, 3, 224, 224\]"),
    ],
)
def test_stage2_rejects_missing_or_malformed_tactile_images(
    tactile_images: torch.Tensor | None,
    message: str,
) -> None:
    model, _ = _stage2_model()
    inputs = _inputs()
    inputs["tactile_images"] = tactile_images

    with pytest.raises(ValueError, match=message):
        model(**inputs, training=True)


def test_stage2_uses_zero_scalar_gates_and_rank_configured_pi_adapters() -> None:
    model, _ = _stage2_model(rank=7)

    for block in model.mmattn:
        assert block.tactile_gate.shape == torch.Size([])
        assert block.tactile_gate.item() == 0.0
        adapters = (
            block.img_qkv_pi,
            block.img_proj_pi,
            block.img_mlp_pi,
            block.act_qkv_pi,
            block.act_proj_pi,
            block.act_mlp_pi,
        )
        assert all(adapter.down.weight.shape[0] == 7 for adapter in adapters)
        assert block.img_qkv_pi.up.weight.shape == (96, 7)
        assert block.act_proj_pi.up.weight.shape == (32, 7)
        assert all(torch.count_nonzero(adapter.up.weight) == 0 for adapter in adapters)
        assert all(torch.count_nonzero(adapter.up.bias) == 0 for adapter in adapters)


def test_zero_gate_backpropagates_to_gates_then_nonzero_gate_reaches_tactile_kv() -> None:
    model, _ = _stage2_model()
    with torch.no_grad():
        model.linear.weight.fill_(0.1)
    model.train()

    torch.manual_seed(11)
    prediction, _ = model(**_inputs(), training=True)
    prediction.square().mean().backward()

    assert all(block.tactile_gate.grad is not None for block in model.mmattn)
    assert any(block.tactile_gate.grad.abs().item() > 0 for block in model.mmattn)
    assert all(
        block.tactile_key.weight.grad is None
        or torch.count_nonzero(block.tactile_key.weight.grad) == 0
        for block in model.mmattn
    )
    assert any(
        adapter.up.weight.grad is not None
        and torch.count_nonzero(adapter.up.weight.grad) > 0
        for block in model.mmattn
        for adapter in (block.img_qkv_pi, block.act_qkv_pi)
    )

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for block in model.mmattn:
            block.tactile_gate.fill_(0.5)
    torch.manual_seed(11)
    prediction, _ = model(**_inputs(), training=True)
    prediction.square().mean().backward()

    assert any(
        block.tactile_key.weight.grad is not None
        and torch.count_nonzero(block.tactile_key.weight.grad) > 0
        for block in model.mmattn
    )
    assert any(
        block.tactile_value.weight.grad is not None
        and torch.count_nonzero(block.tactile_value.weight.grad) > 0
        for block in model.mmattn
    )
