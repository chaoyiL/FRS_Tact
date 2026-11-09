"""Compare the native Torch port against the actual TacFRS JAX Flow network."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")
from train_pi05_frs.utils.model import DecoderConfig, TactileConditionedFlowDecoder, decode_fireflow
from train_baseline_act_inter.config import RIGHT_TACTILE_KEYS, TACTILE_KEYS
from train_baseline_act_inter.model import DirectDecoderConfig, DirectTactileActionDecoder, flow_matching_loss


def _copy_linear(destination, source):
    destination.weight.copy_(
        torch.tensor(np.asarray(source.kernel.value).reshape(source.kernel.value.shape[0], -1).T.copy())
    )
    if destination.bias is not None:
        destination.bias.copy_(torch.tensor(np.asarray(source.bias.value).reshape(-1).copy()))


def _copy_norm(destination, source):
    destination.weight.copy_(torch.tensor(np.asarray(source.scale.value).copy()))
    destination.bias.copy_(torch.tensor(np.asarray(source.bias.value).copy()))


def _copy_attention(destination, source):
    weights, biases = [], []
    for name in ("query", "key", "value"):
        layer = getattr(source, name)
        weights.append(torch.tensor(np.asarray(layer.kernel.value).reshape(256, 256).T.copy()))
        biases.append(torch.tensor(np.asarray(layer.bias.value).reshape(-1).copy()))
    destination.in_proj_weight.copy_(torch.cat(weights))
    destination.in_proj_bias.copy_(torch.cat(biases))
    destination.out_proj.weight.copy_(
        torch.tensor(np.asarray(source.out.kernel.value).reshape(256, 256).T.copy())
    )
    destination.out_proj.bias.copy_(torch.tensor(np.asarray(source.out.bias.value).copy()))


@pytest.mark.parametrize("width,sensors,state_dim", [(10, 2, 7), (20, 4, 20)])
def test_velocity_and_fireflow_match_tacfrs_with_identical_weights(width, sensors, state_dim):
    reference = TactileConditionedFlowDecoder(
        DecoderConfig(
            action_dim=width,
            action_horizon=50,
            tactile_window=10,
            model_dim=256,
            depth=6,
            num_heads=4,
            mlp_ratio=4,
            num_tactile_tokens=sensors,
            state_conditioning=True,
            state_dim=state_dim,
        ),
        rngs=nnx.Rngs(3),
    )
    model = DirectTactileActionDecoder(
        DirectDecoderConfig(
            action_dim=width, tactile_keys=RIGHT_TACTILE_KEYS if sensors == 2 else TACTILE_KEYS
        )
    ).eval()
    with torch.no_grad():
        for name in ("action_in", "tactile_proj", "state_fc1", "state_fc2", "action_out"):
            _copy_linear(getattr(model, name), getattr(reference, name))
        for name in ("fc1", "fc2"):
            _copy_linear(getattr(model.time_mlp, name), getattr(reference.time_mlp, name))
        for name in ("dense_i", "dense_h"):
            _copy_linear(getattr(model.tactile_gru, name), getattr(reference.tactile_gru.cell, name))
        for name in ("state_norm", "out_norm"):
            _copy_norm(getattr(model, name), getattr(reference, name))
        for block, ref_block in zip(model.blocks, reference.blocks, strict=True):
            for name in ("norm_self", "norm_cross_q", "norm_cross_kv", "norm_mlp"):
                _copy_norm(getattr(block, name), getattr(ref_block, name))
            for name in ("fc1", "fc2"):
                _copy_linear(getattr(block, name), getattr(ref_block, name))
            for name in ("self_attention", "cross_attention"):
                _copy_attention(getattr(block, name), getattr(ref_block, name))
    rng = np.random.default_rng(18)
    action = rng.normal(size=(1, 50, width)).astype(np.float32)
    tactile = rng.normal(size=(1, 10, sensors, 512)).astype(np.float32)
    state = rng.normal(size=(1, state_dim)).astype(np.float32)
    time = np.array([0.37], np.float32)
    with torch.no_grad():
        actual = model(
            torch.from_numpy(action),
            torch.from_numpy(time),
            torch.from_numpy(tactile),
            state=torch.from_numpy(state),
        ).numpy()
        decoded = model.decode(
            torch.from_numpy(action),
            torch.from_numpy(tactile),
            state=torch.from_numpy(state),
            preserve_grippers=False,
        ).numpy()
    expected = np.asarray(reference(action, time, tactile, state=state))
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-5)
    integrated = np.asarray(decode_fireflow(reference, action, tactile, num_steps=10, state=state))
    np.testing.assert_allclose(decoded, integrated, rtol=4e-4, atol=5e-5)

    # Compare the complete TacFRS Flow training objective using the same VLA origin.
    from train_pi05_frs.utils.model import (
        gated_loss_components_per_sample,
        bimanual_loss_components_per_sample,
    )

    target = rng.normal(size=action.shape).astype(np.float32)
    initial = rng.normal(size=(1, sensors, 512)).astype(np.float32)
    torch.manual_seed(8)
    sampled_t = torch.rand(1).numpy()
    keep = (torch.rand(1) >= 0.1).numpy().astype(np.float32)
    torch.manual_seed(8)
    with torch.no_grad():
        actual_loss = flow_matching_loss(
            model,
            torch.from_numpy(action),
            torch.from_numpy(target),
            torch.from_numpy(tactile),
            torch.from_numpy(state),
            torch.ones((1, 50), dtype=torch.bool),
            baseline_tactile=torch.from_numpy(initial),
        ).item()
    normalize = lambda value: value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8)
    per_sensor = 1 - (normalize(tactile[:, -1]) * normalize(initial)).sum(-1)
    change = per_sensor.mean(-1) if width == 10 else per_sensor.reshape(1, 2, 2).mean(-1)
    gates = 1 / (1 + np.exp(-(change - (0.4 if width == 10 else 0.65)) / (0.1 if width == 10 else 0.15)))
    objective = gated_loss_components_per_sample if width == 10 else bimanual_loss_components_per_sample
    kwargs = {"use_composite_endpoint": True} if width == 10 else {}
    components = objective(
        reference,
        action,
        target,
        action,
        sampled_t,
        tactile,
        gates,
        state=state,
        state_keep_mask=keep,
        gate_lambda=1.0,
        aux_decode_weight=4.0,
        aux_decode_steps=10,
        aux_decode_solver="fireflow",
        low_gate_safety_weight=0.5,
        low_gate_safety_margin=0.03,
        rank_weight=4.0,
        repair_weight=2.0,
        **kwargs,
    )
    expected_loss = float(sum(np.asarray(value).mean() for value in components.values()))
    np.testing.assert_allclose(actual_loss, expected_loss, rtol=4e-4, atol=2e-4)
