from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")

from lerobot.policies.smolvla_jax.functional import (  # noqa: E402
    apply_rope,
    eager_attention,
    layer_norm,
    linear,
    make_att_2d_masks,
    rms_norm,
    sinusoidal_time_embedding,
)
from lerobot.policies.smolvla_jax.preprocessing import (  # noqa: E402
    aloha_decode_state,
    aloha_encode_actions,
    aloha_encode_actions_inverse,
    resize_with_pad,
)
from lerobot.policies.smolvla_jax.rtc import JaxRTCConfig, prefix_weights, rtc_guided_velocity  # noqa: E402


def test_linear_matches_torch() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 3, 5)).astype(np.float32)
    weight = rng.normal(size=(11, 5)).astype(np.float32)
    bias = rng.normal(size=(11,)).astype(np.float32)
    expected = torch.nn.functional.linear(
        torch.from_numpy(x), torch.from_numpy(weight), torch.from_numpy(bias)
    ).numpy()
    actual = np.asarray(linear(jnp.asarray(x), jnp.asarray(weight), jnp.asarray(bias)))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_normalization_matches_torch() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(2, 4, 8)).astype(np.float32)
    weight = rng.normal(size=(8,)).astype(np.float32)
    bias = rng.normal(size=(8,)).astype(np.float32)
    expected_layer = torch.nn.functional.layer_norm(
        torch.from_numpy(x), (8,), torch.from_numpy(weight), torch.from_numpy(bias), 1e-6
    ).numpy()
    actual_layer = np.asarray(layer_norm(jnp.asarray(x), jnp.asarray(weight), jnp.asarray(bias), 1e-6))
    np.testing.assert_allclose(actual_layer, expected_layer, rtol=2e-6, atol=2e-6)

    x_torch = torch.from_numpy(x)
    expected_rms = (
        weight * (x_torch * torch.rsqrt(x_torch.square().mean(dim=-1, keepdim=True) + 1e-5)).numpy()
    )
    actual_rms = np.asarray(rms_norm(jnp.asarray(x), jnp.asarray(weight), 1e-5))
    np.testing.assert_allclose(actual_rms, expected_rms, rtol=2e-6, atol=2e-6)


def test_smolvla_rope_matches_torch_formula() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 7, 5, 64)).astype(np.float32)
    positions = np.stack((np.arange(7), np.arange(7) + 2)).astype(np.int32)
    x_torch = torch.from_numpy(x)
    pos_torch = torch.from_numpy(positions)
    half = x.shape[-1] // 2
    timescale = 10_000 ** ((2.0 / x.shape[-1]) * torch.arange(half, dtype=torch.float32))
    radians = pos_torch[..., None].float() / timescale[None, None, :]
    sin, cos = radians.sin()[..., None, :], radians.cos()[..., None, :]
    x1, x2 = x_torch.split(half, dim=-1)
    expected = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).numpy()
    actual = np.asarray(apply_rope(jnp.asarray(x), jnp.asarray(positions)))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_attention_and_masks_match_reference() -> None:
    rng = np.random.default_rng(5)
    query = rng.normal(size=(2, 4, 6, 8)).astype(np.float32)
    key = rng.normal(size=(2, 5, 2, 8)).astype(np.float32)
    value = rng.normal(size=(2, 5, 2, 8)).astype(np.float32)
    mask = np.ones((2, 4, 5), dtype=np.bool_)
    mask[:, :, -1] = False

    q_t = torch.from_numpy(query).transpose(1, 2)
    k_t = torch.from_numpy(key)
    v_t = torch.from_numpy(value)
    k_t = k_t[:, :, :, None, :].expand(2, 5, 2, 3, 8).reshape(2, 5, 6, 8).transpose(1, 2)
    v_t = v_t[:, :, :, None, :].expand(2, 5, 2, 3, 8).reshape(2, 5, 6, 8).transpose(1, 2)
    scores = q_t @ k_t.transpose(2, 3) * (8**-0.5)
    scores = torch.where(torch.from_numpy(mask)[:, None], scores, torch.finfo(torch.float32).min)
    expected = (scores.softmax(dim=-1) @ v_t).transpose(1, 2).reshape(2, 4, -1).numpy()
    actual = np.asarray(
        eager_attention(jnp.asarray(query), jnp.asarray(key), jnp.asarray(value), jnp.asarray(mask))
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    pad = jnp.asarray([[True, True, False, True], [True, True, True, True]])
    attention_ar = jnp.asarray([[False, False, False, True], [False, False, True, True]])
    mask_2d = np.asarray(make_att_2d_masks(pad, attention_ar))
    assert mask_2d.shape == (2, 4, 4)
    assert not mask_2d[0, 3, 2]


def test_time_embedding_shape_and_endpoints() -> None:
    result = sinusoidal_time_embedding(jnp.asarray([0.0, 1.0]), 16, 4e-3, 4.0)
    assert result.shape == (2, 16)
    np.testing.assert_allclose(np.asarray(result[0, :8]), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(result[0, 8:]), 1.0, atol=1e-7)


def test_resize_with_pad_matches_torch() -> None:
    rng = np.random.default_rng(17)
    image = rng.normal(size=(2, 3, 11, 17)).astype(np.float32)
    resized = torch.nn.functional.interpolate(
        torch.from_numpy(image), size=(20, 32), mode="bilinear", align_corners=False
    )
    expected = torch.nn.functional.pad(resized, (0, 0, 12, 0), value=0).numpy()
    actual = np.asarray(resize_with_pad(jnp.asarray(image), width=32, height=32))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_aloha_adaptation_matches_reference_formulas() -> None:
    from lerobot.policies.smolvla.modeling_smolvla import (
        aloha_gripper_from_angular,
        aloha_gripper_from_angular_inv,
        aloha_gripper_to_angular,
    )

    state = torch.linspace(0.1, 0.9, 28).reshape(2, 14)
    expected_state = state.clone()
    expected_state[:, [1, 2, 8, 9]] *= -1
    for index in (6, 13):
        expected_state[:, index] = aloha_gripper_to_angular(expected_state[:, index])
    actual_state = np.asarray(aloha_decode_state(jnp.asarray(state.numpy())))
    np.testing.assert_allclose(actual_state, expected_state.numpy(), rtol=2e-6, atol=2e-6)

    actions = torch.linspace(0.05, 0.95, 56).reshape(2, 2, 14)
    expected = actions.clone()
    expected[:, :, [1, 2, 8, 9]] *= -1
    inverse = actions.clone()
    inverse[:, :, [1, 2, 8, 9]] *= -1
    for index in (6, 13):
        expected[:, :, index] = aloha_gripper_from_angular(expected[:, :, index])
        inverse[:, :, index] = aloha_gripper_from_angular_inv(inverse[:, :, index])
    np.testing.assert_allclose(
        np.asarray(aloha_encode_actions(jnp.asarray(actions.numpy()))),
        expected.numpy(),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(aloha_encode_actions_inverse(jnp.asarray(actions.numpy()))),
        inverse.numpy(),
        rtol=2e-6,
        atol=2e-6,
    )


@pytest.mark.parametrize("schedule", ["ZEROS", "ONES", "LINEAR", "EXP"])
def test_rtc_prefix_weights_match_torch(schedule: str) -> None:
    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.rtc import RTCConfig, RTCProcessor

    torch_processor = RTCProcessor(RTCConfig(prefix_attention_schedule=RTCAttentionSchedule(schedule)))
    expected = torch_processor.get_prefix_weights(2, 7, 10).numpy()
    actual = np.asarray(prefix_weights(schedule, 2, 7, 10))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_rtc_guidance_matches_current_reference() -> None:
    from lerobot.policies.rtc import RTCConfig, RTCProcessor

    x = torch.linspace(-1, 1, 24).reshape(1, 4, 6)
    previous = torch.zeros_like(x)
    processor = RTCProcessor(RTCConfig(execution_horizon=3, max_guidance_weight=10.0))
    expected = processor.denoise_step(
        x,
        previous,
        inference_delay=1,
        time=0.5,
        original_denoise_step_partial=lambda value: value * 0.25,
        execution_horizon=3,
    )
    actual = rtc_guided_velocity(
        lambda value: value * 0.25,
        jnp.asarray(x.numpy()),
        jnp.asarray(previous.numpy()),
        time=jnp.asarray(0.5),
        inference_delay=1,
        execution_horizon=3,
        config=JaxRTCConfig(execution_horizon=3, max_guidance_weight=10.0),
    )
    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), rtol=2e-6, atol=2e-6)


def test_rtc_broadcasts_a_single_previous_chunk_like_torch() -> None:
    from lerobot.policies.rtc import RTCConfig, RTCProcessor

    x = torch.linspace(-1, 1, 48).reshape(2, 4, 6)
    previous = torch.ones(1, 2, 3)
    processor = RTCProcessor(RTCConfig(execution_horizon=2))
    expected = processor.denoise_step(
        x,
        previous,
        inference_delay=1,
        time=0.5,
        original_denoise_step_partial=lambda value: value * 0.25,
        execution_horizon=2,
    )
    actual = rtc_guided_velocity(
        lambda value: value * 0.25,
        jnp.asarray(x.numpy()),
        jnp.asarray(previous.numpy()),
        time=jnp.asarray(0.5),
        inference_delay=1,
        execution_horizon=2,
        config=JaxRTCConfig(execution_horizon=2),
    )
    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), rtol=2e-6, atol=2e-6)
