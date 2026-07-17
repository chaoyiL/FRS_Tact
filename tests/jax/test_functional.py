from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

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


def _numpy_linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight.T + bias


def _numpy_layer_norm(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias


def _numpy_rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return weight * (x * (1.0 / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)))


def _numpy_apply_rope(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    timescale = 10_000 ** ((2.0 / x.shape[-1]) * np.arange(half, dtype=np.float32))
    radians = positions[..., None].astype(np.float32) / timescale[None, None, :]
    sin = np.sin(radians)[..., None, :]
    cos = np.cos(radians)[..., None, :]
    x1, x2 = np.split(x, 2, axis=-1)
    return np.concatenate((x1 * cos - x2 * sin, x2 * cos + x1 * sin), axis=-1)


def _numpy_prefix_weights(schedule: str, start: int, end: int, total: int) -> np.ndarray:
    start = min(start, end)
    if schedule == "ZEROS":
        return (np.arange(total) < start).astype(np.float32)
    if schedule == "ONES":
        return (np.arange(total) < end).astype(np.float32)
    middle_length = max(end - start, 0)
    if middle_length:
        middle = np.linspace(1.0, 0.0, middle_length + 2, dtype=np.float32)[1:-1]
        if schedule == "EXP":
            middle = middle * np.expm1(middle) / (math.e - 1.0)
    else:
        middle = np.zeros((0,), dtype=np.float32)
    return np.concatenate(
        (
            np.ones((min(start, total),), dtype=np.float32),
            middle,
            np.zeros((max(total - end, 0),), dtype=np.float32),
        )
    )[:total]


def _numpy_aloha_gripper_to_angular(value: np.ndarray) -> np.ndarray:
    linear_position = value * (0.05800 - 0.01844) + 0.01844
    ratio = (0.022**2 + linear_position**2 - 0.036**2) / (2 * 0.022 * linear_position)
    radians = np.arcsin(np.clip(ratio, -1.0, 1.0))
    return (radians - 0.4) / (1.5 - 0.4)


def _numpy_aloha_gripper_from_angular(value: np.ndarray) -> np.ndarray:
    radians = value * (1.5 - 0.4) + 0.4
    return (radians - (-0.6213)) / (1.4910 - (-0.6213))


def _numpy_aloha_gripper_from_angular_inv(value: np.ndarray) -> np.ndarray:
    radians = value * (1.4910 - (-0.6213)) + (-0.6213)
    return (radians - 0.4) / (1.5 - 0.4)


def test_linear_matches_numpy() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 3, 5)).astype(np.float32)
    weight = rng.normal(size=(11, 5)).astype(np.float32)
    bias = rng.normal(size=(11,)).astype(np.float32)
    expected = _numpy_linear(x, weight, bias)
    actual = np.asarray(linear(jnp.asarray(x), jnp.asarray(weight), jnp.asarray(bias)))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_normalization_matches_numpy() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(2, 4, 8)).astype(np.float32)
    weight = rng.normal(size=(8,)).astype(np.float32)
    bias = rng.normal(size=(8,)).astype(np.float32)
    expected_layer = _numpy_layer_norm(x, weight, bias, 1e-6)
    actual_layer = np.asarray(layer_norm(jnp.asarray(x), jnp.asarray(weight), jnp.asarray(bias), 1e-6))
    np.testing.assert_allclose(actual_layer, expected_layer, rtol=2e-6, atol=2e-6)

    expected_rms = _numpy_rms_norm(x, weight, 1e-5)
    actual_rms = np.asarray(rms_norm(jnp.asarray(x), jnp.asarray(weight), 1e-5))
    np.testing.assert_allclose(actual_rms, expected_rms, rtol=2e-6, atol=2e-6)


def test_smolvla_rope_matches_numpy_formula() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 7, 5, 64)).astype(np.float32)
    positions = np.stack((np.arange(7), np.arange(7) + 2)).astype(np.int32)
    expected = _numpy_apply_rope(x, positions)
    actual = np.asarray(apply_rope(jnp.asarray(x), jnp.asarray(positions)))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_attention_and_masks_match_reference() -> None:
    rng = np.random.default_rng(5)
    query = rng.normal(size=(2, 4, 6, 8)).astype(np.float32)
    key = rng.normal(size=(2, 5, 2, 8)).astype(np.float32)
    value = rng.normal(size=(2, 5, 2, 8)).astype(np.float32)
    mask = np.ones((2, 4, 5), dtype=np.bool_)
    attn = np.asarray(
        eager_attention(jnp.asarray(query), jnp.asarray(key), jnp.asarray(value), jnp.asarray(mask))
    )
    assert attn.shape == (2, 4, 48)

    pad = np.ones((2, 5), dtype=np.bool_)
    att_masks = np.zeros((2, 5), dtype=np.bool_)
    att2d = np.asarray(make_att_2d_masks(jnp.asarray(pad), jnp.asarray(att_masks)))
    assert att2d.shape == (2, 5, 5)


def test_sinusoidal_time_embedding_shape() -> None:
    times = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float32)
    emb = np.asarray(sinusoidal_time_embedding(times, 16, min_period=4e-3, max_period=4.0))
    assert emb.shape == (3, 16)


def test_resize_with_pad_matches_numpy_bilinear() -> None:
    rng = np.random.default_rng(17)
    image = rng.normal(size=(2, 3, 11, 17)).astype(np.float32)
    # JAX resize_with_pad uses ratio = max(w/W, h/H) then left/top pad.
    ratio = max(17 / 32, 11 / 32)
    resized_h = int(11 / ratio)
    resized_w = int(17 / ratio)
    actual = np.asarray(resize_with_pad(jnp.asarray(image), width=32, height=32))
    assert actual.shape == (2, 3, 32, 32)
    # Non-padded region should be non-trivial; top pad of height (32 - resized_h) is zeros.
    pad_h = 32 - resized_h
    assert pad_h >= 0
    if pad_h:
        np.testing.assert_allclose(actual[:, :, :pad_h, :], 0.0, atol=1e-6)
    assert resized_w <= 32


def test_aloha_adaptation_matches_numpy_formulas() -> None:
    state = np.linspace(0.1, 0.9, 28, dtype=np.float32).reshape(2, 14)
    expected_state = state.copy()
    expected_state[:, [1, 2, 8, 9]] *= -1
    for index in (6, 13):
        expected_state[:, index] = _numpy_aloha_gripper_to_angular(expected_state[:, index])
    actual_state = np.asarray(aloha_decode_state(jnp.asarray(state)))
    np.testing.assert_allclose(actual_state, expected_state, rtol=2e-6, atol=2e-6)

    actions = np.linspace(0.05, 0.95, 56, dtype=np.float32).reshape(2, 2, 14)
    expected = actions.copy()
    expected[:, :, [1, 2, 8, 9]] *= -1
    inverse = actions.copy()
    inverse[:, :, [1, 2, 8, 9]] *= -1
    for index in (6, 13):
        expected[:, :, index] = _numpy_aloha_gripper_from_angular(expected[:, :, index])
        inverse[:, :, index] = _numpy_aloha_gripper_from_angular_inv(inverse[:, :, index])
    np.testing.assert_allclose(
        np.asarray(aloha_encode_actions(jnp.asarray(actions))),
        expected,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(aloha_encode_actions_inverse(jnp.asarray(actions))),
        inverse,
        rtol=2e-6,
        atol=2e-6,
    )


@pytest.mark.parametrize("schedule", ["ZEROS", "ONES", "LINEAR", "EXP"])
def test_rtc_prefix_weights_match_numpy(schedule: str) -> None:
    expected = _numpy_prefix_weights(schedule, 2, 7, 10)
    actual = np.asarray(prefix_weights(schedule, 2, 7, 10))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_rtc_guidance_is_deterministic() -> None:
    x = np.linspace(-1, 1, 24, dtype=np.float32).reshape(1, 4, 6)
    previous = np.zeros_like(x)
    actual = rtc_guided_velocity(
        lambda value: value * 0.25,
        jnp.asarray(x),
        jnp.asarray(previous),
        time=jnp.asarray(0.5),
        inference_delay=1,
        execution_horizon=3,
        config=JaxRTCConfig(execution_horizon=3, max_guidance_weight=10.0),
    )
    again = rtc_guided_velocity(
        lambda value: value * 0.25,
        jnp.asarray(x),
        jnp.asarray(previous),
        time=jnp.asarray(0.5),
        inference_delay=1,
        execution_horizon=3,
        config=JaxRTCConfig(execution_horizon=3, max_guidance_weight=10.0),
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(again), rtol=0, atol=0)
    assert np.asarray(actual).shape == x.shape


def test_rtc_broadcasts_a_single_previous_chunk() -> None:
    x = np.linspace(-1, 1, 48, dtype=np.float32).reshape(2, 4, 6)
    previous = np.ones((1, 2, 3), dtype=np.float32)
    actual = rtc_guided_velocity(
        lambda value: value * 0.25,
        jnp.asarray(x),
        jnp.asarray(previous),
        time=jnp.asarray(0.5),
        inference_delay=1,
        execution_horizon=2,
        config=JaxRTCConfig(execution_horizon=2),
    )
    assert np.asarray(actual).shape == x.shape
