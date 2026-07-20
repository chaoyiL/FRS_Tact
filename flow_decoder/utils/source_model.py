from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from openpi.src.openpi.models import model as _model
from flow_decoder.utils.integration import euler_integrate_velocity
from flow_decoder.utils.integration import fireflow_integrate_velocity


def make_attn_mask(input_mask: jax.Array, mask_ar: jax.Array) -> jax.Array:
    """Build the same prefix/action block mask used by openpi.models.pi0."""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumulative = jnp.cumsum(mask_ar, axis=1)
    attention_mask = cumulative[:, None, :] <= cumulative[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attention_mask, valid_mask)


@dataclasses.dataclass(frozen=True)
class VelocityContext:
    observation: _model.Observation
    prefix_tokens: jax.Array
    prefix_mask: jax.Array
    kv_cache: Any

    def tree_flatten(self):
        return ((self.observation, self.prefix_tokens, self.prefix_mask, self.kv_cache), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


jax.tree_util.register_pytree_node_class(VelocityContext)


def stack_observations(observations: Sequence[_model.Observation]) -> _model.Observation:
    if not observations:
        raise ValueError("Cannot stack an empty observation batch.")
    return jax.tree.map(
        lambda *xs: None if xs[0] is None else jnp.stack([jnp.asarray(x) for x in xs], axis=0),
        *observations,
    )


def create_velocity_context(model: _model.BaseModel, observation: _model.Observation) -> VelocityContext:
    image_keys = model.image_keys if model.image_keys is not None else list(observation.images.keys())
    observation = _model.preprocess_observation(None, observation, train=False, image_keys=image_keys)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    return VelocityContext(observation, prefix_tokens, prefix_mask, kv_cache)


def predict_velocity_with_context(
    model: _model.BaseModel, context: VelocityContext, x: jax.Array, t: jax.Array
) -> jax.Array:
    batch_size = context.observation.state.shape[0]
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(context.observation, x, t)
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = jnp.broadcast_to(
        context.prefix_mask[:, None, :],
        (batch_size, suffix_tokens.shape[1], context.prefix_tokens.shape[1]),
    )
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = jnp.sum(context.prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=context.kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    del prefix_out
    return model.action_out_proj(suffix_out[:, -model.action_horizon :]).astype(jnp.float32)


_REVERSE_CACHE: dict[tuple[int, int, str], Any] = {}


def reverse_integrate_actions(
    model: _model.BaseModel,
    observation: _model.Observation,
    actions: jax.Array,
    *,
    num_steps: int,
    solver: Literal["euler", "fireflow"] = "euler",
) -> jax.Array:
    if solver not in ("euler", "fireflow"):
        raise ValueError(f"solver must be 'euler' or 'fireflow', got {solver!r}.")
    context = create_velocity_context(model, observation)
    cache_key = (id(model), num_steps, solver)
    run = _REVERSE_CACHE.get(cache_key)
    if run is None:
        integrate = euler_integrate_velocity if solver == "euler" else fireflow_integrate_velocity

        @jax.jit
        def run(context_arg: VelocityContext, actions_arg: jax.Array) -> jax.Array:
            return integrate(
                lambda x, t: predict_velocity_with_context(model, context_arg, x, t),
                actions_arg,
                num_steps=num_steps,
            )

        _REVERSE_CACHE[cache_key] = run
    return run(context, jnp.asarray(actions, dtype=jnp.float32))


def deterministic_noise(indices: Sequence[int], shape: tuple[int, int], *, seed: int) -> jax.Array:
    base_key = jax.random.key(seed)
    noises = [
        jax.random.normal(jax.random.fold_in(base_key, int(index)), shape, dtype=jnp.float32)
        for index in indices
    ]
    return jnp.stack(noises, axis=0)


def inversion_mse(x_base: jax.Array, initial_noise: jax.Array) -> np.ndarray:
    axes = tuple(range(1, x_base.ndim))
    return np.asarray(jax.device_get(jnp.mean(jnp.square(x_base - initial_noise), axis=axes)), dtype=np.float32)
