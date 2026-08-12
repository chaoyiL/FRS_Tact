from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from typing import Literal
from typing import Protocol

import jax
import jax.numpy as jnp
from flax import struct

from train_smolvla.modeling import PrefixContext
from utils.integration import euler_integrate_velocity
from utils.integration import fireflow_integrate_velocity
from utils.integration import slerpflow_integrate_velocity

ReverseSolver = Literal["euler", "fireflow", "slerpflow"]


class ReverseFlowConfig(Protocol):
    max_action_dim: int


class ReverseFunctionalModel(Protocol):
    def build_prefix_context(
        self,
        params: Any,
        images: jax.Array,
        image_masks: jax.Array,
        language_tokens: jax.Array,
        language_masks: jax.Array,
        state: jax.Array,
    ) -> PrefixContext:
        ...

    def denoise_step(
        self,
        params: Any,
        context: PrefixContext,
        x_t: jax.Array,
        timestep: jax.Array,
    ) -> jax.Array:
        ...


class PreparedReverseModel(Protocol):
    config: ReverseFlowConfig
    params: Any
    model: ReverseFunctionalModel


@struct.dataclass
class VelocityContext:
    pad_mask: jax.Array
    cache: tuple[tuple[jax.Array, jax.Array], ...]


_PREFIX_CACHE: dict[int, Any] = {}
_REVERSE_CACHE: dict[tuple[int, int, str], Any] = {}


def _jitted_prefix_builder(model: PreparedReverseModel):
    cache_key = id(model)
    run = _PREFIX_CACHE.get(cache_key)
    if run is not None:
        return run

    functional_model = model.model

    @jax.jit
    def run(
        params,
        images: jax.Array,
        image_masks: jax.Array,
        language_tokens: jax.Array,
        language_masks: jax.Array,
        state: jax.Array,
    ) -> VelocityContext:
        prefix = functional_model.build_prefix_context(
            params,
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
        )
        return VelocityContext(pad_mask=prefix.pad_mask, cache=prefix.cache)

    _PREFIX_CACHE[cache_key] = run
    return run


def build_velocity_context_from_prepared(
    model: PreparedReverseModel,
    batch: Mapping[str, Any],
) -> VelocityContext:
    """JIT-compiled prefix encode from a preprocessor output mapping."""
    return _jitted_prefix_builder(model)(
        model.params,
        batch["images"],
        batch["image_masks"],
        batch["language_tokens"],
        batch["language_masks"],
        batch["state"],
    )


def _jitted_reverse_from_context(
    model: PreparedReverseModel,
    *,
    num_steps: int,
    solver: ReverseSolver,
):
    cache_key = (id(model), num_steps, solver)
    run = _REVERSE_CACHE.get(cache_key)
    if run is not None:
        return run

    if solver == "euler":
        integrate = euler_integrate_velocity
    elif solver == "fireflow":
        integrate = fireflow_integrate_velocity
    else:
        integrate = slerpflow_integrate_velocity
    functional_model = model.model
    max_action_dim = int(model.config.max_action_dim)

    @jax.jit
    def run(params, context: VelocityContext, actions: jax.Array) -> jax.Array:
        actions = jnp.asarray(actions, dtype=jnp.float32)

        def velocity_fn(x: jax.Array, t: jax.Array) -> jax.Array:
            x_in = x
            pad = max_action_dim - x.shape[-1]
            if pad > 0:
                x_in = jnp.pad(x, ((0, 0), (0, 0), (0, pad)))
            t = jnp.asarray(t, dtype=jnp.float32)
            if t.ndim == 0:
                t = jnp.full((x.shape[0],), t)
            velocity = functional_model.denoise_step(
                params,
                PrefixContext(pad_mask=context.pad_mask, cache=context.cache),
                x_in,
                t,
            )
            return velocity[..., : x.shape[-1]].astype(jnp.float32)

        return integrate(velocity_fn, actions, num_steps=num_steps)

    _REVERSE_CACHE[cache_key] = run
    return run


def reverse_integrate_prepared_actions(
    model: PreparedReverseModel,
    batch: Mapping[str, Any],
    normalized_actions: jax.Array,
    *,
    num_steps: int,
    solver: ReverseSolver = "slerpflow",
) -> jax.Array:
    """Reverse normalized actions using an already prepared observation batch."""
    if solver not in ("euler", "fireflow", "slerpflow"):
        raise ValueError(
            f"solver must be 'euler', 'fireflow', or 'slerpflow', got {solver!r}."
        )
    context = build_velocity_context_from_prepared(model, batch)
    return _jitted_reverse_from_context(model, num_steps=num_steps, solver=solver)(
        model.params, context, jnp.asarray(normalized_actions, dtype=jnp.float32)
    )
