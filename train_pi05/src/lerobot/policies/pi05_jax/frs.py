"""FRS-specific extensions to `Pi0`. NOT part of openpi.

This is the one piece of genuinely new model-level logic on this branch; everything else under
`pi05_jax/` is a verbatim copy of openpi (see README.md). It lives in its own module, as free
functions rather than `Pi0` methods, precisely so that `pi0.py` can stay byte-for-byte upstream --
diffing it against openpi should show only the provenance header and rewritten imports.

Why FRS needs this: `Pi0.sample_actions` only exposes a fixed t:1->0 forward loop, with the
per-step velocity computation inlined in a `step()` closure. FRS integrates the same flow-matching
velocity field *backwards* (t:0->1; see `utils/integration.py`'s euler/fireflow solvers, driven
from `utils/pi05_source_model.py`), which means evaluating v(x, t) at arbitrary (x, t) pairs.
`denoise_step` below is that closure body copied out verbatim, with `observation`/`prefix_mask`/
`kv_cache` coming from an explicit cache instead of the enclosing scope; `build_prefix_cache` is
the KV-cache-filling half of `sample_actions` that precedes it.

`sample_actions` itself is untouched, so ordinary forward sampling still runs the exact upstream
code path -- a bug here cannot affect it. The corresponding equivalence check (drive `denoise_step`
manually t:1->0 and compare against `sample_actions` on the same input) is item 1 of README.md's
verification list.
"""

from __future__ import annotations

import einops
import flax.struct as struct
import jax.numpy as jnp

from . import array_typing as at
from . import gemma as _gemma
from . import model as _model
from .pi0 import Pi0, make_attn_mask


@struct.dataclass
class Pi0PrefixCache:
    """Pytree bundle returned by `build_prefix_cache` and consumed by `denoise_step`."""

    observation: _model.Observation
    prefix_mask: at.Bool[at.Array, "b s"]
    kv_cache: _gemma.KVCache


@at.typecheck
def build_prefix_cache(model: Pi0, observation: _model.Observation) -> Pi0PrefixCache:
    """Prefix half of `Pi0.sample_actions`, factored out for reuse.

    Encodes images/language once into a KV cache. The returned `Pi0PrefixCache` can then be fed to
    `denoise_step` at arbitrary `(x_t, t)` pairs -- both for ordinary forward sampling and for
    FRS's reverse ODE integration, which needs many velocity evaluations at arbitrary intermediate
    times rather than a fixed t:1->0 sweep.
    """
    observation = _model.preprocess_observation(
        None, observation, train=False, image_keys=model.image_keys
    )
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    return Pi0PrefixCache(observation=observation, prefix_mask=prefix_mask, kv_cache=kv_cache)


@at.typecheck
def denoise_step(
    model: Pi0,
    cache: Pi0PrefixCache,
    x_t: _model.Actions,
    timestep: at.Float[at.Array, " b"],
) -> _model.Actions:
    """Single flow-matching velocity evaluation v(x_t, t) against a prebuilt prefix cache.

    Exactly the body of the `step()` closure inside `Pi0.sample_actions`, with no step
    direction/count baked in -- callers drive `(x_t, timestep)` themselves.
    """
    batch_size = cache.observation.state.shape[0]
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        cache.observation, x_t, jnp.broadcast_to(timestep, batch_size)
    )
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(cache.prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = jnp.sum(cache.prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=cache.kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    assert prefix_out is None
    return model.action_out_proj(suffix_out[:, -model.action_horizon :])
