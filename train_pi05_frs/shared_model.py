"""Masked joint-wrist flow matching for mixed single/dual-arm tasks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from train_pi05_frs.utils.integration import euler_integrate_velocity, fireflow_integrate_velocity
from train_pi05_frs.utils.model import bimanual_loss_components_per_sample


def masked_mse(left, right, action_mask):
    mask = action_mask[:, None, :]
    return jnp.sum(jnp.square(left - right) * mask, axis=(1, 2)) / jnp.maximum(
        left.shape[1] * jnp.sum(action_mask, axis=1), 1
    )


def shared_condition(model, batch):
    if model.config.state_conditioning:
        raise ValueError(
            "Shared tactile FRS uses no state token; native VLA checkpoints still receive their own state"
        )
    tactile = jnp.where(batch["token_mask"][:, None, :, None].astype(bool), batch["tactile"], 0)
    return model.encode_condition(tactile)


def shared_velocity(model, x, t, condition, batch):
    mask = batch["action_mask"][:, None, :]
    return model.velocity_from_condition(x * mask, t, condition, batch["token_mask"]) * mask


def decode_shared(model, batch, *, num_steps: int, solver: str = "fireflow"):
    """Mask at every ODE step; idle actions cannot drift or affect active inputs."""
    condition = shared_condition(model, batch)
    if solver not in ("euler", "fireflow"):
        raise ValueError(f"Unsupported shared decoder solver: {solver}")
    integrate = fireflow_integrate_velocity if solver == "fireflow" else euler_integrate_velocity
    return integrate(
        lambda x, t: shared_velocity(model, x, t, condition, batch),
        batch["x_base"] * batch["action_mask"][:, None, :],
        num_steps=num_steps,
    )


class _MaskedModelView(nnx.Module):
    """Reuse the established bimanual FM and safety/rank/repair objective unchanged."""

    def __init__(self, model, batch):
        self.model, self.config = model, model.config
        self.action_mask = nnx.Variable(batch["action_mask"])
        self.token_mask = nnx.Variable(batch["token_mask"])
        self.tactile = nnx.Variable(batch["tactile"])

    @property
    def batch(self):
        return {
            "action_mask": self.action_mask[...],
            "token_mask": self.token_mask[...],
            "tactile": self.tactile[...],
        }

    def encode_condition(self, tactile_seq, state=None, state_keep_mask=None):
        return shared_condition(self.model, self.batch)

    def velocity_from_condition(self, x, t, condition):
        return shared_velocity(self.model, x, t, condition, self.batch)

    def __call__(self, x, t, tactile_seq, **kwargs):
        return self.velocity_from_condition(x, t, self.encode_condition(tactile_seq))


def shared_flow_loss(
    model,
    batch,
    t,
    *,
    loss_mode: str,
    low_threshold: float,
    high_threshold: float,
    aux_decode_weight: float = 0,
    aux_decode_steps: int = 10,
    aux_decode_solver: str = "fireflow",
    low_gate_safety_weight: float = 0,
    low_gate_safety_margin: float = 0.03,
    rank_weight: float = 0,
    rank_margin: float = 0,
    repair_weight: float = 0,
    repair_margin: float = 0,
):
    mask = batch["action_mask"][:, None, :].astype(bool)
    batch = {**batch, **{key: jnp.where(mask, batch[key], 0) for key in ("x_base", "gt", "predicted")}}
    weights = batch.get("sample_weight", jnp.ones_like(t))
    if loss_mode == "bimanual_gated":
        active = batch["action_mask"].reshape(-1, 2, 10).any(axis=-1)
        # Mid-region idle gates have zero low/high strength, excluding idle wrists
        # from the existing source-group safety/ranking/repair normalization.
        gates = jnp.where(active, batch["gates"], (low_threshold + high_threshold) / 2)
        components = bimanual_loss_components_per_sample(
            _MaskedModelView(model, batch),
            batch["x_base"],
            batch["gt"],
            batch["predicted"],
            t,
            batch["tactile"],
            gates,
            source_indices=batch["source"],
            aux_decode_weight=aux_decode_weight,
            aux_decode_steps=aux_decode_steps,
            aux_decode_solver=aux_decode_solver,
            low_gate_safety_weight=low_gate_safety_weight,
            low_gate_safety_margin=low_gate_safety_margin,
            rank_weight=rank_weight,
            rank_margin=rank_margin,
            repair_weight=repair_weight,
            repair_margin=repair_margin,
            low_gate_threshold=low_threshold,
            high_gate_threshold=high_threshold,
        )
        scale = 20 / jnp.sum(batch["action_mask"], axis=-1)
        components["composite_fm"] *= scale
        components["decode"] *= scale
        metrics = {
            key: jnp.mean(value * weights)
            if key in ("composite_fm", "decode", "gt_fm", "vla_fm")
            else jnp.mean(value)
            for key, value in components.items()
        }
        return sum(metrics.values()), metrics
    if loss_mode == "gt":
        endpoint = batch["gt"]
    else:
        raise ValueError("Shared FRS supports gt or bimanual_gated flow matching")
    x_base = batch["x_base"]
    x_t = (1 - t[:, None, None]) * x_base + t[:, None, None] * endpoint
    velocity = shared_velocity(model, x_t, t, shared_condition(model, batch), batch)
    flow = masked_mse(velocity, endpoint - x_base, batch["action_mask"])
    auxiliary = jnp.zeros_like(flow)
    if aux_decode_weight:
        decoded = decode_shared(model, batch, num_steps=aux_decode_steps, solver=aux_decode_solver)
        # Same independently gated VLA/mixed/GT endpoint as the FM trajectory.
        auxiliary = float(aux_decode_weight) * masked_mse(decoded, endpoint, batch["action_mask"])
    return jnp.mean((flow + auxiliary) * weights), {
        "fm_loss": jnp.mean(flow * weights),
        "decode_loss": jnp.mean(auxiliary * weights),
    }


def make_shared_train_step(**loss_options):
    # Match both supported NNX optimizer APIs without changing the frozen base models.
    import inspect

    requires_model = "model" in inspect.signature(nnx.Optimizer.update).parameters

    @nnx.jit
    def step(model, optimizer, batch, key):
        t = jax.random.uniform(key, (batch["x_base"].shape[0],))

        def loss_fn(candidate):
            return shared_flow_loss(candidate, batch, t, **loss_options)

        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        if requires_model:
            optimizer.update(model, grads)
        else:
            optimizer.update(grads)
        return loss, metrics

    return step


def repair_native_actions(
    model, x_base, tactile, *, active_wrists, predicted_action=None, num_steps=10, solver="fireflow"
):
    """Deployment adapter: native 10D insert -> joint 20D -> native right-hand 10D.

    ``x_base`` is the source Pi0.5's reverse-integrated latent, never its sampled
    action endpoint. All inputs use that source checkpoint's normalized space.
    """
    from train_pi05_frs.shared_data import canonical_action, canonical_tactile

    native_width = x_base.shape[-1]
    keys = (
        ["tactile_left_1", "tactile_right_1"]
        if tactile.shape[-2] == 2
        else ["tactile_left_0", "tactile_right_0", "tactile_left_1", "tactile_right_1"]
    )
    batch_size = x_base.shape[0]
    wrists = jnp.asarray(["left" in active_wrists, "right" in active_wrists])
    batch = {
        "x_base": jnp.asarray(canonical_action(x_base, active_wrists)),
        "tactile": jnp.asarray(canonical_tactile(tactile, keys, active_wrists)),
        "action_mask": jnp.broadcast_to(jnp.repeat(wrists, 10), (batch_size, 20)),
        "token_mask": jnp.broadcast_to(jnp.repeat(wrists, 2), (batch_size, 4)),
    }
    repaired = decode_shared(model, batch, num_steps=num_steps, solver=solver)
    if native_width == 10:
        return repaired[..., 10:20]
    # Restore idle native coordinates; padded coordinates are not generated by FRS.
    if predicted_action is None and (native_width == 32 or len(active_wrists) != 2):
        raise ValueError(
            "The frozen VLA predicted_action endpoint is required to restore idle/padded coordinates"
        )
    native = jnp.asarray(x_base if predicted_action is None else predicted_action)
    physical = jnp.where(batch["action_mask"][:, None, :], repaired, native[..., :20])
    return jnp.concatenate((physical, native[..., 20:]), axis=-1)
