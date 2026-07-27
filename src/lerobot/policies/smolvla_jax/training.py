from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp
import optax
from flax import struct

from .checkpoint import load_params, save_portable_params, write_effective_config
from .configuration import JaxSmolVLAConfig
from .lora import initialize_lora_params, is_trainable_parameter
from .modeling import JaxSmolVLA
from .sharding import create_data_parallel_mesh, replicate_tree, shard_batch

Array = jax.Array
Params = dict[str, Array]


@struct.dataclass
class TrainState:
    step: Array
    params: Params
    opt_state: optax.OptState
    rng: Array


def partition_params(params: Mapping[str, Array], config: JaxSmolVLAConfig) -> tuple[Params, Params]:
    trainable: Params = {}
    frozen: Params = {}
    for name, value in params.items():
        (trainable if is_trainable_parameter(name, config) else frozen)[name] = value
    return trainable, frozen


def merge_params(trainable: Mapping[str, Array], frozen: Mapping[str, Array]) -> Params:
    return {**frozen, **trainable}


def cosine_warmup_schedule(config: JaxSmolVLAConfig, total_steps: int | None = None):
    warmup_steps = config.scheduler_warmup_steps
    decay_steps = config.scheduler_decay_steps
    if total_steps is not None and total_steps < decay_steps:
        scale = total_steps / decay_steps
        warmup_steps = int(warmup_steps * scale)
        decay_steps = total_steps

    def schedule(step: Array) -> Array:
        step = jnp.asarray(step, dtype=jnp.float32)
        warmup_denominator = max(warmup_steps, 1)
        warmup_start = config.optimizer_lr / (warmup_steps + 1)
        warmup = warmup_start + (config.optimizer_lr - warmup_start) * (step / warmup_denominator)
        clipped_step = jnp.minimum(step, decay_steps)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * clipped_step / max(decay_steps, 1)))
        decay = config.scheduler_decay_lr + (config.optimizer_lr - config.scheduler_decay_lr) * cosine
        return jnp.where(step < warmup_steps, warmup, decay)

    return schedule


def create_optimizer(config: JaxSmolVLAConfig, total_steps: int | None = None):
    schedule = cosine_warmup_schedule(config, total_steps)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.optimizer_grad_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=config.optimizer_beta1,
            b2=config.optimizer_beta2,
            eps=config.optimizer_eps,
            weight_decay=config.optimizer_weight_decay,
        ),
    )
    return optimizer, schedule


class JaxSmolVLATrainer:
    """JIT-compiled single- or multi-device SmolVLA training state machine."""

    def __init__(
        self,
        model: JaxSmolVLA,
        params: Mapping[str, Array],
        *,
        seed: int = 0,
        total_steps: int | None = None,
    ):
        self.model = model
        self.config = model.config
        params = initialize_lora_params(params, self.config, seed=seed)
        trainable, self.frozen_params = partition_params(params, self.config)
        self.optimizer, self.learning_rate = create_optimizer(self.config, total_steps)
        self.state = TrainState(
            step=jnp.asarray(0, dtype=jnp.int32),
            params=trainable,
            opt_state=self.optimizer.init(trainable),
            rng=jax.random.key(seed),
        )
        self._compiled_step = jax.jit(self._train_step, donate_argnums=(0,))
        self.mesh = None

    def enable_data_parallel(self) -> None:
        """Replicate model state and shard future batches over all visible devices."""

        self.mesh = create_data_parallel_mesh()
        self.state = replicate_tree(self.state, self.mesh)
        self.frozen_params = replicate_tree(self.frozen_params, self.mesh)

    def _train_step(
        self,
        state: TrainState,
        frozen_params: Mapping[str, Array],
        batch: Mapping[str, Array],
    ) -> tuple[TrainState, dict[str, Array]]:
        next_rng, loss_rng = jax.random.split(state.rng)
        loss_rng = jax.random.fold_in(loss_rng, state.step)

        def loss_fn(trainable_params: Mapping[str, Array]) -> Array:
            params = merge_params(trainable_params, frozen_params)
            return self.model.loss(params, batch, loss_rng)

        loss, gradients = jax.value_and_grad(loss_fn)(state.params)
        updates, opt_state = self.optimizer.update(gradients, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        metrics = {
            "loss": loss,
            "grad_norm": optax.tree.norm(gradients),
            "learning_rate": self.learning_rate(state.step),
        }
        return state.replace(
            step=state.step + 1,
            params=params,
            opt_state=opt_state,
            rng=next_rng,
        ), metrics

    def step(self, batch: Mapping[str, Any]) -> dict[str, Array]:
        batch = jax.tree.map(jnp.asarray, batch)
        if self.mesh is not None:
            batch = shard_batch(batch, self.mesh)
        self.state, metrics = self._compiled_step(self.state, self.frozen_params, batch)
        return metrics

    def _eval_batch(
        self,
        params: Params,
        batch: Mapping[str, Array],
        rng: Array,
        *,
        rollout: bool,
        rollout_steps: int,
    ) -> dict[str, Array]:
        loss_rng, sample_rng = jax.random.split(rng)
        loss = self.model.loss(params, batch, loss_rng)
        metrics: dict[str, Array] = {
            "loss": loss,
            "n_samples": jnp.asarray(batch["actions"].shape[0], dtype=jnp.float32),
        }
        if not rollout:
            metrics["action_mse"] = jnp.asarray(0.0, dtype=jnp.float32)
            return metrics

        predicted = self.model.sample_actions(
            params,
            batch["images"],
            batch["image_masks"],
            batch["language_tokens"],
            batch["language_masks"],
            batch["state"],
            sample_rng,
            num_steps=rollout_steps,
        )
        target = batch["actions"][..., : self.config.action_dim]
        errors = jnp.square(predicted - target)
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is not None:
            valid = (~action_is_pad).astype(errors.dtype)[..., None]
            errors = errors * valid
            denominator = jnp.maximum(jnp.sum(valid) * errors.shape[-1], 1.0)
            action_mse = jnp.sum(errors) / denominator
        else:
            action_mse = jnp.mean(errors)
        metrics["action_mse"] = action_mse
        return metrics

    def evaluate(
        self,
        batches: Iterable[Mapping[str, Any]],
        *,
        seed: int = 0,
        max_batches: int | None = None,
        rollout: bool = True,
        rollout_steps: int | None = None,
    ) -> dict[str, float]:
        """Run FM validation (and optional action-chunk rollouts) over finite batches."""

        steps = self.config.num_steps if rollout_steps is None else int(rollout_steps)
        if steps <= 0:
            raise ValueError(f"rollout_steps must be positive, got {steps}")

        def eval_fn(params: Params, batch: Mapping[str, Array], rng: Array) -> dict[str, Array]:
            return self._eval_batch(
                params,
                batch,
                rng,
                rollout=bool(rollout),
                rollout_steps=steps,
            )

        compiled = jax.jit(eval_fn)

        total_loss = 0.0
        total_mse = 0.0
        total_weight = 0.0
        n_batches = 0
        rng = jax.random.key(seed)
        params = self.full_params

        for batch in batches:
            if max_batches is not None and n_batches >= max_batches:
                break
            batch = jax.tree.map(jnp.asarray, batch)
            if self.mesh is not None:
                batch = shard_batch(batch, self.mesh)
            rng, batch_rng = jax.random.split(rng)
            metrics = jax.device_get(compiled(params, batch, batch_rng))
            weight = float(metrics["n_samples"])
            total_loss += float(metrics["loss"]) * weight
            total_mse += float(metrics["action_mse"]) * weight
            total_weight += weight
            n_batches += 1

        if n_batches == 0 or total_weight <= 0:
            raise ValueError("validation produced no batches")

        return {
            "loss": total_loss / total_weight,
            "action_mse": total_mse / total_weight if rollout else float("nan"),
            "n_samples": total_weight,
            "n_batches": float(n_batches),
        }

    @property
    def full_params(self) -> Params:
        return merge_params(self.state.params, self.frozen_params)

    def save(self, destination: str | Path, *, source_dir: str | Path | None = None) -> Path:
        destination = save_portable_params(
            self.full_params,
            destination,
            source_dir=source_dir,
            overwrite=True,
        )
        write_effective_config(destination, self.config)
        training_state = {
            "step": self.state.step,
            "opt_state": self.state.opt_state,
            "rng_data": jax.random.key_data(self.state.rng),
        }
        (destination / "training_state.msgpack").write_bytes(flax.serialization.to_bytes(training_state))
        with (destination / "trainable_keys.json").open("w") as file:
            json.dump(sorted(self.state.params), file, indent=2)
            file.write("\n")
        return destination

    def restore(self, checkpoint: str | Path) -> None:
        checkpoint = Path(checkpoint)
        params = initialize_lora_params(load_params(checkpoint), self.config)
        trainable, frozen = partition_params(params, self.config)
        target = {
            "step": self.state.step,
            "opt_state": self.optimizer.init(trainable),
            "rng_data": jax.random.key_data(self.state.rng),
        }
        state_file = checkpoint / "training_state.msgpack"
        if not state_file.is_file():
            raise FileNotFoundError(f"training state not found: {state_file}")
        restored = flax.serialization.from_bytes(target, state_file.read_bytes())
        self.frozen_params = frozen
        self.state = TrainState(
            step=restored["step"],
            params=trainable,
            opt_state=restored["opt_state"],
            rng=jax.random.wrap_key_data(restored["rng_data"]),
        )
