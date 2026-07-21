from __future__ import annotations

import json
from collections.abc import Mapping
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
