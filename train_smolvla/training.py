from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from .checkpoint import load_params, save_portable_params, write_effective_config
from .configuration import JaxSmolVLAConfig
from .lora import initialize_lora_params, is_trainable_parameter
from .modeling import JaxSmolVLA
from .modality_dropout import ModalityDropoutConfig, apply_modality_dropout
from .sharding import create_data_parallel_mesh, replicate_tree, shard_batch

Array = jax.Array
Params = dict[str, Array]
RESUME_METADATA_FILENAME = "resume_metadata.json"
RESUME_METADATA_VERSION = 1


@struct.dataclass
class TrainState:
    step: Array
    params: Params
    opt_state: optax.OptState
    rng: Array


def partition_params(
    params: Mapping[str, Array],
    config: JaxSmolVLAConfig,
    *,
    classifier=is_trainable_parameter,
) -> tuple[Params, Params]:
    trainable: Params = {}
    frozen: Params = {}
    for name, value in params.items():
        (trainable if classifier(name, config) else frozen)[name] = value
    return trainable, frozen


def promote_trainable_params_to_fp32(
    params: Mapping[str, Array],
    config: JaxSmolVLAConfig,
    *,
    classifier=is_trainable_parameter,
) -> Params:
    """Keep FP32 master weights for every trainable floating-point parameter."""

    promoted: Params = {}
    for name, value in params.items():
        if (
            classifier(name, config)
            and jnp.issubdtype(value.dtype, jnp.inexact)
            and value.dtype != jnp.float32
        ):
            value = value.astype(jnp.float32)
        promoted[name] = value
    return promoted


def cast_trainable_params_for_compute(params: Mapping[str, Array]) -> Params:
    """Use BF16 trainable weights for forward/backward while retaining FP32 masters."""

    return {
        name: value.astype(jnp.bfloat16)
        if jnp.issubdtype(value.dtype, jnp.inexact)
        else value
        for name, value in params.items()
    }


def merge_params(trainable: Mapping[str, Array], frozen: Mapping[str, Array]) -> Params:
    return {**frozen, **trainable}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parameter_signature(params: Mapping[str, Array]) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(params.items())
    }


def _dropout_signature(config: ModalityDropoutConfig) -> dict[str, Any]:
    return {
        "enable": config.enable,
        "every_n_steps": config.every_n_steps,
        "prob": config.prob,
        "drop_language": config.drop_language,
        "drop_state": config.drop_state,
        "camera_indices": config.camera_indices,
    }


def _mapping_differences(saved: Any, current: Any, prefix: str = "") -> list[str]:
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        differences: list[str] = []
        for key in sorted(set(saved) | set(current)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in saved:
                differences.append(f"{path}: missing from checkpoint")
            elif key not in current:
                differences.append(f"{path}: missing from current configuration")
            else:
                differences.extend(_mapping_differences(saved[key], current[key], path))
        return differences
    if saved != current:
        return [f"{prefix}: checkpoint={saved!r}, current={current!r}"]
    return []


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

    def _prepare_parameter_partition(
        self,
        params: Mapping[str, Array],
    ) -> tuple[Params, Params]:
        params = initialize_lora_params(params, self.config, seed=self.seed)
        params = promote_trainable_params_to_fp32(params, self.config)
        return partition_params(params, self.config)

    def _write_effective_config(self, destination: str | Path) -> Path:
        return write_effective_config(destination, self.config)

    def __init__(
        self,
        model: JaxSmolVLA,
        params: Mapping[str, Array],
        *,
        seed: int = 0,
        total_steps: int | None = None,
        modality_dropout: ModalityDropoutConfig | Mapping[str, Any] | None = None,
    ):
        self.model = model
        self.config = model.config
        self.seed = int(seed)
        self.total_steps = None if total_steps is None else int(total_steps)
        if isinstance(modality_dropout, ModalityDropoutConfig):
            self.modality_dropout = modality_dropout
        else:
            self.modality_dropout = ModalityDropoutConfig.from_dict(modality_dropout)
        self._modality_dropout_rng = np.random.default_rng(self.seed + 17)
        self._host_step = 0
        self.last_dropout_info: dict[str, Any] = {
            "applied": False,
            "modality": "none",
            "camera_index": -1,
        }
        trainable, self.frozen_params = self._prepare_parameter_partition(params)
        self.optimizer, self.learning_rate = create_optimizer(self.config, self.total_steps)
        self.state = TrainState(
            step=jnp.asarray(0, dtype=jnp.int32),
            params=trainable,
            opt_state=self.optimizer.init(trainable),
            rng=jax.random.key(self.seed),
        )
        self._compiled_step = jax.jit(self._train_step, donate_argnums=(0,))
        self._compiled_evals: dict[tuple[bool, int], Any] = {}
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

        def loss_fn(
            trainable_params: Mapping[str, Array],
        ) -> tuple[Array, Mapping[str, Array]]:
            compute_params = cast_trainable_params_for_compute(trainable_params)
            params = merge_params(compute_params, frozen_params)
            loss, metrics = self.model.compute_training_loss(
                params,
                batch=batch,
                rng=loss_rng,
            )
            return loss, metrics

        (loss, model_metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, opt_state = self.optimizer.update(gradients, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        metrics = {
            **model_metrics,
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
        batch, drop_info = apply_modality_dropout(
            batch,
            step=self._host_step,
            rng=self._modality_dropout_rng,
            config=self.modality_dropout,
        )
        if self.mesh is not None:
            batch = shard_batch(batch, self.mesh)
        self.state, metrics = self._compiled_step(self.state, self.frozen_params, batch)
        self._host_step += 1
        self.last_dropout_info = drop_info
        metrics = dict(metrics)
        metrics["modality_dropout_applied"] = np.asarray(
            1.0 if drop_info["applied"] else 0.0, dtype=jnp.float32
        )
        # Encode dropped modality for logging: -1=none, -2=language, -3=state, >=0=camera index.
        if not drop_info["applied"]:
            dropped_code = -1
        elif drop_info["modality"] == "language":
            dropped_code = -2
        elif drop_info["modality"] == "state":
            dropped_code = -3
        else:
            dropped_code = int(drop_info["camera_index"])
        metrics["modality_dropout_code"] = np.asarray(dropped_code, dtype=jnp.int32)
        return metrics

    @property
    def step_count(self) -> int:
        """Host-side step counter that avoids synchronizing the GPU every iteration."""

        return self._host_step

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
        loss, model_metrics = self.model.compute_training_loss(
            params,
            batch=batch,
            rng=loss_rng,
        )
        metrics: dict[str, Array] = {
            **model_metrics,
            "loss": loss,
            "n_samples": jnp.asarray(batch["actions"].shape[0], dtype=jnp.float32),
        }
        if not rollout:
            metrics["action_mse"] = jnp.asarray(0.0, dtype=jnp.float32)
            return metrics

        predicted = self.model.sample_actions_from_batch(
            params,
            batch,
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

        cache_key = (bool(rollout), steps)
        compiled = self._compiled_evals.get(cache_key)
        if compiled is None:
            rollout_enabled = bool(rollout)

            def eval_fn(
                params: Params,
                batch: Mapping[str, Array],
                rng: Array,
            ) -> dict[str, Array]:
                return self._eval_batch(
                    params,
                    batch,
                    rng,
                    rollout=rollout_enabled,
                    rollout_steps=steps,
                )

            compiled = jax.jit(eval_fn)
            self._compiled_evals[cache_key] = compiled

        total_loss = 0.0
        total_mse = 0.0
        total_weight = 0.0
        n_batches = 0
        rng = jax.random.key(seed)
        params = self.compute_params

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

    @property
    def compute_params(self) -> Params:
        return merge_params(
            cast_trainable_params_for_compute(self.state.params),
            self.frozen_params,
        )

    def _resume_signature(self) -> dict[str, Any]:
        return _jsonable(
            {
                "seed": self.seed,
                "total_steps": self.total_steps,
                "model": self.config.to_dict(),
                "modality_dropout": _dropout_signature(self.modality_dropout),
            }
        )

    def _validate_resume_compatibility(
        self,
        checkpoint: Path,
        trainable: Mapping[str, Array],
    ) -> dict[str, Any] | None:
        trainable_keys_file = checkpoint / "trainable_keys.json"
        if not trainable_keys_file.is_file():
            raise FileNotFoundError(
                f"resume checkpoint is missing trainable key manifest: {trainable_keys_file}"
            )
        with trainable_keys_file.open(encoding="utf-8") as file:
            saved_keys = set(json.load(file))
        current_keys = set(trainable)
        if saved_keys != current_keys:
            added = sorted(current_keys - saved_keys)
            removed = sorted(saved_keys - current_keys)
            raise ValueError(
                "resume trainable parameter set does not match the checkpoint; "
                f"newly trainable={added[:8]} no longer trainable={removed[:8]}. "
                "Keep module_modes and LoRA targets unchanged for strict resume."
            )

        metadata_path = checkpoint / RESUME_METADATA_FILENAME
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as file:
                metadata = json.load(file)
            if int(metadata.get("version", -1)) != RESUME_METADATA_VERSION:
                raise ValueError(f"unsupported resume metadata version in {metadata_path}")
            saved_signature = metadata.get("resume_signature")
            if isinstance(saved_signature, Mapping) and "seed" in saved_signature:
                # The checkpoint seed is authoritative for the resumed data stream.
                self.seed = int(saved_signature["seed"])
            differences = _mapping_differences(
                saved_signature,
                self._resume_signature(),
            )
            saved_parameters = metadata.get("trainable_parameters")
            differences.extend(
                _mapping_differences(saved_parameters, _parameter_signature(trainable), "trainable")
            )
            if differences:
                preview = "\n  ".join(differences[:20])
                raise ValueError(
                    "resume configuration is incompatible with the checkpoint:\n  " + preview
                )
            return metadata

        # Checkpoints produced before resume_metadata.json still contain the
        # effective LoRA configuration in config.json.
        config_path = checkpoint / "config.json"
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as file:
                saved_config = json.load(file)
            legacy_saved = {
                "module_modes": saved_config.get("module_modes"),
                "lora_rank": saved_config.get("lora_rank"),
                "lora_alpha": saved_config.get("lora_alpha"),
                "vlm_lora_target_modules": saved_config.get("vlm_lora_target_modules", []),
            }
            legacy_current = {
                "module_modes": self.config.module_modes,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "vlm_lora_target_modules": list(self.config.vlm_lora_target_modules),
            }
            differences = _mapping_differences(legacy_saved, _jsonable(legacy_current))
            if differences:
                raise ValueError(
                    "resume LoRA configuration is incompatible with the checkpoint:\n  "
                    + "\n  ".join(differences)
                )
        return None

    def save(self, destination: str | Path, *, source_dir: str | Path | None = None) -> Path:
        destination = save_portable_params(
            self.full_params,
            destination,
            source_dir=source_dir,
            overwrite=True,
        )
        self._write_effective_config(destination)
        training_state = {
            "step": self.state.step,
            "opt_state": self.state.opt_state,
            "rng_data": jax.random.key_data(self.state.rng),
        }
        (destination / "training_state.msgpack").write_bytes(flax.serialization.to_bytes(training_state))
        with (destination / "trainable_keys.json").open("w") as file:
            json.dump(sorted(self.state.params), file, indent=2)
            file.write("\n")
        resume_metadata = {
            "version": RESUME_METADATA_VERSION,
            "resume_signature": self._resume_signature(),
            "trainable_parameters": _parameter_signature(self.state.params),
            "modality_dropout_rng_state": _jsonable(
                self._modality_dropout_rng.bit_generator.state
            ),
        }
        with (destination / RESUME_METADATA_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(resume_metadata, file, indent=2, sort_keys=True)
            file.write("\n")
        return destination

    def restore(self, checkpoint: str | Path) -> None:
        checkpoint = Path(checkpoint)
        params = load_params(checkpoint)
        trainable, frozen = self._prepare_parameter_partition(params)
        resume_metadata = self._validate_resume_compatibility(checkpoint, trainable)
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
        self._host_step = int(np.asarray(jax.device_get(restored["step"])))
        if resume_metadata is not None:
            dropout_rng_state = resume_metadata.get("modality_dropout_rng_state")
            if dropout_rng_state is None:
                raise ValueError("resume metadata is missing modality_dropout_rng_state")
            self._modality_dropout_rng.bit_generator.state = dropout_rng_state
        elif self.modality_dropout.enable:
            warnings.warn(
                "legacy checkpoint has no modality-dropout RNG state; "
                "the dropout sequence cannot resume exactly",
                stacklevel=2,
            )
