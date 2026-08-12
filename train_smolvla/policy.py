from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from utils.source_flow import ReverseSolver, reverse_integrate_prepared_actions

from .checkpoint import load_config, load_params, resolve_checkpoint
from .modeling import JaxSmolVLA
from .preprocessing import JaxSmolVLAPreprocessor, aloha_encode_actions

Array = jax.Array


class JaxSmolVLAPolicy:
    """User-facing stateful policy wrapper around the pure JAX model."""

    def _load_config(self, checkpoint: Path):
        return load_config(checkpoint)

    def _make_model(self, config):
        return JaxSmolVLA(config)

    def _make_preprocessor(
        self,
        checkpoint: Path,
        config,
        *,
        rename_map: Mapping[str, str] | None,
        local_files_only: bool,
    ):
        return JaxSmolVLAPreprocessor(
            checkpoint,
            config,
            rename_map=rename_map,
            local_files_only=local_files_only,
        )

    def _sample_prepared_batch(self, params, batch, rng, **kwargs):
        return self.model.sample_actions(
            params,
            batch["images"],
            batch["image_masks"],
            batch["language_tokens"],
            batch["language_masks"],
            batch["state"],
            rng,
            **kwargs,
        )

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        rename_map: Mapping[str, str] | None = None,
        local_files_only: bool = True,
        revision: str | None = None,
    ):
        self.checkpoint = resolve_checkpoint(checkpoint, revision=revision, local_files_only=local_files_only)
        self.config = self._load_config(self.checkpoint)
        self.params = load_params(self.checkpoint)
        self.model = self._make_model(self.config)
        self.preprocessor = self._make_preprocessor(
            self.checkpoint,
            self.config,
            rename_map=rename_map,
            local_files_only=local_files_only,
        )
        self._compiled_samples: dict[
            tuple[int, int | None, int | None, bool], Any
        ] = {}
        self.reset()

    @classmethod
    def from_pretrained(cls, checkpoint: str | Path, **kwargs: Any) -> JaxSmolVLAPolicy:
        return cls(checkpoint, **kwargs)

    def reset(self) -> None:
        self._action_queue: Array | None = None
        self._queue_index = 0
        self._chunk_index = 0

    def _get_compiled_sample(
        self,
        num_steps: int,
        inference_delay: int | None,
        execution_horizon: int | None,
        has_previous_chunk: bool,
    ):
        cache_key = (
            num_steps,
            inference_delay,
            execution_horizon,
            has_previous_chunk,
        )
        if cache_key not in self._compiled_samples:
            def sample(
                params,
                batch,
                noise,
                previous,
            ):
                return self._sample_prepared_batch(
                    params,
                    batch,
                    jax.random.key(0),
                    noise=noise,
                    num_steps=num_steps,
                    previous_chunk=previous if has_previous_chunk else None,
                    inference_delay=inference_delay,
                    execution_horizon=execution_horizon,
                )

            self._compiled_samples[cache_key] = jax.jit(sample)
        return self._compiled_samples[cache_key]

    def predict_action_chunk(
        self,
        observation: Mapping[str, Any],
        task: str,
        *,
        seed: int = 0,
        noise: Array | None = None,
        jit: bool = True,
        normalized: bool = False,
        num_steps: int | None = None,
        previous_chunk: Array | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
    ) -> Array:
        batch = self.preprocessor.prepare(observation, task)
        if noise is None:
            noise = jax.random.normal(
                jax.random.key(seed),
                (batch["state"].shape[0], self.config.chunk_size, self.config.max_action_dim),
                dtype=jnp.float32,
            )
        num_steps = self.config.num_steps if num_steps is None else num_steps
        if self.config.rtc_config is not None and self.config.rtc_config.enabled and inference_delay is None:
            raise ValueError("RTC inference requires inference_delay")
        previous_argument = previous_chunk
        if previous_argument is None:
            previous_argument = jnp.zeros_like(noise)
        if jit:
            actions = self._get_compiled_sample(
                num_steps,
                inference_delay,
                execution_horizon,
                previous_chunk is not None,
            )(
                self.params,
                batch,
                noise,
                previous_argument,
            )
        else:
            actions = self._sample_prepared_batch(
                self.params,
                batch,
                jax.random.key(seed),
                noise=noise,
                num_steps=num_steps,
                previous_chunk=previous_chunk,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
            )
        if self.config.adapt_to_pi_aloha:
            actions = aloha_encode_actions(actions)
        return actions if normalized else self.preprocessor.unnormalize_actions(actions)

    def reverse_action_chunk(
        self,
        observation: Mapping[str, Any],
        task: str,
        normalized_actions: jax.Array,
        *,
        num_steps: int,
        solver: ReverseSolver,
    ) -> jax.Array:
        actions = jnp.asarray(normalized_actions, dtype=jnp.float32)
        expected = (1, self.config.chunk_size, self.config.action_dim)
        if actions.shape != expected:
            raise ValueError(f"normalized_actions must have shape {expected}, got {actions.shape}")
        if not bool(jnp.isfinite(actions).all()):
            raise ValueError("normalized_actions must be finite")
        batch = self.preprocessor.prepare(observation, task)
        result = reverse_integrate_prepared_actions(
            self,
            batch,
            actions,
            num_steps=num_steps,
            solver=solver,
        )
        if result.shape != expected or not bool(jnp.isfinite(result).all()):
            raise RuntimeError("reverse integration returned an invalid normalized chunk")
        return result

    def select_action(
        self,
        observation: Mapping[str, Any],
        task: str,
        *,
        seed: int = 0,
        jit: bool = True,
        **predict_kwargs: Any,
    ) -> Array:
        if self._action_queue is None or self._queue_index >= self.config.n_action_steps:
            self._action_queue = self.predict_action_chunk(
                observation,
                task,
                seed=seed + self._chunk_index,
                jit=jit,
                **predict_kwargs,
            )
            self._queue_index = 0
            self._chunk_index += 1
        action = self._action_queue[:, self._queue_index]
        self._queue_index += 1
        return action
