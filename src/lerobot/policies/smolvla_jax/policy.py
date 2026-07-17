from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from .checkpoint import load_config, load_params, resolve_checkpoint
from .modeling import JaxSmolVLA
from .preprocessing import JaxSmolVLAPreprocessor, aloha_encode_actions

Array = jax.Array


class JaxSmolVLAPolicy:
    """User-facing stateful policy wrapper around the pure JAX model."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        rename_map: Mapping[str, str] | None = None,
        local_files_only: bool = True,
        revision: str | None = None,
    ):
        self.checkpoint = resolve_checkpoint(checkpoint, revision=revision, local_files_only=local_files_only)
        self.config = load_config(self.checkpoint)
        self.params = load_params(self.checkpoint)
        self.model = JaxSmolVLA(self.config)
        self.preprocessor = JaxSmolVLAPreprocessor(
            self.checkpoint,
            self.config,
            rename_map=rename_map,
            local_files_only=local_files_only,
        )
        self._compiled_samples: dict[tuple[int, int | None, int | None, bool], Any] = {}
        self.reset()

    @classmethod
    def from_pretrained(cls, checkpoint: str | Path, **kwargs: Any) -> JaxSmolVLAPolicy:
        return cls(checkpoint, **kwargs)

    def reset(self) -> None:
        self._action_queue: Array | None = None
        self._queue_index = 0

    def _get_compiled_sample(
        self,
        num_steps: int,
        inference_delay: int | None,
        execution_horizon: int | None,
        has_previous_chunk: bool,
    ):
        cache_key = (num_steps, inference_delay, execution_horizon, has_previous_chunk)
        if cache_key not in self._compiled_samples:
            model = self.model

            def sample(params, images, image_masks, tokens, language_masks, state, noise, previous):
                return model.sample_actions(
                    params,
                    images,
                    image_masks,
                    tokens,
                    language_masks,
                    state,
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
                batch["images"],
                batch["image_masks"],
                batch["language_tokens"],
                batch["language_masks"],
                batch["state"],
                noise,
                previous_argument,
            )
        else:
            actions = self.model.sample_actions(
                self.params,
                batch["images"],
                batch["image_masks"],
                batch["language_tokens"],
                batch["language_masks"],
                batch["state"],
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
                seed=seed,
                jit=jit,
                **predict_kwargs,
            )
            self._queue_index = 0
        action = self._action_queue[:, self._queue_index]
        self._queue_index += 1
        return action
