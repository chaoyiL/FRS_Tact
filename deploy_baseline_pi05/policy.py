"""Frozen visual-only Pi0.5 action sampler for direct tactile steering."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from lerobot.policies.pi05_jax import transforms
from lerobot.policies.pi05_jax.model import Observation
from lerobot.policies.pi05_jax.nnx_utils import module_jit
from lerobot.policies.pi05_jax.pi0_config import Pi0Config
from lerobot.policies.pi05_jax.policies.pick_tube_policy import PickTubeInputs
from lerobot.policies.pi05_jax.policy_config import load_norm_stats, load_pi0, resolve_checkpoint
from lerobot.policies.pi05_jax.tokenizer import PaligemmaTokenizer

from .deployment import DeploymentConfig


class Pi05VisualPolicy:
    """Load a strict Pi0.5 LoRA checkpoint and sample its visual action chunk once."""

    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config.source
        model_config = Pi0Config(
            pi05=True,
            action_dim=config.source.model_action_dim,
            action_horizon=config.source.action_horizon,
            paligemma_variant=config.source.paligemma_variant,
            action_expert_variant=config.source.action_expert_variant,
            image_keys=tuple(config.source.camera_map),
        )
        self._model_config = model_config
        self.checkpoint = resolve_checkpoint(config.source.checkpoint)
        self.model = load_pi0(self.checkpoint, config=model_config, allow_extra_params=False)
        all_stats = load_norm_stats(config.norm_stats.directory, config.norm_stats.asset_id)
        if set(("state", "actions")) - set(all_stats):
            raise ValueError("Pi0.5 norm stats must contain state and actions")
        self._action_stats = all_stats["actions"]
        self._state_stats = all_stats["state"]
        if np.asarray(self._action_stats.mean).shape != (self.config.action_dim,):
            raise ValueError(f"Pi0.5 action norm stats must be exactly {self.config.action_dim}D")
        if np.asarray(self._state_stats.mean).shape != (self.config.state_dim,):
            raise ValueError(f"Pi0.5 state norm stats must be exactly {self.config.state_dim}D")
        self._input_transform = transforms.compose(
            [
                PickTubeInputs(model_type=model_config.model_type, image_keys=tuple(config.source.camera_map)),
                transforms.Normalize({"state": self._state_stats}, use_quantiles=config.norm_stats.use_quantile_norm, strict=True),
                transforms.ResizeImages(224, 224),
                transforms.TokenizePrompt(PaligemmaTokenizer(model_config.max_token_len), discrete_state_input=model_config.discrete_state_input),
                transforms.PadStatesAndActions(model_config.action_dim),
            ]
        )
        self._use_quantiles = config.norm_stats.use_quantile_norm
        # Freeze graph/state once so every inference reuses the whole-policy executable.
        self._sample_actions = module_jit(self.model.sample_actions, static_argnames=("num_steps",))
        # Match the training cache's fixed_noise(batch_size=1) exactly.
        self._rng = jax.random.key(0)
        self._noise = jax.random.normal(
            self._rng, (1, self.config.action_horizon, self.config.model_action_dim), dtype=jnp.float32
        )

    def _model_input(self, observation: Mapping[str, Any], task: str) -> dict[str, Any]:
        state_key = "observation.state"
        if state_key not in observation:
            raise ValueError("visual Pi0.5 observation is missing observation.state")
        state = np.asarray(observation[state_key], dtype=np.float32)
        if state.shape != (self.config.state_dim,) or not np.isfinite(state).all():
            raise ValueError(f"visual Pi0.5 state must be finite with shape ({self.config.state_dim},)")
        images: dict[str, np.ndarray] = {}
        for slot, source_key in self.config.camera_map.items():
            if source_key not in observation:
                raise ValueError(f"visual Pi0.5 observation is missing {source_key}")
            image = np.asarray(observation[source_key])
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(f"{source_key} must be uint8 RGB shaped (224, 224, 3)")
            images[slot] = image
        return {"image": images, "state": state, "prompt": str(task)}

    def _prepare(self, observation: Mapping[str, Any], task: str) -> Observation:
        # Build a fresh visual-only mapping; tactile keys cannot reach the source policy.
        transformed = self._input_transform(self._model_input(observation, task))
        data = jax.tree.map(np.asarray, transformed)
        prepared = Observation.from_dict(data)
        return jax.tree.map(lambda value: jnp.asarray(value)[None, ...], prepared)

    def predict_action_chunk(
        self,
        observation: Mapping[str, Any],
        task: str,
        *,
        seed: int = 0,
        num_steps: int = 10,
    ) -> np.ndarray:
        if seed != 0 or num_steps != 10:
            raise ValueError("direct Pi0.5 sampling requires fixed seed=0 and num_steps=10")
        actions = self._sample_actions(
            self._rng, self._prepare(observation, task), noise=self._noise, num_steps=10
        )
        array = np.ascontiguousarray(np.asarray(jax.device_get(actions), dtype=np.float32))
        expected = (1, self.config.action_horizon, self.config.model_action_dim)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"Pi0.5 sample_actions must return finite {expected}, got {array.shape}")
        return np.ascontiguousarray(array[..., : self.config.action_dim])

    def unnormalize_actions(self, actions: Any) -> np.ndarray:
        normalized = np.asarray(actions, dtype=np.float32)
        if normalized.ndim == 0 or normalized.shape[-1] != self.config.action_dim or not np.isfinite(normalized).all():
            raise ValueError(f"normalized Pi0.5 actions must be finite with a {self.config.action_dim}D last axis")
        if self._use_quantiles:
            q01 = np.asarray(self._action_stats.q01, dtype=np.float32)
            q99 = np.asarray(self._action_stats.q99, dtype=np.float32)
            physical = (normalized + 1.0) * 0.5 * (q99 - q01 + 1e-6) + q01
        else:
            mean = np.asarray(self._action_stats.mean, dtype=np.float32)
            std = np.asarray(self._action_stats.std, dtype=np.float32)
            physical = normalized * (std + 1e-6) + mean
        physical = np.ascontiguousarray(np.asarray(physical, dtype=np.float32))
        if not np.isfinite(physical).all():
            raise ValueError("inverse-normalized Pi0.5 actions must be finite")
        return physical
