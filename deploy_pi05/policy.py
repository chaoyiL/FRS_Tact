"""Live-observation wrapper around the vendored JAX pi0.5 model."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from lerobot.policies.pi05_jax import nnx_utils, transforms
from lerobot.policies.pi05_jax.model import Observation
from lerobot.policies.pi05_jax.normalize import NormStats
from lerobot.policies.pi05_jax.pi0_config import Pi0Config
from lerobot.policies.pi05_jax.policies.pick_tube_policy import PickTubeInputs
from lerobot.policies.pi05_jax.policy_config import load_norm_stats, load_pi0, resolve_checkpoint
from lerobot.policies.pi05_jax.tokenizer import PaligemmaTokenizer
from utils.pi05_source_model import reverse_integrate_actions


def _stats_dim(stats: NormStats) -> int:
    return int(np.asarray(stats.mean).shape[-1])


def _validate_stats(stats: NormStats, *, dim: int, name: str) -> NormStats:
    actual = _stats_dim(stats)
    if actual != dim:
        raise ValueError(
            f"{name} norm stats must have exactly {dim} dimensions for deployment, got {actual}. "
            "Use the assets written by the pi0.5 fine-tune that produced this checkpoint."
        )
    return stats


@dataclasses.dataclass(frozen=True)
class Pi05DeploymentConfig:
    checkpoint: str
    assets_dir: str
    asset_id: str
    camera_map: dict[str, str]
    empty_cameras: tuple[str, ...]
    state_dim: int = 20
    robot_action_dim: int = 20
    action_dim: int = 32
    action_horizon: int = 50
    paligemma_variant: str = "gemma_2b_lora"
    action_expert_variant: str = "gemma_300m_lora"
    use_quantile_norm: bool = True
    state_action_profile: str = "dual-arm-20x20"

    def __post_init__(self) -> None:
        if self.state_action_profile not in {
            "dual-arm-20x20",
            "single-right-arm-7x10",
        }:
            raise ValueError(f"unsupported state/action profile: {self.state_action_profile!r}")
        if min(self.state_dim, self.robot_action_dim, self.action_dim, self.action_horizon) <= 0:
            raise ValueError("pi0.5 dimensions and action horizon must be positive")
        if self.state_dim > self.action_dim or self.robot_action_dim > self.action_dim:
            raise ValueError("robot state/action dimensions cannot exceed model.action_dim")
        mapped = set(self.camera_map)
        empty = set(self.empty_cameras)
        expected = {"left_wrist_0_rgb", "right_wrist_0_rgb"}
        if mapped & empty:
            raise ValueError(f"camera slots cannot be both mapped and empty: {sorted(mapped & empty)}")
        if mapped != expected or empty:
            raise ValueError(
                "pi0.5 pick_tube deployment requires exactly the left/right wrist cameras "
                f"and no empty camera slots; got mapped={sorted(mapped)}, empty={sorted(empty)}"
            )
        if len(set(self.camera_map.values())) != len(self.camera_map):
            raise ValueError("model.camera_map robot observation keys must be unique")


class Pi05RemotePolicy:
    """Prepare live robot observations, sample pi0.5, and invert its flow for FRS."""

    def __init__(self, config: Pi05DeploymentConfig) -> None:
        self.config = config
        image_keys = tuple(config.camera_map)
        model_config = Pi0Config(
            pi05=True,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            paligemma_variant=config.paligemma_variant,
            action_expert_variant=config.action_expert_variant,
            image_keys=image_keys,
        )
        self.checkpoint = resolve_checkpoint(config.checkpoint)
        self.model = load_pi0(self.checkpoint, config=model_config)
        all_stats = load_norm_stats(config.assets_dir, config.asset_id)
        missing = {"state", "actions"} - set(all_stats)
        if missing:
            raise ValueError(f"norm stats are missing keys: {sorted(missing)}")
        self.state_stats = _validate_stats(all_stats["state"], dim=config.state_dim, name="state")
        self.action_stats = _validate_stats(all_stats["actions"], dim=config.robot_action_dim, name="actions")
        self._normalize_state = transforms.Normalize(
            {"state": self.state_stats}, use_quantiles=config.use_quantile_norm, strict=True
        )
        self._unnormalize_actions = transforms.Unnormalize(
            {"actions": self.action_stats}, use_quantiles=config.use_quantile_norm
        )
        self._input_transform = transforms.compose(
            [
                PickTubeInputs(model_type=model_config.model_type, image_keys=image_keys),
                transforms.Normalize(
                    {"state": self.state_stats},
                    use_quantiles=config.use_quantile_norm,
                    strict=True,
                ),
                transforms.TokenizePrompt(
                    PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input,
                ),
                transforms.PadStatesAndActions(model_config.action_dim),
            ]
        )
        self._sample_actions = nnx_utils.module_jit(
            self.model.sample_actions,
            static_argnames=("num_steps",),
        )
        self._rng: jax.Array | None = None
        self._rng_seed: int | None = None

    @property
    def robot_image_keys(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.config.camera_map.values()))

    def _model_input(self, observation: Mapping[str, Any], task: str) -> dict[str, Any]:
        missing = [key for key in self.config.camera_map.values() if key not in observation]
        if missing:
            raise ValueError(f"robot observation is missing RGB keys: {missing}")
        if "observation.state" not in observation:
            raise ValueError("robot observation is missing observation.state")
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        if state.shape != (self.config.state_dim,):
            raise ValueError(f"expected {self.config.state_dim}D state, got {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("robot state contains NaN or Inf")
        images = {}
        for slot, robot_key in self.config.camera_map.items():
            image = np.asarray(observation[robot_key])
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(
                    f"{robot_key} must have shape (224, 224, 3) and dtype uint8, "
                    f"got shape {image.shape} and dtype {image.dtype}"
                )
            images[slot] = image
        return {"image": images, "state": state, "prompt": str(task)}

    def prepare_observation(self, observation: Mapping[str, Any], task: str) -> Observation:
        data = jax.tree.map(np.asarray, self._input_transform(self._model_input(observation, task)))
        prepared = Observation.from_dict(data)
        return jax.tree.map(lambda value: jnp.asarray(value)[None, ...], prepared)

    def normalize_state(self, state: Any) -> jax.Array:
        array = np.asarray(state, dtype=np.float32)
        if array.shape[-1] != self.config.state_dim:
            raise ValueError(f"state last dimension must be {self.config.state_dim}, got {array.shape}")
        normalized = self._normalize_state({"state": array})["state"]
        return jnp.asarray(normalized, dtype=jnp.float32)

    def unnormalize_actions(self, actions: Any) -> np.ndarray:
        array = np.asarray(jax.device_get(actions), dtype=np.float32)
        if array.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"model-space action last dimension must be {self.config.action_dim}, got {array.shape}"
            )
        real = array[..., : self.config.robot_action_dim]
        output = self._unnormalize_actions({"actions": real})["actions"]
        output = np.asarray(output, dtype=np.float32)
        if not np.isfinite(output).all():
            raise ValueError("unnormalized pi0.5 action contains NaN or Inf")
        return output

    def predict_action_chunk(
        self,
        observation: Mapping[str, Any],
        task: str,
        *,
        seed: int,
        num_steps: int,
    ) -> jax.Array:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        prepared = self.prepare_observation(observation, task)
        if self._rng is None:
            self._rng = jax.random.key(seed)
            self._rng_seed = int(seed)
        elif int(seed) != self._rng_seed:
            raise ValueError(
                f"pi0.5 inference seed changed from {self._rng_seed} to {seed} after sampling started"
            )
        self._rng, sample_rng = jax.random.split(self._rng)
        actions = self._sample_actions(
            sample_rng,
            prepared,
            num_steps=num_steps,
        )
        expected = (1, self.config.action_horizon, self.config.action_dim)
        if actions.shape != expected:
            raise ValueError(f"pi0.5 action must have shape {expected}, got {actions.shape}")
        return actions

    def reverse_action_chunk(
        self,
        observation: Mapping[str, Any],
        task: str,
        actions: Any,
        *,
        num_steps: int,
        solver: str,
    ) -> jax.Array:
        prepared = self.prepare_observation(observation, task)
        return reverse_integrate_actions(
            self.model,
            prepared,
            jnp.asarray(actions, dtype=jnp.float32),
            num_steps=num_steps,
            solver=solver,
        )
