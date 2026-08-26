"""Forward-only Pi0.5 source sampling for frozen action-cache production."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np


def fixed_noise(batch_size: int, *, seed: int, horizon: int, action_dim: int):
    """Return one seeded normal sample, copied exactly for every batch row."""
    if batch_size <= 0 or horizon <= 0 or action_dim <= 0:
        raise ValueError("batch_size, horizon, and action_dim must be positive")
    import jax
    import jax.numpy as jnp

    one = jax.random.normal(jax.random.key(seed), (1, horizon, action_dim), dtype=jnp.float32)
    return jnp.broadcast_to(one, (batch_size, horizon, action_dim))


def sample_coarse_actions(model: Any, params: Any, observation: Any, noise: Any, num_steps: int) -> np.ndarray:
    """Run native Pi0.5 forward Euler sampling from supplied fixed noise."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    sampler = getattr(model, "sample_actions", None)
    if not callable(sampler):
        raise TypeError("Pi0.5 model must provide sample_actions")
    try:
        actions = sampler(params, observation, noise=noise, num_steps=num_steps)
    except TypeError:
        # The native NNX module takes its RNG as the first positional argument.
        actions = sampler(params, observation, noise=noise, num_steps=num_steps)
    result = np.asarray(actions, dtype=np.float32)
    if result.ndim != 3 or not np.isfinite(result).all():
        raise ValueError("Pi0.5 sample_actions must return finite [batch, horizon, action] actions")
    return result


def validate_pi05_model(model: Any, *, action_horizon: int) -> int:
    """Validate the native model variant without loading unrelated runtime components."""
    if not bool(getattr(model, "pi05", True)):
        raise ValueError("source checkpoint is not a Pi0.5 model")
    width = int(getattr(model, "action_dim", 0))
    if width < 20:
        raise ValueError("source model action width must be at least 20")
    horizon = int(getattr(model, "action_horizon", action_horizon))
    if horizon != action_horizon:
        raise ValueError("source model horizon does not match cache horizon")
    return width


def load_pi05_source_model(
    checkpoint: str | Path,
    *,
    config: Any,
    model_loader: Callable[..., Any] | None = None,
) -> tuple[Any, int]:
    """Load a native Orbax checkpoint and return its validated model and action width."""
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        raise FileNotFoundError(f"native Orbax checkpoint directory is missing: {checkpoint_path}")
    if not (checkpoint_path / "params").exists():
        raise ValueError("source checkpoint must contain the native Orbax params directory")
    if model_loader is None:
        from lerobot.policies.pi05_jax import load_pi0

        model_loader = load_pi0
    model = model_loader(checkpoint_path, config=config)
    return model, validate_pi05_model(model, action_horizon=int(config.action_horizon))
