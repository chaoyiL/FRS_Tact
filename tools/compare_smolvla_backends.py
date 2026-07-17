#!/usr/bin/env python
"""Numerically compare the PyTorch and JAX SmolVLA implementations."""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import jax
import jax.numpy as jnp
import numpy as np
import torch

from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.policies.smolvla_jax import JaxSmolVLA, JaxSmolVLAConfig
from lerobot.policies.smolvla_jax.checkpoint import load_params


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference = reference.astype(np.float32).reshape(-1)
    actual = actual.astype(np.float32).reshape(-1)
    difference = actual - reference
    denominator = np.linalg.norm(reference) * np.linalg.norm(actual)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "cosine": float(np.vdot(reference, actual) / denominator),
    }


def compare_transformer(
    torch_policy: SmolVLAPolicy,
    jax_model: JaxSmolVLA,
    jax_params: dict[str, jax.Array],
    seed: int,
) -> dict[str, dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)
    prefix = torch.randn(1, 5, 960, generator=generator)
    suffix = torch.randn(1, 4, 720, generator=generator)
    pad = torch.ones(1, 9, dtype=torch.bool)
    attention_ar = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.bool)
    attention_mask = make_att_2d_masks(pad, attention_ar)
    position_ids = torch.cumsum(pad, dim=1) - 1
    with torch.inference_mode():
        (torch_prefix, torch_suffix), _ = torch_policy.model.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
            fill_kv_cache=False,
        )
    jax_prefix, jax_suffix, _ = jax_model.transformer(
        jax_params,
        jnp.asarray(prefix.numpy()),
        jnp.asarray(suffix.numpy()),
        jnp.asarray(attention_mask.numpy()),
        jnp.asarray(position_ids.numpy()),
    )
    return {
        "prefix": metrics(torch_prefix.float().numpy(), np.asarray(jax_prefix, dtype=np.float32)),
        "suffix": metrics(torch_suffix.float().numpy(), np.asarray(jax_suffix, dtype=np.float32)),
    }


def compare_image(
    torch_policy: SmolVLAPolicy,
    jax_model: JaxSmolVLA,
    jax_params: dict[str, jax.Array],
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(1, 3, 512, 512, generator=generator).clamp(-1, 1)
    with torch.inference_mode():
        torch_output = torch_policy.model.vlm_with_expert.embed_image(image)
    jax_output = jax_model.embed_image(jax_params, jnp.asarray(image.numpy()))
    return metrics(torch_output.float().numpy(), np.asarray(jax_output, dtype=np.float32))


def compare_denoise(
    torch_policy: SmolVLAPolicy,
    jax_model: JaxSmolVLA,
    jax_params: dict[str, jax.Array],
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(1, 3, 512, 512, generator=generator).clamp(-1, 1)
    image_mask = torch.ones(1, dtype=torch.bool)
    tokens = torch.randint(0, 1000, (1, 8), generator=generator)
    language_mask = torch.ones_like(tokens, dtype=torch.bool)
    state = torch.randn(1, 32, generator=generator)
    noisy_actions = torch.randn(1, 50, 32, generator=generator)
    timestep = torch.full((1,), 0.5)

    with torch.inference_mode():
        prefix, prefix_pad, prefix_ar = torch_policy.model.embed_prefix(
            [image], [image_mask], tokens, language_mask, state
        )
        prefix_mask = make_att_2d_masks(prefix_pad, prefix_ar)
        prefix_positions = torch.cumsum(prefix_pad, dim=1) - 1
        _, cache = torch_policy.model.vlm_with_expert.forward(
            attention_mask=prefix_mask,
            position_ids=prefix_positions,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        torch_velocity = torch_policy.model.denoise_step(
            prefix_pad,
            cache,
            noisy_actions,
            timestep,
        )

    context = jax_model.build_prefix_context(
        jax_params,
        jnp.asarray(image.numpy())[:, None],
        jnp.asarray(image_mask.numpy())[:, None],
        jnp.asarray(tokens.numpy()),
        jnp.asarray(language_mask.numpy()),
        jnp.asarray(state.numpy()),
    )
    jax_velocity = jax_model.denoise_step(
        jax_params,
        context,
        jnp.asarray(noisy_actions.numpy()),
        jnp.asarray(timestep.numpy()),
    )
    return metrics(torch_velocity.numpy(), np.asarray(jax_velocity, dtype=np.float32))


def compare_sample(
    torch_policy: SmolVLAPolicy,
    jax_model: JaxSmolVLA,
    jax_params: dict[str, jax.Array],
    seed: int,
    num_steps: int,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(1, 3, 512, 512, generator=generator).clamp(-1, 1)
    image_mask = torch.ones(1, dtype=torch.bool)
    tokens = torch.randint(0, 1000, (1, 8), generator=generator)
    language_mask = torch.ones_like(tokens, dtype=torch.bool)
    state = torch.randn(1, 32, generator=generator)
    noise = torch.randn(1, 50, 32, generator=generator)
    torch_policy.model.config.num_steps = num_steps
    with torch.inference_mode():
        torch_actions = torch_policy.model.sample_actions(
            [image], [image_mask], tokens, language_mask, state, noise=noise
        )
    jax_actions = jax_model.sample_actions(
        jax_params,
        jnp.asarray(image.numpy())[:, None],
        jnp.asarray(image_mask.numpy())[:, None],
        jnp.asarray(tokens.numpy()),
        jnp.asarray(language_mask.numpy()),
        jnp.asarray(state.numpy()),
        jax.random.key(seed),
        noise=jnp.asarray(noise.numpy()),
        num_steps=num_steps,
    )
    torch_actions = torch_actions[..., : jax_model.config.action_dim]
    return metrics(torch_actions.numpy(), np.asarray(jax_actions, dtype=np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--stage",
        choices=("transformer", "image", "denoise", "sample", "all"),
        default="transformer",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--layers", type=int, help="Compare only the first N decoder layers")
    parser.add_argument(
        "--float32", action="store_true", help="Upcast all weights for semantic parity checks"
    )
    parser.add_argument("--num-steps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = JaxSmolVLAConfig.from_pretrained(args.checkpoint)
    if args.layers is not None:
        config = dataclasses.replace(config, num_vlm_layers=args.layers)
    jax_model = JaxSmolVLA(config)
    jax_params = load_params(args.checkpoint)
    if args.float32:
        jax_params = jax.tree.map(
            lambda value: value.astype(jnp.float32) if jnp.issubdtype(value.dtype, jnp.inexact) else value,
            jax_params,
        )
    torch_policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    torch_policy.eval()
    if args.float32:
        torch_policy.float()
    if args.layers is not None:
        torch_policy.model.vlm_with_expert.num_vlm_layers = args.layers

    if args.stage in ("transformer", "all"):
        print({"transformer": compare_transformer(torch_policy, jax_model, jax_params, args.seed)})
    if args.stage in ("image", "all"):
        print({"image": compare_image(torch_policy, jax_model, jax_params, args.seed)})
    if args.stage in ("denoise", "all"):
        print({"denoise": compare_denoise(torch_policy, jax_model, jax_params, args.seed)})
    if args.stage in ("sample", "all"):
        print({"sample": compare_sample(torch_policy, jax_model, jax_params, args.seed, args.num_steps)})


if __name__ == "__main__":
    main()
