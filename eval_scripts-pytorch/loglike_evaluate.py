from __future__ import annotations

import csv
import dataclasses
import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from utils import EpisodeData, infer_modality_keys, stack_frame_batches


DEFAULT_HUTCHINSON_SAMPLES = 1
DEFAULT_HUTCHINSON_SEED = 0
ODE_SOLVER_EULER = "euler"
ODE_SOLVER_FIREFLOW = "fireflow"
ODE_SOLVERS = (ODE_SOLVER_EULER, ODE_SOLVER_FIREFLOW)
MODALITIES = ("vision", "tactile", "state", "language_prompt")


@dataclasses.dataclass(frozen=True)
class LikelihoodIntegrationResult:
    x_base: Tensor
    r_tot: Tensor
    log_p_base: Tensor
    log_likelihood: Tensor
    nfe: int


@dataclasses.dataclass(frozen=True)
class VelocityContext:
    prefix_pad_masks: Tensor
    past_key_values: Any
    action_dim: int


VelocityFn = Callable[[Tensor, Tensor], Tensor]
VelocityTraceFn = Callable[[Tensor, Tensor, int], tuple[Tensor, Tensor]]


def _validate_solver(ode_solver: str) -> str:
    if ode_solver not in ODE_SOLVERS:
        raise ValueError(f"ode_solver must be one of {ODE_SOLVERS}, got {ode_solver!r}")
    return ode_solver


def _detach_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    return value


def _repeat_pairs(tensor: Tensor) -> Tensor:
    return tensor.repeat_interleave(2, dim=0)


def _last_true_indices(mask: Tensor) -> Tensor:
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand_as(mask)
    return torch.where(mask, positions, torch.full_like(positions, -1)).amax(dim=1)


def create_paired_velocity_context(
    policy: SmolVLAPolicy,
    frame_batches: Sequence[dict[str, Any]],
    *,
    modality: str,
    vision_keys: Sequence[str] | None = None,
    tactile_keys: Sequence[str] | None = None,
) -> VelocityContext:
    """Build interleaved original/ablated prefix KV caches for several frames."""

    if modality not in MODALITIES:
        raise ValueError(f"Unsupported modality {modality!r}; expected one of {MODALITIES}")
    if not frame_batches:
        raise ValueError("At least one frame batch is required")
    if not policy.config.use_cache:
        raise ValueError("Likelihood evaluation requires policy.config.use_cache=True")

    present_image_keys = tuple(
        key for key in policy.config.image_features if key in frame_batches[0]
    )
    keys = (
        *present_image_keys,
        OBS_STATE,
        OBS_LANGUAGE_TOKENS,
        OBS_LANGUAGE_ATTENTION_MASK,
    )
    stacked = stack_frame_batches(frame_batches, keys)
    if not present_image_keys:
        raise ValueError("No policy image features are present in the evaluation batch")
    inferred_vision, inferred_tactile = infer_modality_keys(
        present_image_keys,
        vision_keys=vision_keys,
        tactile_keys=tactile_keys,
    )
    ablated_image_keys: set[str]
    if modality == "vision":
        ablated_image_keys = set(inferred_vision)
    elif modality == "tactile":
        ablated_image_keys = set(inferred_tactile)
    else:
        ablated_image_keys = set()
    if modality in ("vision", "tactile") and not ablated_image_keys:
        raise ValueError(
            f"No image keys were classified as {modality!r}. "
            f"Available policy image keys: {list(present_image_keys)}"
        )

    images, image_masks = policy.prepare_images(stacked)
    state = policy.prepare_state(stacked)
    lang_tokens = stacked[OBS_LANGUAGE_TOKENS]
    lang_masks = stacked[OBS_LANGUAGE_ATTENTION_MASK].bool()

    paired_images = [_repeat_pairs(image) for image in images]
    paired_image_masks = [_repeat_pairs(mask.bool()) for mask in image_masks]
    paired_state = _repeat_pairs(state)
    paired_lang_tokens = _repeat_pairs(lang_tokens)
    paired_lang_masks = _repeat_pairs(lang_masks)
    ablated_rows = torch.arange(paired_state.shape[0], device=paired_state.device) % 2 == 1

    for image_index, key in enumerate(present_image_keys):
        if key in ablated_image_keys:
            paired_image_masks[image_index] = paired_image_masks[image_index].clone()
            paired_image_masks[image_index][ablated_rows] = False
    if modality == "language_prompt":
        paired_lang_masks = paired_lang_masks.clone()
        paired_lang_masks[ablated_rows] = False

    model = policy.model
    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            paired_images,
            paired_image_masks,
            paired_lang_tokens,
            paired_lang_masks,
            state=paired_state,
        )
        if modality == "state":
            state_positions = _last_true_indices(prefix_pad_masks)
            if bool((state_positions < 0).any()):
                raise ValueError("Could not locate the state token in the prefix mask")
            prefix_pad_masks = prefix_pad_masks.clone()
            row_indices = torch.arange(prefix_pad_masks.shape[0], device=prefix_pad_masks.device)
            prefix_pad_masks[row_indices[ablated_rows], state_positions[ablated_rows]] = False

        prefix_attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_attention_mask,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

    action_feature = policy.config.action_feature
    if action_feature is None:
        raise ValueError("Checkpoint config has no action feature")
    return VelocityContext(
        prefix_pad_masks=prefix_pad_masks.detach(),
        past_key_values=_detach_tree(past_key_values),
        action_dim=int(action_feature.shape[0]),
    )


def predict_velocity_with_context(
    policy: SmolVLAPolicy,
    context: VelocityContext,
    x: Tensor,
    timestep: Tensor,
) -> Tensor:
    """Evaluate the SmolVLA probability-flow velocity with a cached prefix."""

    velocity = policy.model.denoise_step(
        prefix_pad_masks=context.prefix_pad_masks,
        past_key_values=context.past_key_values,
        x_t=x,
        timestep=timestep,
    ).to(dtype=torch.float32)
    if context.action_dim < velocity.shape[-1]:
        velocity = velocity.clone()
        velocity[..., context.action_dim :] = 0
    return velocity


def _rademacher_probe(
    x: Tensor,
    *,
    seed: int,
    action_dim: int | None = None,
) -> Tensor:
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    event_shape = (1, *x.shape[1:])
    probe = torch.randint(0, 2, event_shape, generator=generator, device=x.device)
    probe = probe.to(dtype=x.dtype).mul_(2).sub_(1).expand_as(x)
    if action_dim is not None and action_dim < x.shape[-1]:
        probe = probe.clone()
        probe[..., action_dim:] = 0
    return probe


def velocity_and_hutchinson_trace(
    velocity_fn: Callable[[Tensor], Tensor],
    x: Tensor,
    *,
    num_samples: int,
    seed: int,
    action_dim: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Estimate div(v) with Rademacher Hutchinson probes using reverse-mode AD."""

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    x_with_grad = x.detach().to(dtype=torch.float32).requires_grad_(True)
    with torch.enable_grad():
        velocity = velocity_fn(x_with_grad).to(dtype=torch.float32)
        trace = torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)
        event_axes = tuple(range(1, x.ndim))
        for sample_index in range(num_samples):
            probe = _rademacher_probe(
                x_with_grad,
                seed=seed + sample_index,
                action_dim=action_dim,
            )
            vector_jacobian = torch.autograd.grad(
                outputs=velocity,
                inputs=x_with_grad,
                grad_outputs=probe,
                retain_graph=sample_index + 1 < num_samples,
                create_graph=False,
            )[0]
            trace = trace + (vector_jacobian * probe).sum(dim=event_axes)
    return velocity.detach(), (trace / num_samples).detach()


def _run_euler_likelihood(
    *,
    x: Tensor,
    r_tot: Tensor,
    t: Tensor,
    num_steps: int,
    dt: float,
    velocity_trace_fn: VelocityTraceFn,
) -> tuple[Tensor, Tensor, int]:
    for step in range(num_steps):
        velocity, divergence = velocity_trace_fn(x, t, step)
        x = (x + velocity * dt).detach()
        r_tot = r_tot + divergence * dt
        t = t + dt
    return x, r_tot, num_steps


def _run_fireflow_likelihood(
    *,
    x: Tensor,
    r_tot: Tensor,
    t: Tensor,
    num_steps: int,
    dt: float,
    velocity_fn: VelocityFn,
    velocity_trace_fn: VelocityTraceFn,
) -> tuple[Tensor, Tensor, int]:
    velocity_initial = velocity_fn(x, t)
    x_mid = x + 0.5 * dt * velocity_initial
    t_mid = t + 0.5 * dt
    velocity_mid_previous, divergence_mid = velocity_trace_fn(x_mid, t_mid, 0)
    x = (x + dt * velocity_mid_previous).detach()
    r_tot = r_tot + dt * divergence_mid
    t = t + dt

    for step in range(1, num_steps):
        x_mid = x + 0.5 * dt * velocity_mid_previous
        t_mid = t + 0.5 * dt
        velocity_mid, divergence_mid = velocity_trace_fn(x_mid, t_mid, step)
        x = (x + dt * velocity_mid).detach()
        r_tot = r_tot + dt * divergence_mid
        t = t + dt
        velocity_mid_previous = velocity_mid
    return x, r_tot, num_steps + 1


def standard_normal_log_prob(x: Tensor, *, action_dim: int | None = None) -> Tensor:
    if action_dim is not None:
        x = x[..., :action_dim]
    event_axes = tuple(range(1, x.ndim))
    event_size = math.prod(x.shape[1:])
    return -0.5 * (
        x.square().sum(dim=event_axes) + event_size * math.log(2.0 * math.pi)
    )


def integrate_to_base_log_likelihood(
    policy: SmolVLAPolicy,
    context: VelocityContext,
    reference_actions: Tensor,
    *,
    num_steps: int,
    hutchinson_samples: int = DEFAULT_HUTCHINSON_SAMPLES,
    hutchinson_seed: int = DEFAULT_HUTCHINSON_SEED,
    ode_solver: str = ODE_SOLVER_EULER,
) -> LikelihoodIntegrationResult:
    """Integrate normalized data actions at t=0 to standard-normal noise at t=1."""

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if hutchinson_samples <= 0:
        raise ValueError(f"hutchinson_samples must be positive, got {hutchinson_samples}")
    ode_solver = _validate_solver(ode_solver)

    x = reference_actions.detach().to(
        device=context.prefix_pad_masks.device,
        dtype=torch.float32,
    )
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.shape[0] != context.prefix_pad_masks.shape[0]:
        raise ValueError(
            f"Action batch size {x.shape[0]} does not match context batch size "
            f"{context.prefix_pad_masks.shape[0]}"
        )
    if context.action_dim < x.shape[-1]:
        x = x.clone()
        x[..., context.action_dim :] = 0

    batch_size = x.shape[0]
    t = torch.zeros(batch_size, dtype=torch.float32, device=x.device)
    r_tot = torch.zeros(batch_size, dtype=torch.float32, device=x.device)
    dt = 1.0 / num_steps

    def velocity_fn(x_arg: Tensor, t_arg: Tensor) -> Tensor:
        with torch.no_grad():
            return predict_velocity_with_context(policy, context, x_arg, t_arg).detach()

    def velocity_trace_fn(x_arg: Tensor, t_arg: Tensor, step: int) -> tuple[Tensor, Tensor]:
        return velocity_and_hutchinson_trace(
            lambda differentiable_x: predict_velocity_with_context(
                policy, context, differentiable_x, t_arg
            ),
            x_arg,
            num_samples=hutchinson_samples,
            seed=hutchinson_seed + step * max(hutchinson_samples, 1),
            action_dim=context.action_dim,
        )

    if ode_solver == ODE_SOLVER_EULER:
        x, r_tot, nfe = _run_euler_likelihood(
            x=x,
            r_tot=r_tot,
            t=t,
            num_steps=num_steps,
            dt=dt,
            velocity_trace_fn=velocity_trace_fn,
        )
    else:
        x, r_tot, nfe = _run_fireflow_likelihood(
            x=x,
            r_tot=r_tot,
            t=t,
            num_steps=num_steps,
            dt=dt,
            velocity_fn=velocity_fn,
            velocity_trace_fn=velocity_trace_fn,
        )

    log_p_base = standard_normal_log_prob(x, action_dim=context.action_dim)
    log_likelihood = log_p_base + r_tot
    return LikelihoodIntegrationResult(
        x_base=x,
        r_tot=r_tot,
        log_p_base=log_p_base,
        log_likelihood=log_likelihood,
        nfe=nfe,
    )


def _scalar(tensor: Tensor) -> float:
    return float(tensor.detach().cpu().reshape(-1)[0])


def compute_episode_modality_contributions(
    policy: SmolVLAPolicy,
    episode: EpisodeData,
    *,
    modality: str,
    num_steps: int,
    hutchinson_samples: int = DEFAULT_HUTCHINSON_SAMPLES,
    hutchinson_seed: int = DEFAULT_HUTCHINSON_SEED,
    ode_solver: str = ODE_SOLVER_EULER,
    eval_batch_size: int = 4,
    vision_keys: Sequence[str] | None = None,
    tactile_keys: Sequence[str] | None = None,
) -> list[dict[str, float | int]]:
    if eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {eval_batch_size}")

    rows: list[dict[str, float | int]] = []
    for start in range(0, len(episode.frames), eval_batch_size):
        stop = min(start + eval_batch_size, len(episode.frames))
        chunk_batches = episode.batches[start:stop]
        context = create_paired_velocity_context(
            policy,
            chunk_batches,
            modality=modality,
            vision_keys=vision_keys,
            tactile_keys=tactile_keys,
        )
        action_batch = torch.stack(episode.actions[start:stop], dim=0)
        paired_actions = _repeat_pairs(action_batch)
        result = integrate_to_base_log_likelihood(
            policy,
            context,
            paired_actions,
            num_steps=num_steps,
            hutchinson_samples=hutchinson_samples,
            hutchinson_seed=hutchinson_seed,
            ode_solver=ode_solver,
        )

        for offset, (frame, dataset_index) in enumerate(
            zip(
                episode.frames[start:stop],
                episode.dataset_indices[start:stop],
                strict=True,
            )
        ):
            original_index = 2 * offset
            ablated_index = original_index + 1
            row = {
                "frame": int(frame),
                "dataset_index": int(dataset_index),
                "original_log_likelihood": _scalar(result.log_likelihood[original_index]),
                "ablated_log_likelihood": _scalar(result.log_likelihood[ablated_index]),
                "original_r_tot": _scalar(result.r_tot[original_index]),
                "ablated_r_tot": _scalar(result.r_tot[ablated_index]),
                "delta_logp": _scalar(
                    result.log_p_base[original_index] - result.log_p_base[ablated_index]
                ),
                "delta_r_tot": _scalar(
                    result.r_tot[original_index] - result.r_tot[ablated_index]
                ),
                "contribution": _scalar(
                    result.log_likelihood[original_index]
                    - result.log_likelihood[ablated_index]
                ),
            }
            rows.append(row)
            print(
                f"modality={modality} frame={row['frame']} dataset_index={row['dataset_index']} "
                f"original_log_likelihood={row['original_log_likelihood']:.6f} "
                f"ablated_log_likelihood={row['ablated_log_likelihood']:.6f} "
                f"delta_logp={row['delta_logp']:.6f} "
                f"delta_r_tot={row['delta_r_tot']:.6f}"
            )
    return rows


def save_contribution_curve(
    rows: Sequence[dict[str, float | int]],
    *,
    output_dir: Path,
    modality: str,
    episode_index: str,
) -> tuple[Path, Path]:
    if not rows:
        raise ValueError("Cannot save an empty contribution curve")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{modality}_contribution_episode_{episode_index}.csv"
    fieldnames = [
        "frame",
        "dataset_index",
        "original_log_likelihood",
        "ablated_log_likelihood",
        "original_r_tot",
        "ablated_r_tot",
        "delta_logp",
        "delta_r_tot",
        "contribution",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plot_path = output_dir / f"{modality}_contribution_components_episode_{episode_index}.png"
    frames = [row["frame"] for row in rows]
    curves = (
        ("contribution", f"{modality} contribution"),
        ("delta_logp", "delta_logp(x_base)"),
        ("delta_r_tot", "delta_r_tot"),
    )
    figure, axes = plt.subplots(len(curves), 1, figsize=(10, 9), sharex=True)
    figure.suptitle(f"{modality} contribution components over episode {episode_index}")
    for axis, (field, label) in zip(axes, curves, strict=True):
        axis.plot(frames, [row[field] for row in rows], marker="o", linewidth=1.5)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Episode frame")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return csv_path, plot_path

