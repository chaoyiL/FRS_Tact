"""Condition-swap experiment for a trained unconditional action decoder.

For a fixed normalized action chunk ``a_i``, this program computes

    z_ij = E(a_i, o_j),    a_hat_ij = D(z_ij)

where ``E`` is SmolVLA reverse integration under observation ``o_j`` and ``D``
is a trained :class:`SelfAttentionFlowDecoder`.  It reports both the exact
quantities from the experiment,

    R_ij = ||a_hat_ij - a_i||_2^2
    Delta z_ijk = ||z_ij - z_ik||_2,
    r_null = ||J_D(z_ii)(z_ij - z_ii)||_2 / ||z_ij - z_ii||_2,

and element-normalized MSE/RMS variants that are easier to compare across
action shapes. The Jacobian-vector product is evaluated directly with JAX JVP;
the full decoder Jacobian is never materialized.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Literal

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla_jax.data import action_delta_timestamps

from modalities_eval.utils import SmolVLAEvalModel
from modalities_eval.utils import load_model
from utils.cache import CachedPairs
from utils.cache import atomic_write_json
from utils.checkpoint import load_checkpoint
from utils.integration import fireflow_integrate_velocity
from utils.model import FlowSolver
from utils.model import SelfAttentionFlowDecoder
from utils.model import decode_actions
from utils.source_model import reverse_integrate_actions
from utils.source_model import stack_observations

SourceSolver = Literal["euler", "fireflow", "slerpflow"]

_DEFAULT_CACHE_DIR = _PROJECT_ROOT / "cache" / "tactile_test_05"
_DEFAULT_DECODER_CHECKPOINT = _PROJECT_ROOT / "decode_tests" / "result" / "best"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "decode_tests" / "condition_swap_result"


def select_experiment_indices(
    candidates: np.ndarray,
    episode_indices: np.ndarray,
    *,
    num_actions: int,
    num_conditions: int,
    seed: int,
    anchor_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select anchors and per-anchor conditions, with own condition in column zero.

    Alternative conditions preferentially come from distinct episodes.  This
    avoids filling an experiment with adjacent, nearly identical video frames.
    """
    candidates = np.unique(np.asarray(candidates, dtype=np.int64))
    episodes = np.asarray(episode_indices, dtype=np.int64)
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("Candidate cache indices must be a non-empty 1-D array.")
    if int(candidates.min()) < 0 or int(candidates.max()) >= len(episodes):
        raise ValueError("Candidate cache indices are outside the cache bounds.")
    if num_actions <= 0:
        raise ValueError(f"num_actions must be positive, got {num_actions}.")
    if num_conditions < 2:
        raise ValueError(
            f"num_conditions must be at least 2 to compare conditions, got {num_conditions}."
        )
    if candidates.size < num_conditions:
        raise ValueError(
            f"Need at least {num_conditions} candidate samples, got {candidates.size}."
        )

    rng = np.random.default_rng(seed)
    candidate_set = set(int(index) for index in candidates)
    if anchor_indices is None:
        if candidates.size < num_actions:
            raise ValueError(
                f"Need {num_actions} anchor actions but the selected split has {candidates.size}."
            )
        anchors = rng.choice(candidates, size=num_actions, replace=False).astype(np.int64)
    else:
        anchors = np.asarray(anchor_indices, dtype=np.int64)
        if anchors.ndim != 1 or anchors.size == 0:
            raise ValueError("anchor_indices must be a non-empty 1-D sequence.")
        if len(np.unique(anchors)) != len(anchors):
            raise ValueError("anchor_indices contains duplicates.")
        missing = [int(index) for index in anchors if int(index) not in candidate_set]
        if missing:
            raise ValueError(
                f"Anchor cache indices are not in the selected split: {missing}. "
                "Use --split all to allow any cache row."
            )

    condition_rows: list[list[int]] = []
    for anchor in anchors:
        anchor_int = int(anchor)
        shuffled = rng.permutation(candidates)
        alternatives = [int(index) for index in shuffled if int(index) != anchor_int]
        chosen = [anchor_int]
        seen_episodes = {int(episodes[anchor_int])}

        # First pass: different episode for every alternative where possible.
        for index in alternatives:
            episode = int(episodes[index])
            if episode in seen_episodes:
                continue
            chosen.append(index)
            seen_episodes.add(episode)
            if len(chosen) == num_conditions:
                break

        # Small datasets may not contain enough distinct episodes.
        if len(chosen) < num_conditions:
            chosen_set = set(chosen)
            for index in alternatives:
                if index in chosen_set:
                    continue
                chosen.append(index)
                chosen_set.add(index)
                if len(chosen) == num_conditions:
                    break
        if len(chosen) != num_conditions:
            raise ValueError(
                f"Could only select {len(chosen)} conditions for anchor {anchor_int}; "
                f"requested {num_conditions}."
            )
        condition_rows.append(chosen)

    return anchors, np.asarray(condition_rows, dtype=np.int64)


def classify_outcome(
    *,
    mean_latent_delta_rms: float,
    reconstruction_degradation_mse: float,
    latent_rms_epsilon: float,
    reconstruction_degradation_epsilon: float,
) -> tuple[str, str]:
    """Map aggregate measurements to the three hypotheses in the experiment."""
    if latent_rms_epsilon < 0 or reconstruction_degradation_epsilon < 0:
        raise ValueError("Classification epsilons must be non-negative.")
    if mean_latent_delta_rms <= latent_rms_epsilon:
        return (
            "condition_ignored",
            "Delta z is approximately zero; the encoder is effectively ignoring the condition.",
        )
    if reconstruction_degradation_mse <= reconstruction_degradation_epsilon:
        return (
            "decoder_redundant_or_null_direction",
            "Delta z is non-zero without material reconstruction degradation; condition "
            "information lies in decoder-redundant/null directions.",
        )
    return (
        "action_code_damaged",
        "Delta z is non-zero and reconstruction degrades; changing the condition damages "
        "the action code.",
    )


def decoder_directional_derivatives(
    model: SelfAttentionFlowDecoder,
    base_latents: jax.Array,
    latent_directions: jax.Array,
    *,
    num_steps: int,
    solver: FlowSolver,
) -> jax.Array:
    """Compute batched ``J_D(z) @ delta_z`` without materializing a Jacobian.

    Each batch row is independent. ``base_latents[b]`` is the linearization
    point and ``latent_directions[b]`` is its observation-induced direction.
    """
    base_latents = jnp.asarray(base_latents, dtype=jnp.float32)
    latent_directions = jnp.asarray(latent_directions, dtype=jnp.float32)
    if base_latents.shape != latent_directions.shape:
        raise ValueError(
            "base_latents and latent_directions must have identical shapes, got "
            f"{base_latents.shape} and {latent_directions.shape}."
        )
    if base_latents.ndim != 3:
        raise ValueError(f"Expected latent batches [B, T, A], got {base_latents.shape}.")

    def decode_batch(latents: jax.Array) -> jax.Array:
        # Match utils.model.decode_actions without nesting its nnx.jit wrapper
        # inside jax.jvp (nested NNX graph extraction crosses trace levels).
        if solver == "euler":
            batch_size = latents.shape[0]
            dt = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)

            def body(step: int, x_t: jax.Array) -> jax.Array:
                t = jnp.full((batch_size,), step * dt, dtype=jnp.float32)
                return x_t + dt * model(x_t, t)

            return jax.lax.fori_loop(0, num_steps, body, latents)
        if solver == "fireflow":
            return fireflow_integrate_velocity(
                lambda x, t: model(x, t), latents, num_steps=num_steps
            )
        raise ValueError(f"Unsupported decoder solver: {solver!r}.")

    _, directional_derivatives = jax.jvp(
        decode_batch, (base_latents,), (latent_directions,)
    )
    return directional_derivatives


def jacobian_null_ratios(
    directional_derivatives: jax.Array,
    latent_directions: jax.Array,
) -> jax.Array:
    """Return ``||J_D(z) delta_z||_2 / ||delta_z||_2`` per batch row."""
    directional_derivatives = jnp.asarray(directional_derivatives, dtype=jnp.float32)
    latent_directions = jnp.asarray(latent_directions, dtype=jnp.float32)
    if directional_derivatives.shape != latent_directions.shape:
        raise ValueError(
            "Decoder JVP and latent directions must have identical shapes, got "
            f"{directional_derivatives.shape} and {latent_directions.shape}."
        )
    if latent_directions.ndim < 2:
        raise ValueError(
            f"Expected a batch plus feature dimensions, got {latent_directions.shape}."
        )
    axes = tuple(range(1, latent_directions.ndim))
    output_norm = jnp.sqrt(jnp.sum(jnp.square(directional_derivatives), axis=axes))
    input_norm = jnp.sqrt(jnp.sum(jnp.square(latent_directions), axis=axes))
    return jnp.where(input_norm > 0, output_norm / input_norm, jnp.nan)


def _read_split_indices(
    pairs: CachedPairs,
    *,
    split: str,
    split_json: pathlib.Path | None,
) -> np.ndarray:
    sample_count = int(pairs.manifest["sample_count"])
    if split == "all":
        return np.arange(sample_count, dtype=np.int64)
    if split_json is None or not split_json.is_file():
        raise FileNotFoundError(
            f"Decoder split file not found: {split_json}. "
            "Pass --split-json or use --split all."
        )
    with split_json.open(encoding="utf-8") as file:
        payload = json.load(file)
    expected_digest = str(pairs.manifest["records_sha256"])
    actual_digest = str(payload.get("records_sha256", ""))
    if actual_digest != expected_digest:
        raise ValueError(
            "Decoder split and cache do not match: "
            f"records_sha256 {actual_digest!r} != {expected_digest!r}."
        )
    key = f"{split}_indices"
    if key not in payload:
        raise ValueError(f"Split file {split_json} has no {key!r} array.")
    indices = np.asarray(payload[key], dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"Split array {key!r} is empty or not 1-D.")
    if int(indices.min()) < 0 or int(indices.max()) >= sample_count:
        raise ValueError(f"Split array {key!r} contains out-of-range cache indices.")
    return indices


def _manifest_path(
    configuration: Mapping[str, Any],
    key: str,
    override: pathlib.Path | None,
) -> pathlib.Path | None:
    if override is not None:
        return override.expanduser().resolve()
    value = configuration.get(key)
    if value is None:
        return None
    return pathlib.Path(str(value)).expanduser().resolve()


def _load_source_model(
    pairs: CachedPairs,
    *,
    source_checkpoint_dir: pathlib.Path | None,
    dataset_root: pathlib.Path | None,
    allow_download: bool,
) -> SmolVLAEvalModel:
    configuration = pairs.manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("Cache manifest has no valid source-model configuration.")
    checkpoint_dir = _manifest_path(
        configuration, "checkpoint_dir", source_checkpoint_dir
    )
    if checkpoint_dir is None:
        raise ValueError(
            "Source checkpoint is absent from the cache manifest; pass --source-checkpoint-dir."
        )
    resolved_dataset_root = _manifest_path(configuration, "dataset_root", dataset_root)
    dataset_repo_id = configuration.get("dataset_repo_id")
    if not dataset_repo_id:
        raise ValueError("dataset_repo_id is absent from the cache manifest.")
    rename_map = configuration.get("rename_map")
    if rename_map is not None and not isinstance(rename_map, Mapping):
        raise ValueError("Cache manifest rename_map must be an object or null.")
    return load_model(
        checkpoint_dir,
        dataset_repo_id=str(dataset_repo_id),
        dataset_root=resolved_dataset_root,
        dataset_revision=configuration.get("dataset_revision"),
        action_key=configuration.get("action_key"),
        rename_map=rename_map,
        local_files_only=not allow_download,
    )


def _create_dataset(model: SmolVLAEvalModel) -> LeRobotDataset:
    metadata = LeRobotDatasetMetadata(
        model.dataset_repo_id,
        root=model.dataset_root,
        revision=model.dataset_revision,
    )
    return LeRobotDataset(
        model.dataset_repo_id,
        root=model.dataset_root,
        revision=model.dataset_revision,
        delta_timestamps=action_delta_timestamps(
            model.action_key,
            model.config.chunk_size,
            metadata.fps,
        ),
    )


def _load_condition_observations(
    model: SmolVLAEvalModel,
    pairs: CachedPairs,
    condition_indices: np.ndarray,
) -> dict[int, Any]:
    dataset = _create_dataset(model)
    unique_cache_indices = np.unique(condition_indices)
    observations: dict[int, Any] = {}
    for position, cache_index in enumerate(unique_cache_indices, start=1):
        dataset_index = int(pairs.arrays["dataset_index"][cache_index])
        sample = dataset[dataset_index]
        observation, _, _ = model.prepare_sample(sample)
        observations[int(cache_index)] = observation
        if position == 1 or position % 25 == 0 or position == len(unique_cache_indices):
            print(
                f"loaded observations {position}/{len(unique_cache_indices)}",
                flush=True,
            )
    return observations


def _describe(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> pathlib.Path:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _write_arrays(
    path: pathlib.Path,
    *,
    anchor_indices: np.ndarray,
    condition_indices: np.ndarray,
    actions: np.ndarray,
    latents: np.ndarray,
    reconstructions: np.ndarray,
    decoder_jvps: np.ndarray,
) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        anchor_cache_indices=anchor_indices,
        condition_cache_indices=condition_indices,
        fixed_actions=actions,
        latents=latents,
        reconstructed_actions=reconstructions,
        decoder_jvp_at_original_latent=decoder_jvps,
    )
    temporary.replace(path)
    return path


def _plot_jacobian_null_space(
    path: pathlib.Path,
    *,
    reconstruction_rows: list[dict[str, Any]],
    num_actions: int,
    null_ratio_epsilon: float,
) -> pathlib.Path:
    swapped = [row for row in reconstruction_rows if not row["is_original_condition"]]
    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for anchor_slot in range(num_actions):
        rows = [row for row in swapped if row["anchor_slot"] == anchor_slot]
        axis.scatter(
            [row["delta_z_from_original_rms"] for row in rows],
            [row["r_null"] for row in rows],
            s=34,
            alpha=0.8,
            label=f"a{anchor_slot}",
        )
    axis.axhline(
        null_ratio_epsilon,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"null threshold={null_ratio_epsilon:g}",
    )
    axis.set_xlabel(r"$\Delta z_o$ from original condition (RMS)")
    axis.set_ylabel(r"$r_{null}=\|J_D(z)\Delta z_o\|_2/\|\Delta z_o\|_2$")
    axis.set_title(r"Decoder Jacobian null-space test at $z=z_{ii}$")
    axis.grid(True, alpha=0.25)
    if num_actions <= 10:
        axis.legend(fontsize=8, ncol=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.png")
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    temporary.replace(path)
    return path


def _plot_results(
    path: pathlib.Path,
    *,
    reconstruction_rows: list[dict[str, Any]],
    num_actions: int,
) -> pathlib.Path:
    swapped = [row for row in reconstruction_rows if not row["is_original_condition"]]
    original = [row for row in reconstruction_rows if row["is_original_condition"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    scatter = axes[0]
    for anchor_slot in range(num_actions):
        rows = [row for row in swapped if row["anchor_slot"] == anchor_slot]
        scatter.scatter(
            [row["delta_z_from_original_rms"] for row in rows],
            [row["reconstruction_mse"] for row in rows],
            s=28,
            alpha=0.75,
            label=f"a{anchor_slot}",
        )
    scatter.scatter(
        np.zeros(len(original)),
        [row["reconstruction_mse"] for row in original],
        marker="x",
        s=45,
        color="black",
        label="original condition",
    )
    scatter.set_xlabel(r"$\Delta z$ from original condition (RMS)")
    scatter.set_ylabel("reconstruction MSE")
    scatter.set_title("Condition sensitivity vs reconstruction")
    scatter.grid(True, alpha=0.25)
    if num_actions <= 10:
        scatter.legend(fontsize=7, ncol=2)

    comparison = axes[1]
    positions = np.arange(num_actions)
    original_mse = np.asarray(
        [
            next(
                row["reconstruction_mse"]
                for row in original
                if row["anchor_slot"] == anchor_slot
            )
            for anchor_slot in range(num_actions)
        ]
    )
    swapped_mean = np.asarray(
        [
            np.mean(
                [
                    row["reconstruction_mse"]
                    for row in swapped
                    if row["anchor_slot"] == anchor_slot
                ]
            )
            for anchor_slot in range(num_actions)
        ]
    )
    for position, before, after in zip(positions, original_mse, swapped_mean):
        comparison.plot([position, position], [before, after], color="#999999", linewidth=1)
    comparison.scatter(
        positions, original_mse, color="#4C72B0", label="original condition", zorder=3
    )
    comparison.scatter(
        positions, swapped_mean, color="#C44E52", label="mean swapped", zorder=3
    )
    comparison.set_xlabel("fixed action slot i")
    comparison.set_ylabel("reconstruction MSE")
    comparison.set_title("Original vs swapped-condition reconstruction")
    comparison.set_xticks(positions)
    comparison.grid(True, alpha=0.25)
    comparison.legend(fontsize=8)

    fig.suptitle("Fixed-action condition-swap experiment", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.png")
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    temporary.replace(path)
    return path


def run_condition_swap(
    *,
    cache_dir: pathlib.Path,
    decoder_checkpoint: pathlib.Path,
    output_dir: pathlib.Path,
    split: str,
    split_json: pathlib.Path | None,
    num_actions: int,
    num_conditions: int,
    anchor_indices: Sequence[int] | None,
    seed: int,
    reverse_steps: int | None,
    reverse_solver: SourceSolver | None,
    decoder_steps: int | None,
    decoder_solver: FlowSolver | None,
    latent_rms_epsilon: float,
    reconstruction_degradation_epsilon: float,
    null_ratio_epsilon: float,
    source_checkpoint_dir: pathlib.Path | None,
    dataset_root: pathlib.Path | None,
    allow_download: bool,
) -> dict[str, Any]:
    if split not in ("train", "val", "test", "all"):
        raise ValueError(f"Unsupported split: {split!r}.")
    if null_ratio_epsilon < 0:
        raise ValueError(
            f"null_ratio_epsilon must be non-negative, got {null_ratio_epsilon}."
        )

    pairs = CachedPairs(cache_dir)
    decoder, decoder_metadata = load_checkpoint(decoder_checkpoint)
    extra = decoder_metadata.get("extra_metadata", {})
    expected_digest = extra.get("cache_records_sha256")
    actual_digest = str(pairs.manifest["records_sha256"])
    if expected_digest is not None and str(expected_digest) != actual_digest:
        raise ValueError(
            "Decoder checkpoint was trained on a different cache: "
            f"records_sha256 {expected_digest!r} != {actual_digest!r}."
        )
    expected_shape = (
        int(pairs.manifest["action_horizon"]),
        int(pairs.manifest["action_dim"]),
    )
    decoder_shape = (decoder.config.action_horizon, decoder.config.action_dim)
    if decoder_shape != expected_shape:
        raise ValueError(
            f"Decoder action shape {decoder_shape} does not match cache {expected_shape}."
        )

    if split_json is None:
        split_json = decoder_checkpoint.parent / "split.json"
    candidates = _read_split_indices(pairs, split=split, split_json=split_json)
    anchors, conditions = select_experiment_indices(
        candidates,
        np.asarray(pairs.arrays["episode_index"]),
        num_actions=num_actions,
        num_conditions=num_conditions,
        seed=seed,
        anchor_indices=anchor_indices,
    )

    configuration = pairs.manifest.get("configuration", {})
    if not isinstance(configuration, Mapping):
        raise ValueError("Cache manifest configuration must be an object.")
    selected_reverse_steps = int(
        reverse_steps if reverse_steps is not None else configuration.get("reverse_steps", 20)
    )
    selected_reverse_solver = str(
        reverse_solver
        if reverse_solver is not None
        else configuration.get("reverse_solver", "euler")
    )
    if selected_reverse_solver not in ("euler", "fireflow", "slerpflow"):
        raise ValueError(f"Unsupported source reverse solver: {selected_reverse_solver!r}.")
    selected_decoder_steps = int(
        decoder_steps if decoder_steps is not None else extra.get("validation_steps", 10)
    )
    selected_decoder_solver = str(
        decoder_solver if decoder_solver is not None else extra.get("solver", "euler")
    )
    if selected_decoder_solver not in ("euler", "fireflow"):
        raise ValueError(f"Unsupported decoder solver: {selected_decoder_solver!r}.")
    if selected_reverse_steps <= 0 or selected_decoder_steps <= 0:
        raise ValueError("reverse_steps and decoder_steps must be positive.")

    print(f"jax_devices={jax.devices()}", flush=True)
    print(
        f"loading source SmolVLA and {len(np.unique(conditions))} condition observations",
        flush=True,
    )
    source_model = _load_source_model(
        pairs,
        source_checkpoint_dir=source_checkpoint_dir,
        dataset_root=dataset_root,
        allow_download=allow_download,
    )
    if (source_model.action_horizon, source_model.action_dim) != expected_shape:
        raise ValueError(
            "Source SmolVLA action shape does not match the cache/decoder: "
            f"{(source_model.action_horizon, source_model.action_dim)} != {expected_shape}."
        )
    observations = _load_condition_observations(source_model, pairs, conditions)

    fixed_actions = np.asarray(pairs.arrays["target"][anchors], dtype=np.float32)
    latent_batches: list[np.ndarray] = []
    reconstructed_batches: list[np.ndarray] = []
    decoder_jvp_batches: list[np.ndarray] = []
    for anchor_slot, (action, condition_row) in enumerate(
        zip(fixed_actions, conditions), start=1
    ):
        observation_batch = stack_observations(
            [observations[int(index)] for index in condition_row]
        )
        action_batch = jnp.repeat(
            jnp.asarray(action, dtype=jnp.float32)[None, ...],
            repeats=len(condition_row),
            axis=0,
        )
        latents = reverse_integrate_actions(
            source_model,
            observation_batch,
            action_batch,
            num_steps=selected_reverse_steps,
            solver=selected_reverse_solver,  # type: ignore[arg-type]
        )
        reconstructed = decode_actions(
            decoder,
            latents,
            num_steps=selected_decoder_steps,
            solver=selected_decoder_solver,  # type: ignore[arg-type]
        )
        latent_directions = latents[1:] - latents[0]
        base_latents = jnp.broadcast_to(latents[0], latent_directions.shape)
        swapped_jvps = decoder_directional_derivatives(
            decoder,
            base_latents,
            latent_directions,
            num_steps=selected_decoder_steps,
            solver=selected_decoder_solver,  # type: ignore[arg-type]
        )
        decoder_jvps = jnp.concatenate(
            [jnp.zeros_like(latents[:1]), swapped_jvps], axis=0
        )
        latents_np, reconstructed_np, decoder_jvps_np = jax.device_get(
            (latents, reconstructed, decoder_jvps)
        )
        latent_batches.append(np.asarray(latents_np, dtype=np.float32))
        reconstructed_batches.append(np.asarray(reconstructed_np, dtype=np.float32))
        decoder_jvp_batches.append(np.asarray(decoder_jvps_np, dtype=np.float32))
        print(
            f"encoded/decoded fixed actions {anchor_slot}/{len(anchors)}",
            flush=True,
        )

    all_latents = np.stack(latent_batches, axis=0)
    all_reconstructed = np.stack(reconstructed_batches, axis=0)
    all_decoder_jvps = np.stack(decoder_jvp_batches, axis=0)
    episode_array = np.asarray(pairs.arrays["episode_index"])
    dataset_array = np.asarray(pairs.arrays["dataset_index"])

    reconstruction_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    original_encoder_cache_delta_l2: list[float] = []
    original_encoder_cache_delta_rms: list[float] = []
    for anchor_slot, anchor_cache_index in enumerate(anchors):
        action = fixed_actions[anchor_slot]
        original_latent = all_latents[anchor_slot, 0]
        cached_latent = np.asarray(
            pairs.arrays["x_base"][anchor_cache_index], dtype=np.float32
        )
        cache_delta = original_latent - cached_latent
        cache_delta_l2 = float(np.linalg.norm(cache_delta.reshape(-1)))
        cache_delta_rms = float(
            np.sqrt(np.mean(np.square(cache_delta), dtype=np.float64))
        )
        original_encoder_cache_delta_l2.append(cache_delta_l2)
        original_encoder_cache_delta_rms.append(cache_delta_rms)
        for condition_slot, condition_cache_index in enumerate(conditions[anchor_slot]):
            difference = all_reconstructed[anchor_slot, condition_slot] - action
            squared_l2 = float(np.sum(np.square(difference), dtype=np.float64))
            mse = float(np.mean(np.square(difference), dtype=np.float64))
            mae = float(np.mean(np.abs(difference), dtype=np.float64))
            latent_delta = all_latents[anchor_slot, condition_slot] - original_latent
            latent_delta_l2 = float(np.linalg.norm(latent_delta.reshape(-1)))
            decoder_delta = (
                all_reconstructed[anchor_slot, condition_slot]
                - all_reconstructed[anchor_slot, 0]
            )
            decoder_delta_l2 = float(np.linalg.norm(decoder_delta.reshape(-1)))
            decoder_jvp = all_decoder_jvps[anchor_slot, condition_slot]
            decoder_jvp_l2 = float(np.linalg.norm(decoder_jvp.reshape(-1)))
            if condition_slot == 0:
                r_null: float | str = ""
                finite_difference_gain: float | str = ""
                linearization_relative_error: float | str = ""
                jacobian_null_like: bool | str = ""
            else:
                r_null = decoder_jvp_l2 / latent_delta_l2
                finite_difference_gain = decoder_delta_l2 / latent_delta_l2
                linearization_relative_error = float(
                    np.linalg.norm((decoder_delta - decoder_jvp).reshape(-1))
                    / max(decoder_delta_l2, 1e-12)
                )
                jacobian_null_like = r_null <= null_ratio_epsilon
            reconstruction_rows.append(
                {
                    "anchor_slot": anchor_slot,
                    "condition_slot": condition_slot,
                    "is_original_condition": condition_slot == 0,
                    "anchor_cache_index": int(anchor_cache_index),
                    "anchor_dataset_index": int(dataset_array[anchor_cache_index]),
                    "anchor_episode_index": int(episode_array[anchor_cache_index]),
                    "condition_cache_index": int(condition_cache_index),
                    "condition_dataset_index": int(dataset_array[condition_cache_index]),
                    "condition_episode_index": int(episode_array[condition_cache_index]),
                    "reconstruction_squared_l2_R_ij": squared_l2,
                    "reconstruction_l2": float(np.sqrt(squared_l2)),
                    "reconstruction_mse": mse,
                    "reconstruction_rmse": float(np.sqrt(mse)),
                    "reconstruction_mae": mae,
                    "latent_norm_l2": float(
                        np.linalg.norm(all_latents[anchor_slot, condition_slot].reshape(-1))
                    ),
                    "delta_z_from_original_l2": float(
                        latent_delta_l2
                    ),
                    "delta_z_from_original_rms": float(
                        np.sqrt(np.mean(np.square(latent_delta), dtype=np.float64))
                    ),
                    "original_z_vs_cached_x_base_l2": cache_delta_l2,
                    "original_z_vs_cached_x_base_rms": cache_delta_rms,
                    "decoder_jvp_l2": decoder_jvp_l2,
                    "r_null": r_null,
                    "jacobian_null_like": jacobian_null_like,
                    "decoder_delta_from_original_l2": decoder_delta_l2,
                    "finite_difference_gain": finite_difference_gain,
                    "linearization_relative_error": linearization_relative_error,
                }
            )

        for condition_j in range(conditions.shape[1]):
            for condition_k in range(condition_j + 1, conditions.shape[1]):
                delta = (
                    all_latents[anchor_slot, condition_j]
                    - all_latents[anchor_slot, condition_k]
                )
                cache_j = int(conditions[anchor_slot, condition_j])
                cache_k = int(conditions[anchor_slot, condition_k])
                pairwise_rows.append(
                    {
                        "anchor_slot": anchor_slot,
                        "anchor_cache_index": int(anchor_cache_index),
                        "condition_j_slot": condition_j,
                        "condition_k_slot": condition_k,
                        "condition_j_cache_index": cache_j,
                        "condition_k_cache_index": cache_k,
                        "condition_j_episode_index": int(episode_array[cache_j]),
                        "condition_k_episode_index": int(episode_array[cache_k]),
                        "delta_z_l2": float(np.linalg.norm(delta.reshape(-1))),
                        "delta_z_rms": float(
                            np.sqrt(np.mean(np.square(delta), dtype=np.float64))
                        ),
                    }
                )

    original_rows = [row for row in reconstruction_rows if row["is_original_condition"]]
    swapped_rows = [row for row in reconstruction_rows if not row["is_original_condition"]]
    original_mse = np.asarray([row["reconstruction_mse"] for row in original_rows])
    swapped_mse = np.asarray([row["reconstruction_mse"] for row in swapped_rows])
    original_r = np.asarray(
        [row["reconstruction_squared_l2_R_ij"] for row in original_rows]
    )
    swapped_r = np.asarray(
        [row["reconstruction_squared_l2_R_ij"] for row in swapped_rows]
    )
    own_delta_l2 = np.asarray(
        [row["delta_z_from_original_l2"] for row in swapped_rows]
    )
    own_delta_rms = np.asarray(
        [row["delta_z_from_original_rms"] for row in swapped_rows]
    )
    pair_delta_l2 = np.asarray([row["delta_z_l2"] for row in pairwise_rows])
    pair_delta_rms = np.asarray([row["delta_z_rms"] for row in pairwise_rows])
    r_null_values = np.asarray([row["r_null"] for row in swapped_rows], dtype=np.float64)
    finite_difference_gains = np.asarray(
        [row["finite_difference_gain"] for row in swapped_rows], dtype=np.float64
    )
    linearization_errors = np.asarray(
        [row["linearization_relative_error"] for row in swapped_rows], dtype=np.float64
    )
    null_like_fraction = float(np.mean(r_null_values <= null_ratio_epsilon))
    degradation = float(np.mean(swapped_mse) - np.mean(original_mse))
    outcome, interpretation = classify_outcome(
        mean_latent_delta_rms=float(np.mean(own_delta_rms)),
        reconstruction_degradation_mse=degradation,
        latent_rms_epsilon=latent_rms_epsilon,
        reconstruction_degradation_epsilon=reconstruction_degradation_epsilon,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_csv = _write_csv(
        output_dir / "reconstruction_by_condition.csv", reconstruction_rows
    )
    pairwise_csv = _write_csv(output_dir / "latent_pairwise.csv", pairwise_rows)
    arrays_path = _write_arrays(
        output_dir / "condition_swap_arrays.npz",
        anchor_indices=anchors,
        condition_indices=conditions,
        actions=fixed_actions,
        latents=all_latents,
        reconstructions=all_reconstructed,
        decoder_jvps=all_decoder_jvps,
    )
    plot_path = _plot_results(
        output_dir / "condition_swap.png",
        reconstruction_rows=reconstruction_rows,
        num_actions=len(anchors),
    )
    null_space_plot_path = _plot_jacobian_null_space(
        output_dir / "jacobian_null_space.png",
        reconstruction_rows=reconstruction_rows,
        num_actions=len(anchors),
        null_ratio_epsilon=null_ratio_epsilon,
    )

    summary: dict[str, Any] = {
        "experiment": {
            "equations": {
                "encoding": "z_ij = E(a_i, o_j)",
                "decoding": "a_hat_ij = D(z_ij)",
                "R_ij": "||a_hat_ij - a_i||_2^2",
                "delta_z_ijk": "||z_ij - z_ik||_2",
                "r_null": "||J_D(z_ii) (z_ij - z_ii)||_2 / ||z_ij - z_ii||_2",
            },
            "action_source": "VLA target",
            "split": split,
            "seed": seed,
            "num_actions": int(len(anchors)),
            "num_conditions_per_action": int(conditions.shape[1]),
            "anchor_cache_indices": [int(index) for index in anchors],
            "condition_cache_indices": conditions.tolist(),
        },
        "configuration": {
            "cache_dir": str(cache_dir.resolve()),
            "decoder_checkpoint": str(decoder_checkpoint.resolve()),
            "split_json": str(split_json.resolve()) if split != "all" else None,
            "source_reverse_steps": selected_reverse_steps,
            "source_reverse_solver": selected_reverse_solver,
            "decoder_steps": selected_decoder_steps,
            "decoder_solver": selected_decoder_solver,
            "jacobian_linearization_point": "z_ii = E(a_i, o_i)",
        },
        "metrics": {
            "original_condition_R_squared_l2": _describe(original_r),
            "swapped_condition_R_squared_l2": _describe(swapped_r),
            "original_condition_reconstruction_mse": _describe(original_mse),
            "swapped_condition_reconstruction_mse": _describe(swapped_mse),
            "reconstruction_degradation_mse": degradation,
            "delta_z_original_to_swapped_l2": _describe(own_delta_l2),
            "delta_z_original_to_swapped_rms": _describe(own_delta_rms),
            "delta_z_all_condition_pairs_l2": _describe(pair_delta_l2),
            "delta_z_all_condition_pairs_rms": _describe(pair_delta_rms),
        },
        "jacobian_null_space": {
            "r_null": _describe(r_null_values),
            "null_ratio_epsilon": null_ratio_epsilon,
            "fraction_at_or_below_threshold": null_like_fraction,
            "finite_difference_gain": _describe(finite_difference_gains),
            "linearization_relative_error": _describe(linearization_errors),
            "interpretation": (
                "Small r_null means the local decoder Jacobian is insensitive to the "
                "observation-induced latent direction. The finite-difference gain and "
                "linearization error should also be checked because JVP is local."
            ),
        },
        "sanity_check": {
            "description": (
                "Recomputed E(a_i, o_i) should match the cache x_base used during "
                "decoder training."
            ),
            "available": True,
            "original_z_vs_cached_x_base_l2": _describe(
                np.asarray(original_encoder_cache_delta_l2)
            ),
            "original_z_vs_cached_x_base_rms": _describe(
                np.asarray(original_encoder_cache_delta_rms)
            ),
        },
        "outcome": {
            "label": outcome,
            "interpretation": interpretation,
            "latent_rms_epsilon": latent_rms_epsilon,
            "reconstruction_degradation_epsilon": reconstruction_degradation_epsilon,
            "note": (
                "The automatic label is threshold-based. Use the CSVs and scatter plot to "
                "inspect per-action heterogeneity before drawing a final conclusion."
            ),
        },
        "outputs": {
            "reconstruction_csv": str(reconstruction_csv),
            "latent_pairwise_csv": str(pairwise_csv),
            "arrays": str(arrays_path),
            "plot": str(plot_path),
            "jacobian_null_space_plot": str(null_space_plot_path),
        },
    }
    summary_path = output_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    print(
        f"outcome={outcome} mean_delta_z_rms={np.mean(own_delta_rms):.8f} "
        f"original_mse={np.mean(original_mse):.8f} "
        f"swapped_mse={np.mean(swapped_mse):.8f} degradation={degradation:.8f}",
        flush=True,
    )
    print(
        f"mean_r_null={np.mean(r_null_values):.8f} "
        f"median_r_null={np.median(r_null_values):.8f} "
        f"null_like_fraction={null_like_fraction:.3f} "
        f"threshold={null_ratio_epsilon:g}",
        flush=True,
    )
    print(f"summary={summary_path}", flush=True)
    print(f"plot={plot_path}", flush=True)
    print(f"jacobian_null_space_plot={null_space_plot_path}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fix action a_i, swap observation condition o_j during SmolVLA reverse "
            "encoding, and decode every z_ij with an already-trained decoder."
        )
    )
    parser.add_argument("--cache-dir", type=pathlib.Path, default=_DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--decoder-checkpoint",
        type=pathlib.Path,
        default=_DEFAULT_DECODER_CHECKPOINT,
        help="Directory containing checkpoint.json and params.npz.",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument(
        "--split-json",
        type=pathlib.Path,
        help="Defaults to split.json beside the decoder's best/last directory.",
    )
    parser.add_argument("--num-actions", type=int, default=8)
    parser.add_argument(
        "--num-conditions",
        type=int,
        default=6,
        help="Conditions per action, including its original observation as slot zero.",
    )
    parser.add_argument(
        "--anchor-cache-indices",
        type=int,
        nargs="+",
        help="Optional explicit cache rows for a_i; overrides --num-actions.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reverse-steps", type=int)
    parser.add_argument("--reverse-solver", choices=("euler", "fireflow", "slerpflow"))
    parser.add_argument("--decoder-steps", type=int)
    parser.add_argument("--decoder-solver", choices=("euler", "fireflow"))
    parser.add_argument(
        "--latent-rms-epsilon",
        type=float,
        default=1e-3,
        help="Threshold used to interpret Delta z approximately zero.",
    )
    parser.add_argument(
        "--reconstruction-degradation-epsilon",
        type=float,
        default=1e-2,
        help="Maximum swapped-minus-original MSE interpreted as no material degradation.",
    )
    parser.add_argument(
        "--null-ratio-epsilon",
        type=float,
        default=0.1,
        help=(
            "r_null threshold used to mark an observation-induced latent direction "
            "as approximately decoder-null (default: 0.1)."
        ),
    )
    parser.add_argument(
        "--source-checkpoint-dir",
        type=pathlib.Path,
        help="Override the SmolVLA checkpoint recorded in the cache manifest.",
    )
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        help="Override the dataset root recorded in the cache manifest.",
    )
    parser.add_argument("--allow-download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_condition_swap(
        cache_dir=args.cache_dir,
        decoder_checkpoint=args.decoder_checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        split_json=args.split_json,
        num_actions=args.num_actions,
        num_conditions=args.num_conditions,
        anchor_indices=args.anchor_cache_indices,
        seed=args.seed,
        reverse_steps=args.reverse_steps,
        reverse_solver=args.reverse_solver,
        decoder_steps=args.decoder_steps,
        decoder_solver=args.decoder_solver,
        latent_rms_epsilon=args.latent_rms_epsilon,
        reconstruction_degradation_epsilon=args.reconstruction_degradation_epsilon,
        null_ratio_epsilon=args.null_ratio_epsilon,
        source_checkpoint_dir=args.source_checkpoint_dir,
        dataset_root=args.dataset_root,
        allow_download=args.allow_download,
    )


if __name__ == "__main__":
    main()
