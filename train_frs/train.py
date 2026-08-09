"""Train tactile-conditioned flow decoder.

IMPORTANT: keep module-level imports free of JAX/Flax/data loaders. Mp spawn workers
re-import this file as ``__main__`` under ``CUDA_VISIBLE_DEVICES=""``; eager ``import jax``
there causes ``CUDA_ERROR_NO_DEVICE`` spam and fails the light-import guard.
"""

from __future__ import annotations

import argparse
import math
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal

LossMode = Literal["gt", "predicted", "gated"]


def checkpoint_selection_key(
    metrics: Mapping[str, Any],
    *,
    loss_mode: LossMode,
    low_gate_max_mse_pred: float,
    min_high_gate_gain: float,
    high_gate_rank_margin: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return a lower-is-better key for best-checkpoint selection.

    Gated runs first enforce preservation of the frozen VLA on low-gate
    samples, positive repair on high-gate samples, and the high-gate preference
    ``MSE(FRS, GT) + margin <= MSE(FRS, VLA)``. Among feasible models, larger
    high-gate gain wins. Non-gated or unstratified evaluations retain the legacy
    aggregate-MSE behavior.
    """

    aggregate_mse = float(metrics.get("val_mse", float("inf")))
    if loss_mode != "gated":
        return (0.0, aggregate_mse, 0.0, aggregate_mse)
    low_mse = float(
        metrics.get(
            "val_worst_dataset_mse_pred_low_w",
            metrics.get("val_mse_pred_low_w", float("nan")),
        )
    )
    high_gain = float(
        metrics.get(
            "val_min_dataset_gt_gain_high_w",
            metrics.get("val_gt_gain_high_w", float("nan")),
        )
    )
    if "val_worst_dataset_rank_violation_high_w" in metrics:
        rank_violation = float(metrics["val_worst_dataset_rank_violation_high_w"])
    else:
        high_mse_gt = float(metrics.get("val_mse_gt_high_w", float("nan")))
        high_mse_pred = float(metrics.get("val_mse_pred_high_w", float("nan")))
        rank_violation = max(
            0.0,
            high_mse_gt - high_mse_pred + float(high_gate_rank_margin),
        )
    if not all(math.isfinite(value) for value in (low_mse, high_gain, rank_violation)):
        return (2.0, aggregate_mse, 0.0, aggregate_mse)
    low_violation = max(0.0, low_mse - float(low_gate_max_mse_pred))
    gain_violation = max(0.0, float(min_high_gate_gain) - high_gain)
    total_violation = low_violation + gain_violation + rank_violation
    if total_violation == 0.0:
        return (0.0, -high_gain, low_mse, aggregate_mse)
    return (1.0, total_violation, -high_gain, aggregate_mse)


def _existing_run_artifacts(output_dir: pathlib.Path) -> tuple[pathlib.Path, ...]:
    candidates = (
        output_dir / "history.csv",
        output_dir / "best" / "checkpoint.json",
        output_dir / "last" / "checkpoint.json",
    )
    return tuple(path for path in candidates if path.exists())


def _validate_resume_cache(
    resume_metadata: Mapping[str, Any],
    cache_manifest: Mapping[str, Any],
) -> None:
    extra = resume_metadata.get("extra_metadata")
    if not isinstance(extra, Mapping):
        raise ValueError("resume checkpoint is missing cache provenance metadata")
    expected_digest = cache_manifest.get("records_sha256")
    if extra.get("cache_records_sha256") != expected_digest:
        raise ValueError(
            "resume checkpoint/cache record digest mismatch: "
            f"{extra.get('cache_records_sha256')!r} != {expected_digest!r}"
        )
    expected_configuration = cache_manifest.get("configuration")
    if extra.get("cache_configuration") != expected_configuration:
        raise ValueError(
            "resume checkpoint was trained with a different action-cache configuration"
        )


def _resolve_resume_dir(
    *,
    output_dir: pathlib.Path,
    resume: bool,
    resume_from: pathlib.Path | None,
) -> pathlib.Path | None:
    if resume_from is not None:
        return resume_from
    if resume:
        return output_dir / "last"
    return None


def train_decoder(
    *,
    cache_dir: pathlib.Path | None,
    tactile_encoder_dir: pathlib.Path,
    output_dir: pathlib.Path,
    dataset_repo_id: str | None,
    dataset_root: pathlib.Path | None,
    tactile_window_divisor: int,
    history_stride: int,
    loss_mode: LossMode,
    gate_tau: float,
    gate_temperature: float,
    gate_lambda: float,
    aux_decode_weight: float,
    aux_decode_steps: int,
    rank_weight: float,
    rank_margin: float,
    repair_weight: float,
    repair_margin: float,
    model_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float | None,
    warmup_epochs: int,
    lr_reference_dim: int | None,
    min_learning_rate_ratio: float,
    cosine_decay: bool,
    batch_size: int,
    epochs: int,
    validation_steps: int,
    eval_every: int,
    seed: int,
    write_plots: bool,
    num_workers: int,
    prefetch_batches: int,
    load_threads: int,
    pipeline_prefetch: int,
    image_cache_size: int,
    encode_batch_size: int,
    resume: bool = False,
    resume_from: pathlib.Path | None = None,
    cache_dirs: Sequence[pathlib.Path] | None = None,
    dataset_sources: Sequence[Mapping[str, Any]] | None = None,
    tactile_embedding_cache_root: pathlib.Path | None = None,
    tactile_keys: Sequence[str] | None = None,
    tactile_embedding_dim: int = 512,
    tactile_image_size: int = 224,
    tactile_num_tokens: int = 4,
    best_low_gate_max_mse_pred: float = 0.01,
    best_min_high_gate_gain: float = 0.0,
) -> None:
    import csv
    import json

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    from train_frs.utils.checkpoint import (
        CHECKPOINT_NAME,
        load_checkpoint,
        load_optimizer_state,
        restore_optimizer_state,
        save_checkpoint,
    )
    from train_frs.utils.data import (
        CachedTactileEmbeddingBatches,
        TactileConditionedBatches,
        gate_weights_from_change,
        resolve_tactile_window,
    )
    from train_frs.utils.history_plot import plot_training_history
    from train_frs.utils.metrics import evaluate_split
    from train_frs.utils.model import (
        DEFAULT_GRU_HIDDEN_DIM,
        DecoderConfig,
        TactileConditionedFlowDecoder,
        make_optimizer,
        resolve_peak_learning_rate,
        train_step,
    )
    from utils.cache import CachedPairs, MultiCachedPairs

    history_fields = [
        "epoch",
        "train_loss_total",
        "train_loss_gt_fm",
        "train_loss_vla_fm",
        "train_loss_decode",
        "train_loss_rank",
        "train_loss_repair",
        # Backward-compatible alias for readers of pre-refactor histories.
        "train_flow_loss",
        "val_flow_loss",
        "val_mse",
        "val_rmse",
        "val_mae",
        "val_flow_loss_gt",
        "val_mse_gt",
        "val_rmse_gt",
        "val_mae_gt",
        "val_flow_loss_pred",
        "val_mse_pred",
        "val_rmse_pred",
        "val_mae_pred",
        "val_mse_vla_gt",
        "val_gt_gain",
        "val_relative_gt_error",
        "eval_target",
        "val_mse_gt_high_w",
        "val_mse_gt_low_w",
        "val_mse_pred_high_w",
        "val_mse_pred_low_w",
        "val_mse_vla_gt_high_w",
        "val_mse_vla_gt_low_w",
        "val_gt_gain_high_w",
        "val_gt_gain_low_w",
        "val_relative_gt_error_high_w",
        "val_relative_gt_error_low_w",
        "val_rank_penalty_high_w",
        "val_rank_penalty_low_w",
        "val_rank_satisfied_high_frac",
        "val_rank_satisfied_low_frac",
        "val_repair_penalty_high_w",
        "val_repair_satisfied_high_frac",
        "val_gate_w",
        "val_gate_active_frac",
        "val_gate_w_high_mean",
        "val_gate_w_low_mean",
        "val_gate_w_p10",
        "val_gate_w_p25",
        "val_gate_w_p50",
        "val_gate_w_p75",
        "val_gate_w_p90",
        "val_tactile_change",
        "val_tactile_change_high_mean",
        "val_tactile_change_low_mean",
        "val_tactile_change_p10",
        "val_tactile_change_p25",
        "val_tactile_change_p50",
        "val_tactile_change_p75",
        "val_tactile_change_p90",
        "val_n_high_w",
        "val_n_low_w",
        "val_worst_dataset_mse_pred_low_w",
        "val_min_dataset_gt_gain_high_w",
        "val_worst_dataset_rank_violation_high_w",
        "checkpoint_selection_key",
        "checkpoint_selection_feasible",
    ]

    def _blank_history_row(epoch: int, **filled: float | int | str) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = dict.fromkeys(history_fields, "")
        row["epoch"] = epoch
        row.update(filled)
        return row

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if not 0.0 <= min_learning_rate_ratio <= 1.0:
        raise ValueError("min_learning_rate_ratio must be in [0, 1].")
    if loss_mode not in ("gt", "predicted", "gated"):
        raise ValueError(f"loss_mode must be 'gt', 'predicted', or 'gated', got {loss_mode!r}.")
    eval_target = "predicted" if loss_mode == "predicted" else "gt"
    if gate_temperature <= 0:
        raise ValueError(f"gate_temperature must be positive, got {gate_temperature}.")
    if gate_lambda < 0:
        raise ValueError(f"gate_lambda must be non-negative, got {gate_lambda}.")
    if rank_weight < 0:
        raise ValueError(f"rank_weight must be non-negative, got {rank_weight}.")
    if rank_margin < 0:
        raise ValueError(f"rank_margin must be non-negative, got {rank_margin}.")
    if repair_weight < 0:
        raise ValueError(f"repair_weight must be non-negative, got {repair_weight}.")
    if repair_margin < 0:
        raise ValueError(f"repair_margin must be non-negative, got {repair_margin}.")
    if best_low_gate_max_mse_pred < 0:
        raise ValueError("best_low_gate_max_mse_pred must be non-negative.")
    if loss_mode != "gated" and (rank_weight != 0 or repair_weight != 0):
        raise ValueError(
            "rank_weight and repair_weight are only supported with loss_mode='gated'."
        )
    if eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {eval_every}.")
    if tactile_num_tokens <= 0:
        raise ValueError(f"tactile_num_tokens must be positive, got {tactile_num_tokens}.")

    resume_dir = _resolve_resume_dir(output_dir=output_dir, resume=resume, resume_from=resume_from)
    if resume_dir is None:
        existing_artifacts = _existing_run_artifacts(output_dir)
        if existing_artifacts:
            raise FileExistsError(
                "refusing to start a fresh run in an existing FRS output directory; "
                f"found {list(existing_artifacts)}. Choose a new output or enable resume."
            )
    start_epoch = 1
    resume_metadata: dict | None = None
    resumed_opt_state = None
    resumed_opt_step: int | None = None
    if resume_dir is not None:
        if not (resume_dir / CHECKPOINT_NAME).exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_dir}")
        model, resume_metadata = load_checkpoint(resume_dir)
        resumed_opt_state, resumed_opt_step = load_optimizer_state(resume_dir)
        start_epoch = int(resume_metadata["epoch"]) + 1
        print(
            f"resuming from {resume_dir} epoch={resume_metadata['epoch']} "
            f"next_epoch={start_epoch} has_opt_state={resumed_opt_state is not None}",
            flush=True,
        )
        if start_epoch > epochs:
            print(
                f"already finished: last epoch {resume_metadata['epoch']} >= --epochs {epochs}",
                flush=True,
            )
            return

    print(f"jax_devices={jax.devices()}", flush=True)
    if not any(d.platform == "gpu" for d in jax.devices()):
        print(
            "WARNING: no JAX GPU device visible; ResNet encode + training will run on CPU "
            "(very slow). Check nvidia-smi / CUDA_VISIBLE_DEVICES.",
            flush=True,
        )

    use_cached_embeddings = cache_dirs is not None
    if use_cached_embeddings:
        if not cache_dirs:
            raise ValueError("cache_dirs must be non-empty when provided")
        if dataset_sources is None or len(dataset_sources) != len(cache_dirs):
            raise ValueError("dataset_sources must have one entry per cache_dirs entry")
        if tactile_embedding_cache_root is None:
            raise ValueError("tactile_embedding_cache_root is required for multi-source FRS")
        if not tactile_keys:
            raise ValueError("tactile_keys is required for multi-source FRS")
        source_names = [str(source["repo_id"]) for source in dataset_sources]
        pairs = MultiCachedPairs(cache_dirs, source_names=source_names)
    else:
        if cache_dir is None:
            raise ValueError("cache_dir is required when cache_dirs is not provided")
        pairs = CachedPairs(cache_dir)
    if resume_metadata is not None:
        _validate_resume_cache(resume_metadata, pairs.manifest)
        resume_extra = resume_metadata.get("extra_metadata") or {}
        stored_rank_weight = float(resume_extra.get("rank_weight", 0.0))
        stored_rank_margin = float(resume_extra.get("rank_margin", 0.0))
        stored_repair_weight = float(resume_extra.get("repair_weight", 0.0))
        stored_repair_margin = float(resume_extra.get("repair_margin", 0.0))
        stored_weighting_version = int(resume_extra.get("loss_weighting_version", 1))
        stored_low_gate_limit = float(
            resume_extra.get("best_low_gate_max_mse_pred", best_low_gate_max_mse_pred)
        )
        stored_min_high_gain = float(
            resume_extra.get("best_min_high_gate_gain", best_min_high_gate_gain)
        )
        if (
            stored_weighting_version != 3
            or stored_rank_weight != rank_weight
            or stored_rank_margin != rank_margin
            or stored_repair_weight != repair_weight
            or stored_repair_margin != repair_margin
            or stored_low_gate_limit != best_low_gate_max_mse_pred
            or stored_min_high_gain != best_min_high_gate_gain
        ):
            raise ValueError(
                "Resume checkpoint constraint objective differs from this run: "
                "checkpoint="
                f"(rank_weight={stored_rank_weight:g}, rank_margin={stored_rank_margin:g}, "
                f"repair_weight={stored_repair_weight:g}, "
                f"repair_margin={stored_repair_margin:g}, weighting_v={stored_weighting_version}, "
                f"low_gate_limit={stored_low_gate_limit:g}, min_high_gain={stored_min_high_gain:g}) "
                "requested="
                f"(rank_weight={rank_weight:g}, rank_margin={rank_margin:g}, "
                f"repair_weight={repair_weight:g}, repair_margin={repair_margin:g}, weighting_v=3, "
                f"low_gate_limit={best_low_gate_max_mse_pred:g}, "
                f"min_high_gain={best_min_high_gate_gain:g}). "
                "Start a fresh run in a new frs_training.output directory."
            )
    action_horizon = int(pairs.manifest["action_horizon"])
    tactile_window = resolve_tactile_window(
        action_horizon=action_horizon,
        window_divisor=tactile_window_divisor,
    )
    if use_cached_embeddings:
        assert dataset_sources is not None
        assert tactile_embedding_cache_root is not None
        assert tactile_keys is not None
        conditioner = CachedTactileEmbeddingBatches(
            pairs,
            sources=dataset_sources,
            tactile_cache_root=tactile_embedding_cache_root,
            tactile_encoder_dir=tactile_encoder_dir,
            tactile_keys=tactile_keys,
            tactile_window=tactile_window,
            history_stride=history_stride,
            embedding_dim=tactile_embedding_dim,
            image_size=tactile_image_size,
            build_episode_baselines=(loss_mode == "gated"),
        )
    else:
        conditioner = TactileConditionedBatches(
            pairs,
            tactile_encoder_dir=tactile_encoder_dir,
            tactile_window=tactile_window,
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
            history_stride=history_stride,
            build_episode_baselines=(loss_mode == "gated"),
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
            load_threads=load_threads,
            pipeline_prefetch=pipeline_prefetch,
            image_cache_size=image_cache_size,
            encode_batch_size=encode_batch_size,
        )
    decoder_config = DecoderConfig(
        action_dim=int(pairs.manifest["action_dim"]),
        action_horizon=action_horizon,
        tactile_window=tactile_window,
        gru_hidden_dim=DEFAULT_GRU_HIDDEN_DIM,
        resnet_embedding_dim=conditioner.resnet_embedding_dim,
        model_dim=model_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_tactile_tokens=tactile_num_tokens,
        gate_conditioning=(loss_mode == "gated"),
    )
    if resume_metadata is None:
        model = TactileConditionedFlowDecoder(decoder_config, rngs=nnx.Rngs(seed))
    else:
        ckpt_config = DecoderConfig(**resume_metadata["decoder_config"])
        if dataclasses_asdict_mismatch := _config_diff(ckpt_config, decoder_config):
            print(
                "warning: CLI decoder config differs from resume checkpoint; "
                f"keeping checkpoint weights. diffs={dataclasses_asdict_mismatch}",
                flush=True,
            )
        # ``model`` already loaded above.
        if loss_mode == "gated" and not model.config.gate_conditioning:
            raise ValueError(
                "This checkpoint predates explicit gate conditioning. Start a fresh run in a "
                "new frs_training.output directory instead of resuming it."
            )
    train_samples = len(pairs.indices("train"))
    steps_per_epoch = max(1, (train_samples + batch_size - 1) // batch_size)
    warmup_steps = min(warmup_epochs, epochs) * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    peak_learning_rate = resolve_peak_learning_rate(
        learning_rate,
        model_dim=int(model.config.model_dim),
        lr_reference_dim=lr_reference_dim,
    )
    optimizer = make_optimizer(
        model,
        learning_rate=peak_learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_learning_rate_ratio=min_learning_rate_ratio,
        cosine_decay=cosine_decay,
    )
    if resumed_opt_state is not None:
        restore_optimizer_state(optimizer, opt_state=resumed_opt_state, step=resumed_opt_step)
    elif resume_dir is not None:
        print(
            "warning: optimizer state missing in checkpoint; reinitialized Adam state.",
            flush=True,
        )
    if lr_reference_dim is not None:
        print(
            f"learning_rate={learning_rate:g} scaled by sqrt({lr_reference_dim}/{model.config.model_dim}) "
            f"-> peak={peak_learning_rate:g}"
        )
    else:
        print(f"learning_rate peak={peak_learning_rate:g}")
    print(
        f"tactile_window={tactile_window} "
        f"(action_horizon={action_horizon} / divisor={tactile_window_divisor}) "
        f"gru_hidden_dim={DEFAULT_GRU_HIDDEN_DIM} resnet_dim={conditioner.resnet_embedding_dim} "
        f"(frozen ResNet + trainable shared GRU)"
    )
    if use_cached_embeddings:
        print(
            f"dataloader=precomputed_tactile_embeddings sources={len(dataset_sources or ())} "
            f"cache_root={tactile_embedding_cache_root} "
            f"eval_every={eval_every} start_epoch={start_epoch} epochs={epochs}"
        )
    else:
        print(
            f"dataloader=num_workers={num_workers} prefetch_batches={prefetch_batches} "
            f"load_threads={load_threads} pipeline_prefetch={pipeline_prefetch} "
            f"image_cache_size={image_cache_size} encode_batch_size={encode_batch_size} "
            f"eval_every={eval_every} start_epoch={start_epoch} epochs={epochs}"
        )
    if aux_decode_weight < 0:
        raise ValueError(f"aux_decode_weight must be >= 0, got {aux_decode_weight}.")
    if aux_decode_steps <= 0:
        raise ValueError(f"aux_decode_steps must be positive, got {aux_decode_steps}.")
    if loss_mode == "gt":
        print(
            "loss_mode=gt L=FM(gt)+aux*MSE(decode,gt) "
            f"(aux={aux_decode_weight:g}, decode_steps={aux_decode_steps}; "
            "primary eval vs gt; also log vs predicted)"
        )
    elif loss_mode == "predicted":
        print(
            "loss_mode=predicted (train/eval primary target=predicted_actions; also log vs gt; "
            "no aux decode MSE)"
        )
    else:
        print(
            "loss_mode=gated L=w*FM(gt)+lambda*(1-w)*FM(pred) "
            "+aux*[w*MSE(decode,gt)+(1-w)*MSE(decode,pred)] "
            "+rank_weight*[w*rank_gt+(1-w)*rank_vla] "
            "+repair_weight*w*absolute_repair_loss "
            f"tau={gate_tau:g} T={gate_temperature:g} lambda={gate_lambda:g} "
            f"aux={aux_decode_weight:g} decode_steps={aux_decode_steps} "
            f"rank_weight={rank_weight:g} rank_margin={rank_margin:g} "
            f"repair_weight={repair_weight:g} repair_margin={repair_margin:g} "
            f"(primary eval=gt; also log vs predicted)"
        )
    if cosine_decay:
        print(
            f"lr_schedule=warmup({warmup_steps} steps)+cosine "
            f"min_ratio={min_learning_rate_ratio:g} total_steps={total_steps}"
        )
    elif warmup_steps > 0:
        print(f"lr_schedule=warmup({warmup_steps} steps)+constant total_steps={total_steps}")

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    plot_path = output_dir / "training_curves.png"
    best_key = (float("inf"),) * 4
    best_path = output_dir / "best" / CHECKPOINT_NAME
    if resume_dir is not None and best_path.exists():
        with best_path.open(encoding="utf-8") as file:
            best_meta = json.load(file)
        _validate_resume_cache(best_meta, pairs.manifest)
        best_key = checkpoint_selection_key(
            best_meta.get("metrics", {}),
            loss_mode=loss_mode,
            low_gate_max_mse_pred=best_low_gate_max_mse_pred,
            min_high_gate_gain=best_min_high_gate_gain,
            high_gate_rank_margin=rank_margin,
        )
    base_key = jax.random.key(seed)
    history_exists = history_path.exists() and history_path.stat().st_size > 0
    history_mode = "a" if resume_dir is not None and history_exists else "w"

    def _refresh_training_plot(*, announce: bool = False) -> None:
        if not write_plots:
            return
        try:
            written = plot_training_history(history_path, output_path=plot_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"warning: could not refresh training plot: {exc}", flush=True)
            return
        if announce:
            print(f"plot={written}", flush=True)

    try:
        with history_path.open(history_mode, newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=history_fields)
            if history_mode == "w":
                writer.writeheader()

            for epoch in range(start_epoch, epochs + 1):
                losses: list[float] = []
                component_losses: dict[str, list[float]] = {
                    name: [] for name in ("gt_fm", "vla_fm", "decode", "rank", "repair")
                }
                weights: list[int] = []
                for batch_number, (indices, x_base_np, predicted_np, gt_action_np, tactile_seq) in enumerate(
                    conditioner.batches("train", batch_size=batch_size, shuffle=True, seed=seed + epoch)
                ):
                    step_key = jax.random.fold_in(base_key, epoch * 1_000_000 + batch_number)
                    batch_n = len(x_base_np)
                    if loss_mode == "gated":
                        current_tokens = np.asarray(tactile_seq[:, -1, :, :], dtype=np.float32)
                        change = conditioner.tactile_change_for_cache_indices(indices, current_tokens)
                        gate_w = gate_weights_from_change(
                            change, tau=gate_tau, temperature=gate_temperature
                        )
                        batch_gate_w = float(np.mean(gate_w))
                    else:
                        gate_w = np.ones((batch_n,), dtype=np.float32)
                        batch_gate_w = 1.0
                    loss, loss_components = train_step(
                        model,
                        optimizer,
                        jnp.asarray(x_base_np),
                        jnp.asarray(gt_action_np),
                        jnp.asarray(predicted_np),
                        tactile_seq,
                        jnp.asarray(gate_w),
                        step_key,
                        loss_mode=loss_mode,
                        gate_lambda=gate_lambda,
                        aux_decode_weight=aux_decode_weight,
                        aux_decode_steps=aux_decode_steps,
                        rank_weight=rank_weight,
                        rank_margin=rank_margin,
                        repair_weight=repair_weight,
                        repair_margin=repair_margin,
                    )
                    losses.append(float(jax.device_get(loss)))
                    for name in component_losses:
                        component_losses[name].append(
                            float(jax.device_get(loss_components[name]))
                        )
                    weights.append(batch_n)
                    if batch_number == 0 or (batch_number + 1) % 20 == 0:
                        extra = f" gate_w={batch_gate_w:.4f}" if loss_mode == "gated" else ""
                        print(
                            f"epoch={epoch}/{epochs} batch={batch_number + 1}/{steps_per_epoch} "
                            f"loss_total={losses[-1]:.6f} "
                            f"gt_fm={component_losses['gt_fm'][-1]:.6f} "
                            f"vla_fm={component_losses['vla_fm'][-1]:.6f} "
                            f"decode={component_losses['decode'][-1]:.6f} "
                            f"rank={component_losses['rank'][-1]:.6f} "
                            f"repair={component_losses['repair'][-1]:.6f}{extra}",
                            flush=True,
                        )
                train_loss = float(np.average(losses, weights=weights))
                train_components = {
                    name: float(np.average(values, weights=weights))
                    for name, values in component_losses.items()
                }
                train_metrics: dict[str, float] = {
                    "train_loss_total": train_loss,
                    "train_loss_gt_fm": train_components["gt_fm"],
                    "train_loss_vla_fm": train_components["vla_fm"],
                    "train_loss_decode": train_components["decode"],
                    "train_loss_rank": train_components["rank"],
                    "train_loss_repair": train_components["repair"],
                    "train_flow_loss": train_loss,
                }
                run_eval = (epoch % eval_every == 0) or (epoch == epochs)
                checkpoint_extra = {
                    "cache_records_sha256": pairs.manifest["records_sha256"],
                    "cache_configuration": pairs.manifest["configuration"],
                    "tactile_encoder_dir": str(tactile_encoder_dir.resolve()),
                    "tactile_window_divisor": tactile_window_divisor,
                    "tactile_window": tactile_window,
                    "gru_hidden_dim": DEFAULT_GRU_HIDDEN_DIM,
                    "history_stride": history_stride,
                    "loss_mode": loss_mode,
                    "eval_target": eval_target,
                    "gate_tau": gate_tau,
                    "gate_temperature": gate_temperature,
                    "gate_lambda": gate_lambda,
                    "gate_conditioning": bool(model.config.gate_conditioning),
                    "aux_decode_weight": aux_decode_weight,
                    "aux_decode_steps": aux_decode_steps,
                    "rank_weight": rank_weight,
                    "rank_margin": rank_margin,
                    "repair_weight": repair_weight,
                    "repair_margin": repair_margin,
                    "loss_weighting_version": 3,
                    "validation_steps": validation_steps,
                    "validation_solver": "euler",
                    "best_low_gate_max_mse_pred": best_low_gate_max_mse_pred,
                    "best_min_high_gate_gain": best_min_high_gate_gain,
                    "eval_every": eval_every,
                }
                if run_eval:
                    validation = evaluate_split(
                        model,
                        conditioner,
                        split="val",
                        batch_size=batch_size,
                        num_steps=validation_steps,
                        keep_predictions=False,
                        target=eval_target,
                        gate_tau=gate_tau if loss_mode == "gated" else None,
                        gate_temperature=gate_temperature if loss_mode == "gated" else None,
                        rank_margin=rank_margin if loss_mode == "gated" else 0.0,
                        repair_margin=repair_margin if loss_mode == "gated" else 0.0,
                    )
                    metrics: dict[str, float | str | int] = {
                        **train_metrics,
                        "val_flow_loss": validation.flow_loss,
                        "val_mse": validation.mse,
                        "val_rmse": validation.rmse,
                        "val_mae": validation.mae,
                        "val_flow_loss_gt": validation.flow_loss_gt,
                        "val_mse_gt": validation.mse_gt,
                        "val_rmse_gt": validation.rmse_gt,
                        "val_mae_gt": validation.mae_gt,
                        "val_flow_loss_pred": validation.flow_loss_pred,
                        "val_mse_pred": validation.mse_pred,
                        "val_rmse_pred": validation.rmse_pred,
                        "val_mae_pred": validation.mae_pred,
                        "val_mse_vla_gt": validation.mse_vla_gt,
                        "val_gt_gain": validation.gt_gain,
                        "val_relative_gt_error": validation.relative_gt_error,
                        "eval_target": validation.target,
                    }
                    if validation.n_high_w is not None:
                        metrics.update(
                            {
                                "val_mse_gt_high_w": float(validation.mse_gt_high_w),
                                "val_mse_gt_low_w": float(validation.mse_gt_low_w),
                                "val_mse_pred_high_w": float(validation.mse_pred_high_w),
                                "val_mse_pred_low_w": float(validation.mse_pred_low_w),
                                "val_mse_vla_gt_high_w": float(validation.mse_vla_gt_high_w),
                                "val_mse_vla_gt_low_w": float(validation.mse_vla_gt_low_w),
                                "val_gt_gain_high_w": float(validation.gt_gain_high_w),
                                "val_gt_gain_low_w": float(validation.gt_gain_low_w),
                                "val_relative_gt_error_high_w": float(
                                    validation.relative_gt_error_high_w
                                ),
                                "val_relative_gt_error_low_w": float(
                                    validation.relative_gt_error_low_w
                                ),
                                "val_rank_penalty_high_w": float(
                                    validation.rank_penalty_high_w
                                ),
                                "val_rank_penalty_low_w": float(
                                    validation.rank_penalty_low_w
                                ),
                                "val_rank_satisfied_high_frac": float(
                                    validation.rank_satisfied_high_frac
                                ),
                                "val_rank_satisfied_low_frac": float(
                                    validation.rank_satisfied_low_frac
                                ),
                                "val_repair_penalty_high_w": float(
                                    validation.repair_penalty_high_w
                                ),
                                "val_repair_satisfied_high_frac": float(
                                    validation.repair_satisfied_high_frac
                                ),
                                "val_gate_w": float(validation.gate_w),
                                "val_gate_active_frac": float(validation.gate_active_frac),
                                "val_gate_w_high_mean": float(validation.gate_w_high_mean),
                                "val_gate_w_low_mean": float(validation.gate_w_low_mean),
                                "val_gate_w_p10": float(validation.gate_w_p10),
                                "val_gate_w_p25": float(validation.gate_w_p25),
                                "val_gate_w_p50": float(validation.gate_w_p50),
                                "val_gate_w_p75": float(validation.gate_w_p75),
                                "val_gate_w_p90": float(validation.gate_w_p90),
                                "val_tactile_change": float(validation.tactile_change),
                                "val_tactile_change_high_mean": float(
                                    validation.tactile_change_high_mean
                                ),
                                "val_tactile_change_low_mean": float(
                                    validation.tactile_change_low_mean
                                ),
                                "val_tactile_change_p10": float(validation.tactile_change_p10),
                                "val_tactile_change_p25": float(validation.tactile_change_p25),
                                "val_tactile_change_p50": float(validation.tactile_change_p50),
                                "val_tactile_change_p75": float(validation.tactile_change_p75),
                                "val_tactile_change_p90": float(validation.tactile_change_p90),
                                "val_n_high_w": int(validation.n_high_w),
                                "val_n_low_w": int(validation.n_low_w),
                            }
                        )
                    if (
                        isinstance(pairs, MultiCachedPairs)
                        and validation.sample_gate_w is not None
                    ):
                        source_indices, _ = pairs.source_and_local_indices(
                            validation.cache_indices
                        )
                        low_preservation: list[float] = []
                        high_gains: list[float] = []
                        high_rank_violations: list[float] = []
                        for source_index in range(len(pairs.sources)):
                            source_mask = source_indices == source_index
                            source_gate = validation.sample_gate_w[source_mask]
                            source_low = source_gate <= 0.5
                            source_high = source_gate > 0.5
                            if np.any(source_low):
                                low_preservation.append(
                                    float(
                                        np.mean(
                                            validation.sample_mse_pred[source_mask][source_low]
                                        )
                                    )
                                )
                            if np.any(source_high):
                                high_gains.append(
                                    float(
                                        np.mean(
                                            validation.sample_gt_gain[source_mask][source_high]
                                        )
                                    )
                                )
                                source_mse_gt_high = float(
                                    np.mean(
                                        validation.sample_mse_gt[source_mask][source_high]
                                    )
                                )
                                source_mse_pred_high = float(
                                    np.mean(
                                        validation.sample_mse_pred[source_mask][source_high]
                                    )
                                )
                                high_rank_violations.append(
                                    max(
                                        0.0,
                                        source_mse_gt_high
                                        - source_mse_pred_high
                                        + rank_margin,
                                    )
                                )
                        if low_preservation:
                            metrics["val_worst_dataset_mse_pred_low_w"] = max(
                                low_preservation
                            )
                        if high_gains:
                            metrics["val_min_dataset_gt_gain_high_w"] = min(high_gains)
                        if high_rank_violations:
                            metrics["val_worst_dataset_rank_violation_high_w"] = max(
                                high_rank_violations
                            )
                    selection_key = checkpoint_selection_key(
                        metrics,
                        loss_mode=loss_mode,
                        low_gate_max_mse_pred=best_low_gate_max_mse_pred,
                        min_high_gate_gain=best_min_high_gate_gain,
                        high_gate_rank_margin=rank_margin,
                    )
                    metrics["checkpoint_selection_key"] = ",".join(
                        f"{value:.12g}" for value in selection_key
                    )
                    metrics["checkpoint_selection_feasible"] = int(selection_key[0] == 0.0)
                    writer.writerow(_blank_history_row(epoch, **metrics))
                    history_file.flush()
                    _refresh_training_plot()
                    save_checkpoint(
                        output_dir / "last",
                        model,
                        epoch=epoch,
                        metrics=metrics,
                        extra_metadata=checkpoint_extra,
                        optimizer=optimizer,
                    )
                    if selection_key < best_key:
                        best_key = selection_key
                        save_checkpoint(
                            output_dir / "best",
                            model,
                            epoch=epoch,
                            metrics=metrics,
                            extra_metadata=checkpoint_extra,
                            optimizer=optimizer,
                        )
                    stratified_msg = ""
                    if validation.n_high_w is not None:
                        stratified_msg = (
                            f" mse_gt(w>0.5)={validation.mse_gt_high_w:.4f}"
                            f" mse_gt(w<=0.5)={validation.mse_gt_low_w:.4f}"
                            f" mse_pred(w>0.5)={validation.mse_pred_high_w:.4f}"
                            f" mse_pred(w<=0.5)={validation.mse_pred_low_w:.4f}"
                            f" vla_gt(w>0.5)={validation.mse_vla_gt_high_w:.4f}"
                            f" gain(w>0.5)={validation.gt_gain_high_w:.4f}"
                            f" rel_gt(w>0.5)={validation.relative_gt_error_high_w:.4f}"
                            f" rank_ok_hi={validation.rank_satisfied_high_frac:.3f}"
                            f" rank_ok_lo={validation.rank_satisfied_low_frac:.3f}"
                            f" repair_ok_hi={validation.repair_satisfied_high_frac:.3f}"
                            f" w_mean_hi={validation.gate_w_high_mean:.3f}"
                            f" w_mean_lo={validation.gate_w_low_mean:.3f}"
                            f" n_high={validation.n_high_w} n_low={validation.n_low_w}"
                        )
                    selection_msg = (
                        f" best_feasible={int(selection_key[0] == 0.0)}"
                        f" selection_key={metrics['checkpoint_selection_key']}"
                    )
                    print(
                        f"epoch={epoch}/{epochs} train_loss_total={train_loss:.8f} "
                        f"val_flow_loss={validation.flow_loss:.8f} "
                        f"val_mse={validation.mse:.8f} (target={validation.target}) "
                        f"val_mse_gt={validation.mse_gt:.8f} val_mse_pred={validation.mse_pred:.8f} "
                        f"vla_mse_gt={validation.mse_vla_gt:.8f} "
                        f"gt_gain={validation.gt_gain:.8f} "
                        f"relative_gt_error={validation.relative_gt_error:.4f}"
                        f"{stratified_msg}{selection_msg}",
                        flush=True,
                    )
                else:
                    metrics = dict(train_metrics)
                    writer.writerow(_blank_history_row(epoch, **metrics))
                    history_file.flush()
                    _refresh_training_plot()
                    save_checkpoint(
                        output_dir / "last",
                        model,
                        epoch=epoch,
                        metrics=metrics,
                        extra_metadata=checkpoint_extra,
                        optimizer=optimizer,
                    )
                    print(
                        f"epoch={epoch}/{epochs} train_loss_total={train_loss:.8f} (skip val)",
                        flush=True,
                    )

        print(f"best_checkpoint_selection_key={best_key}")
        print(f"checkpoints={output_dir}")
        _refresh_training_plot(announce=True)
    finally:
        conditioner.close()


def _config_diff(left: object, right: object) -> dict[str, tuple[object, object]]:
    import dataclasses

    diffs: dict[str, tuple[object, object]] = {}
    left_dict = dataclasses.asdict(left)  # type: ignore[arg-type]
    right_dict = dataclasses.asdict(right)  # type: ignore[arg-type]
    for key, left_value in left_dict.items():
        right_value = right_dict.get(key)
        if left_value != right_value:
            diffs[key] = (left_value, right_value)
    return diffs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train tactile GRU + cross-attn flow decoder "
            "(frozen ResNet features; loss-mode gt / predicted / gated)."
        )
    )
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--tactile-encoder-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=None,
        help="Override LeRobot dataset repo id (default: cache manifest configuration).",
    )
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=None,
        help="Optional local dataset root hint (currently unused by image loader; reserved).",
    )
    parser.add_argument(
        "--tactile-window-divisor",
        type=int,
        default=1,
        help="tactile_window = action_horizon // divisor (must divide evenly). Default 1.",
    )
    parser.add_argument(
        "--history-stride",
        type=int,
        default=1,
        help="Frame stride when looking back for the tactile window (default 1 = contiguous).",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("gt", "predicted", "gated"),
        default="gt",
        help=(
            "gt: FM(gt)+aux*MSE(decode,gt) (primary eval vs GT). "
            "predicted: FM vs VLA predicted_actions only (no aux; sanity check). "
            "gated: w*(FM(gt)+aux*MSE(decode,gt))+lambda*(1-w)*FM(pred). "
            "All modes always log both val_mse_gt and val_mse_pred."
        ),
    )
    parser.add_argument(
        "--gate-tau",
        type=float,
        default=0.5,
        help="Soft-gate midpoint tau for w=sigmoid((s-tau)/T). Default 0.5.",
    )
    parser.add_argument(
        "--gate-temperature",
        type=float,
        default=0.1,
        help="Soft-gate temperature T. Default 0.1.",
    )
    parser.add_argument(
        "--gate-lambda",
        type=float,
        default=1.0,
        help="Weight on (1-w)*L_stop in gated mode. Default 1.0.",
    )
    parser.add_argument(
        "--aux-decode-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on MSE(decode(x_base), gt) added inside every GT loss term "
            "(gt mode and gated L*). Set 0 to disable. Default 1.0."
        ),
    )
    parser.add_argument(
        "--aux-decode-steps",
        type=int,
        default=None,
        help=(
            "Euler steps used for aux decode MSE during training "
            "(default: same as --validation-steps)."
        ),
    )
    parser.add_argument(
        "--rank-weight",
        type=float,
        default=0.0,
        help="Weight of the gate-preference ranking loss. Default 0 disables it.",
    )
    parser.add_argument(
        "--rank-margin",
        type=float,
        default=0.0,
        help="Required MSE separation between the preferred and other endpoint.",
    )
    parser.add_argument(
        "--repair-weight",
        type=float,
        default=0.0,
        help="Weight requiring high-gate GT error to beat the frozen VLA baseline.",
    )
    parser.add_argument(
        "--repair-margin",
        type=float,
        default=0.0,
        help="Required high-gate GT MSE improvement over the VLA baseline.",
    )
    parser.add_argument(
        "--best-low-gate-max-mse-pred",
        type=float,
        default=0.01,
        help="Maximum low-gate FRS-to-VLA MSE for a feasible best checkpoint.",
    )
    parser.add_argument(
        "--best-min-high-gate-gain",
        type=float,
        default=0.0,
        help="Minimum high-gate GT gain for a feasible best checkpoint.",
    )

    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--lr-reference-dim", type=int, default=256)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--validation-steps", type=int, default=10)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Run full validation every N epochs (also always on the final epoch). Default 5.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/last (params + optimizer state if present).",
    )
    parser.add_argument(
        "--resume-from",
        type=pathlib.Path,
        help="Resume from an explicit checkpoint directory (overrides --resume).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Spawn process workers for video/parquet decode (0/1 = in-process threads only).",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=8,
        help="In-flight mp decode batches queued ahead of the trainer.",
    )
    parser.add_argument(
        "--load-threads",
        type=int,
        default=16,
        help="Per-process threads for unique-frame decode within a batch.",
    )
    parser.add_argument(
        "--pipeline-prefetch",
        type=int,
        default=4,
        help="Decoded image batches buffered while parent runs ResNet/train step.",
    )
    parser.add_argument(
        "--image-cache-size",
        type=int,
        default=8192,
        help="Total LRU decoded-frame budget (split across mp workers).",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=256,
        help="Frozen ResNet microbatch size on the parent process/GPU.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_decoder(
        cache_dir=args.cache_dir,
        tactile_encoder_dir=args.tactile_encoder_dir,
        output_dir=args.output_dir,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        tactile_window_divisor=args.tactile_window_divisor,
        history_stride=args.history_stride,
        loss_mode=args.loss_mode,
        gate_tau=args.gate_tau,
        gate_temperature=args.gate_temperature,
        gate_lambda=args.gate_lambda,
        aux_decode_weight=args.aux_decode_weight,
        aux_decode_steps=(
            args.validation_steps if args.aux_decode_steps is None else args.aux_decode_steps
        ),
        rank_weight=args.rank_weight,
        rank_margin=args.rank_margin,
        repair_weight=args.repair_weight,
        repair_margin=args.repair_margin,
        model_dim=args.model_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        warmup_epochs=args.warmup_epochs,
        lr_reference_dim=args.lr_reference_dim if args.lr_reference_dim > 0 else None,
        min_learning_rate_ratio=args.min_lr_ratio,
        cosine_decay=args.lr_schedule == "cosine",
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_steps=args.validation_steps,
        eval_every=args.eval_every,
        seed=args.seed,
        write_plots=not args.no_plots,
        num_workers=args.num_workers,
        prefetch_batches=args.prefetch_batches,
        load_threads=args.load_threads,
        pipeline_prefetch=args.pipeline_prefetch,
        image_cache_size=args.image_cache_size,
        encode_batch_size=args.encode_batch_size,
        resume=args.resume,
        resume_from=args.resume_from,
        best_low_gate_max_mse_pred=args.best_low_gate_max_mse_pred,
        best_min_high_gate_gain=args.best_min_high_gate_gain,
    )


if __name__ == "__main__":
    main()
