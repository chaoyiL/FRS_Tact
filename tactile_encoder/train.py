from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils.model import make_learning_rate_schedule

from tactile_encoder.evaluate_retrieval import evaluate_records
from tactile_encoder.utils.checkpoint import CHECKPOINT_NAME
from tactile_encoder.utils.checkpoint import load_checkpoint
from tactile_encoder.utils.checkpoint import load_train_state
from tactile_encoder.utils.checkpoint import save_checkpoint
from tactile_encoder.utils.clip_backend import CLIP_IMAGE_SIZE
from tactile_encoder.utils.clip_backend import DEFAULT_CLIP_MODEL_ID
from tactile_encoder.utils.clip_backend import ClipBackend
from tactile_encoder.utils.data import FutureRecord
from tactile_encoder.utils.data import batches
from tactile_encoder.utils.data import batch_uint8_to_float32
from tactile_encoder.utils.data import build_future_records
from tactile_encoder.utils.data import history_dataset_indices
from tactile_encoder.utils.data import resolve_data_keys
from tactile_encoder.utils.image_dataset import create_image_dataset
from tactile_encoder.utils.model import TactileClipConfig
from tactile_encoder.utils.model import init_memory_bank
from tactile_encoder.utils.model import init_trainable_params
from tactile_encoder.utils.model import make_train_step
from tactile_encoder.utils.prefetch import prefetch_iterator


HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_batch_recall@1",
    "train_batch_recall@5",
    "train_batch_mean_rank",
    "train_bank_filled_frac",
    "train_bank_hard_neg_logit_mean",
    "train_batch_vs_positive_gap",
    "val_recall@1",
    "val_recall@5",
    "val_mean_rank",
]


def _tree_to_jax(batch: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {key: jnp.asarray(value) for key, value in batch.items()}


def _repo_id_for_metadata(repo_id: Any) -> Any:
    if isinstance(repo_id, tuple):
        return list(repo_id)
    return repo_id


def _records_digest(records: Sequence[FutureRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record.dataset_index}:{record.future_dataset_index}:{record.episode_index}:{record.split}\n".encode()
        )
    return digest.hexdigest()


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


def _load_best_metrics(output_dir: pathlib.Path) -> tuple[float, float]:
    best_dir = output_dir / "best"
    if not (best_dir / CHECKPOINT_NAME).exists():
        return -1.0, float("inf")
    _, metadata = load_checkpoint(best_dir)
    metrics = metadata.get("metrics") or {}
    return float(metrics.get("val_recall@1", -1.0)), float(metrics.get("val_mean_rank", float("inf")))


def _make_optimizer(
    *,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float | None,
    warmup_steps: int,
    total_steps: int,
    min_learning_rate_ratio: float,
    cosine_decay: bool,
) -> optax.GradientTransformation:
    lr = make_learning_rate_schedule(
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_learning_rate_ratio=min_learning_rate_ratio,
        cosine_decay=cosine_decay,
    )
    # Keep Adam moments in float32 so checkpoints round-trip cleanly through npz.
    adamw = optax.adamw(lr, weight_decay=weight_decay)
    if grad_clip_norm is None:
        return adamw
    return optax.chain(optax.clip_by_global_norm(grad_clip_norm), adamw)


def train(
    *,
    dataset_repo_ids: str | Sequence[str],
    output_dir: pathlib.Path,
    clip_model_id: str,
    masked_rgb_key: str | None,
    future_offset: int,
    split_seed: int,
    val_fraction: float,
    frame_stride: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float | None,
    warmup_epochs: int,
    min_learning_rate_ratio: float,
    cosine_decay: bool,
    rgb_mask_patch_size: int,
    rgb_mask_ratio: float,
    eval_mask_seed: int,
    seed: int,
    clip_microbatch: int,
    num_workers: int,
    prefetch_batches: int,
    image_cache_size: int,
    preload_images: bool,
    loader: str,
    pair_threads: int,
    pipeline_prefetch: int,
    resume: bool,
    resume_from: pathlib.Path | None,
    eval_every: int,
    positive_temporal_window: int,
    memory_bank_size: int,
    hard_negatives_k: int,
    early_stop_patience: int,
    tactile_history: int,
    config_name: str | None = None,
) -> None:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if clip_microbatch <= 0:
        raise ValueError("clip_microbatch must be positive.")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive.")
    if prefetch_batches <= 0:
        raise ValueError("prefetch_batches must be positive.")
    if image_cache_size < 0:
        raise ValueError("image_cache_size must be non-negative.")
    if eval_every <= 0:
        raise ValueError("eval_every must be positive.")
    if positive_temporal_window < 0:
        raise ValueError("positive_temporal_window must be non-negative.")
    if memory_bank_size < 0:
        raise ValueError("memory_bank_size must be non-negative.")
    if hard_negatives_k < 0:
        raise ValueError("hard_negatives_k must be non-negative.")
    if early_stop_patience < 0:
        raise ValueError("early_stop_patience must be non-negative.")
    if tactile_history < 0:
        raise ValueError("tactile_history must be non-negative.")
    if loader not in ("thread", "mp"):
        raise ValueError(f"loader must be 'thread' or 'mp', got {loader!r}.")
    if pair_threads <= 0:
        raise ValueError("pair_threads must be positive.")
    if pipeline_prefetch <= 0:
        raise ValueError("pipeline_prefetch must be positive.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}.")

    resume_dir = _resolve_resume_dir(output_dir=output_dir, resume=resume, resume_from=resume_from)
    start_epoch = 1
    resumed_params = None
    resumed_opt_state = None
    resumed_memory_bank = None
    resumed_model_id = clip_model_id
    if resume_dir is not None:
        if not (resume_dir / CHECKPOINT_NAME).exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_dir}")
        resumed_params, resumed_opt_state, resume_metadata, resumed_memory_bank = load_train_state(
            resume_dir
        )
        start_epoch = int(resume_metadata["epoch"]) + 1
        resumed_model_id = str(resume_metadata.get("clip_model_id") or clip_model_id)
        if resumed_model_id != clip_model_id:
            print(
                f"warning: resume checkpoint clip_model_id={resumed_model_id!r} "
                f"differs from CLI {clip_model_id!r}; using checkpoint id.",
                flush=True,
            )
            clip_model_id = resumed_model_id
        print(
            f"resuming from {resume_dir} epoch={resume_metadata['epoch']} "
            f"next_epoch={start_epoch} has_opt_state={resumed_opt_state is not None} "
            f"has_memory_bank={resumed_memory_bank is not None}",
            flush=True,
        )
        if start_epoch > epochs:
            print(f"already finished: last epoch {resume_metadata['epoch']} >= --epochs {epochs}")
            return

    dataset_info = create_image_dataset(
        dataset_repo_ids,
        image_size=CLIP_IMAGE_SIZE,
        cache_size=image_cache_size,
        config_name=config_name,
    )
    dataset = dataset_info.dataset
    effective_loader = "thread" if preload_images else loader
    print(
        f"datasets={_repo_id_for_metadata(dataset_info.repo_id)} frames={len(dataset)} "
        f"image_cache_size={image_cache_size} loader={effective_loader} "
        f"num_workers={num_workers} prefetch_batches={prefetch_batches} "
        f"pair_threads={pair_threads} pipeline_prefetch={pipeline_prefetch} "
        f"preload_images={preload_images} tactile_backbone=resnet18"
    )
    keys = resolve_data_keys(masked_rgb_key=masked_rgb_key)
    record_set = build_future_records(
        dataset,
        future_offset=future_offset,
        val_fraction=val_fraction,
        split_seed=split_seed,
        frame_stride=frame_stride,
    )
    if preload_images:
        # Only decode frames touched by training/val pairs (stride + future/history).
        needed_indices = {
            int(record.dataset_index)
            for record in record_set.records
        } | {
            int(record.future_dataset_index)
            for record in record_set.records
        }
        if tactile_history > 0:
            for record in record_set.records:
                needed_indices.update(
                    history_dataset_indices(
                        dataset,
                        record,
                        history=tactile_history,
                        history_stride=frame_stride,
                    )
                )
        needed_indices = sorted(needed_indices)
        print(
            f"preload subset: {len(needed_indices)}/{len(dataset)} unique frames "
            f"(frame_stride={frame_stride}, future_offset={future_offset}, "
            f"tactile_history={tactile_history}, records={len(record_set.records)})",
            flush=True,
        )
        dataset.preload(
            num_workers=max(num_workers, 32),
            store_uint8=True,
            indices=needed_indices,
        )
    train_records = record_set.split_records("train")
    val_records = record_set.split_records("val")
    model_config = TactileClipConfig(
        tactile_image_count=keys.tactile_image_count,
        tactile_history=tactile_history,
    )
    backend = ClipBackend.from_pretrained(clip_model_id)
    if resumed_params is not None:
        params = resumed_params
    else:
        params = init_trainable_params(jax.random.key(seed), model_config)

    train_sample_count = len(train_records) * len(keys.sides)
    steps_per_epoch = max(1, (train_sample_count + batch_size - 1) // batch_size)
    total_steps = epochs * steps_per_epoch
    warmup_steps = min(warmup_epochs, epochs) * steps_per_epoch
    optimizer = _make_optimizer(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_learning_rate_ratio=min_learning_rate_ratio,
        cosine_decay=cosine_decay,
    )
    if resumed_opt_state is not None:
        opt_state = resumed_opt_state
    else:
        opt_state = optimizer.init(params)
        if resume_dir is not None:
            print("warning: optimizer state missing in checkpoint; reinitialized Adam state.", flush=True)
    train_step = make_train_step(
        backend.model,
        optimizer,
        rgb_mask_patch_size=rgb_mask_patch_size,
        rgb_mask_ratio=rgb_mask_ratio,
        config=model_config,
        microbatch_size=clip_microbatch,
        positive_temporal_window=positive_temporal_window,
        memory_bank_size=memory_bank_size,
        hard_negatives_k=hard_negatives_k,
    )
    if (
        resumed_memory_bank is not None
        and int(np.asarray(resumed_memory_bank["keys"]).shape[0]) == memory_bank_size
    ):
        memory_bank = resumed_memory_bank
    else:
        if resumed_memory_bank is not None and memory_bank_size > 0:
            print(
                "warning: resumed memory bank size "
                f"{int(np.asarray(resumed_memory_bank['keys']).shape[0])} != "
                f"--memory-bank-size {memory_bank_size}; reinitializing empty bank.",
                flush=True,
            )
        memory_bank = init_memory_bank(memory_bank_size, model_config.embedding_dim)
    print(
        f"batch_size={batch_size} clip_microbatch={clip_microbatch} "
        f"positive_temporal_window={positive_temporal_window} "
        f"memory_bank_size={memory_bank_size} hard_negatives_k={hard_negatives_k} "
        f"temperature={model_config.temperature} "
        f"tactile_history={tactile_history} history_stride={frame_stride} "
        f"early_stop_patience={early_stop_patience} "
        f"train_samples={train_sample_count} steps_per_epoch={steps_per_epoch} "
        f"start_epoch={start_epoch} epochs={epochs}"
    )
    base_key = jax.random.key(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    best_recall_at_1, best_mean_rank = _load_best_metrics(output_dir)
    records_sha256 = _records_digest(record_set.records)
    data_metadata = {
        "config_name": dataset_info.config_name,
        "dataset_repo_id": _repo_id_for_metadata(dataset_info.repo_id),
        "sides": [
            {
                "name": side.name,
                "current_rgb_key": side.current_rgb_key,
                "future_rgb_key": side.future_rgb_key,
                "masked_rgb_key": side.masked_rgb_key,
                "tactile_keys": list(side.tactile_keys),
            }
            for side in keys.sides
        ],
        "masked_rgb_key": keys.masked_rgb_key,
        "future_offset": future_offset,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "frame_stride": frame_stride,
        "rgb_mask_patch_size": rgb_mask_patch_size,
        "rgb_mask_ratio": rgb_mask_ratio,
        "eval_mask_seed": eval_mask_seed,
        "clip_microbatch": clip_microbatch,
        "positive_temporal_window": positive_temporal_window,
        "memory_bank_size": memory_bank_size,
        "hard_negatives_k": hard_negatives_k,
        "temperature": model_config.temperature,
        "tactile_history": tactile_history,
        "history_stride": frame_stride,
        "batch_size": batch_size,
        "train_episodes": list(record_set.train_episodes),
        "val_episodes": list(record_set.val_episodes),
        "train_record_count": len(train_records),
        "val_record_count": len(val_records),
        "train_sample_count": train_sample_count,
        "val_sample_count": len(val_records) * len(keys.sides),
        "records_sha256": records_sha256,
    }
    # Fresh runs must not inherit best/ metrics from a different dataset split.
    if resume_dir is None and best_recall_at_1 >= 0.0:
        best_dir = output_dir / "best"
        if (best_dir / CHECKPOINT_NAME).exists():
            _, best_meta = load_checkpoint(best_dir)
            prev_sha = (best_meta.get("extra_metadata") or {}).get("records_sha256")
            if prev_sha != records_sha256:
                print(
                    "warning: existing best/ belongs to a different data split "
                    f"(records_sha256 {prev_sha!r} != {records_sha256!r}); "
                    "resetting best metric baseline for this run.",
                    flush=True,
                )
                best_recall_at_1, best_mean_rank = -1.0, float("inf")

    history_exists = history_path.exists() and history_path.stat().st_size > 0
    history_mode = "a" if resume_dir is not None and history_exists else "w"
    mp_loader = None
    train_workers = 1 if preload_images else num_workers
    if effective_loader == "mp" and train_workers > 1:
        from tactile_encoder.utils.mp_batches import MpBatchLoader

        # Keep workers alive across epochs so each process does not re-open Arrow.
        mp_loader = MpBatchLoader(
            repo_ids=dataset_repo_ids,
            records=train_records,
            keys=keys,
            image_size=CLIP_IMAGE_SIZE,
            image_cache_size=image_cache_size,
            num_workers=train_workers,
            prefetch_batches=prefetch_batches,
            pair_threads=pair_threads,
            tactile_history=tactile_history,
            history_stride=frame_stride,
        )
        mp_loader.start()
    epochs_since_improve = 0
    best_epoch = 0
    stopped_early = False
    if best_recall_at_1 >= 0.0:
        # Existing best/ in output_dir: treat "last improve" as just before this run.
        best_epoch = max(0, start_epoch - 1)
    try:
        with history_path.open(history_mode, newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
            if history_mode == "w":
                writer.writeheader()
            compiled_first_step = False
            log_every = 50
            for epoch in range(start_epoch, epochs + 1):
                losses: list[float] = []
                recall1: list[float] = []
                recall5: list[float] = []
                mean_ranks: list[float] = []
                bank_filled: list[float] = []
                hard_neg_logits: list[float] = []
                pos_gaps: list[float] = []
                weights: list[int] = []
                pending_losses: list[Any] = []
                pending_metrics: list[Any] = []
                pending_weights: list[int] = []

                def _flush_pending() -> None:
                    if not pending_losses:
                        return
                    loss_host, metrics_host = jax.device_get((pending_losses, pending_metrics))
                    for loss_i, metrics_i, batch_weight in zip(
                        loss_host, metrics_host, pending_weights, strict=True
                    ):
                        losses.append(float(loss_i))
                        recall1.append(float(metrics_i["batch_recall_at_1"]))
                        recall5.append(float(metrics_i["batch_recall_at_5"]))
                        mean_ranks.append(float(metrics_i["batch_mean_rank"]))
                        bank_filled.append(float(metrics_i["bank_filled_frac"]))
                        hard_neg_logits.append(float(metrics_i["bank_hard_neg_logit_mean"]))
                        pos_gaps.append(float(metrics_i["batch_vs_positive_gap"]))
                        weights.append(int(batch_weight))
                    pending_losses.clear()
                    pending_metrics.clear()
                    pending_weights.clear()
                    print(
                        f"epoch={epoch}/{epochs} step={len(losses)}/{steps_per_epoch} "
                        f"loss={losses[-1]:.6f}",
                        flush=True,
                    )

                batch_source = batches(
                    dataset,
                    train_records,
                    keys,
                    batch_size=batch_size,
                    shuffle=True,
                    seed=split_seed + epoch,
                    image_size=CLIP_IMAGE_SIZE,
                    num_workers=train_workers,
                    prefetch_batches=prefetch_batches,
                    dataset_repo_ids=dataset_repo_ids,
                    loader=effective_loader,
                    image_cache_size=image_cache_size,
                    mp_loader=mp_loader,
                    pair_threads=pair_threads,
                    tactile_history=tactile_history,
                    history_stride=frame_stride,
                )
                if effective_loader == "mp" and train_workers > 1:
                    batch_source = prefetch_iterator(
                        batch_source,
                        buffer_size=pipeline_prefetch,
                        on_item=batch_uint8_to_float32,
                    )
                for batch_number, batch_np in enumerate(batch_source):
                    global_step = (epoch - 1) * steps_per_epoch + batch_number
                    if not compiled_first_step:
                        print(
                            "compiling first train step (XLA); this can take several minutes "
                            "and looks stuck at device_get until it finishes...",
                            flush=True,
                        )
                    params, opt_state, memory_bank, loss, step_metrics = train_step(
                        params,
                        opt_state,
                        backend.params,
                        _tree_to_jax(batch_np),
                        jax.random.fold_in(base_key, global_step),
                        memory_bank,
                    )
                    pending_losses.append(loss)
                    pending_metrics.append(step_metrics)
                    pending_weights.append(int(batch_np["current_rgb"].shape[0]))
                    is_last = batch_number + 1 == steps_per_epoch
                    should_flush = (not compiled_first_step) or is_last or (
                        (batch_number + 1) % log_every == 0
                    )
                    if should_flush:
                        _flush_pending()
                        if not compiled_first_step:
                            print(
                                f"first step ready: loss={losses[-1]:.6f} "
                                f"batch_recall@1={recall1[-1]:.4f}",
                                flush=True,
                            )
                            compiled_first_step = True

                train_loss = float(np.average(losses, weights=weights))
                train_recall1 = float(np.average(recall1, weights=weights))
                train_recall5 = float(np.average(recall5, weights=weights))
                train_mean_rank = float(np.average(mean_ranks, weights=weights))
                train_bank_filled = float(np.average(bank_filled, weights=weights))
                train_hard_neg_logit = float(np.average(hard_neg_logits, weights=weights))
                train_pos_gap = float(np.average(pos_gaps, weights=weights))
                run_eval = (epoch % eval_every == 0) or (epoch == epochs)
                if run_eval:
                    val_metrics, _ = evaluate_records(
                        clip_model=backend.model,
                        params=params,
                        frozen_clip_params=backend.params,
                        dataset=dataset,
                        records=val_records,
                        keys=keys,
                        batch_size=batch_size,
                        eval_mask_seed=eval_mask_seed,
                        rgb_mask_patch_size=rgb_mask_patch_size,
                        rgb_mask_ratio=rgb_mask_ratio,
                        config=model_config,
                        tactile_history=tactile_history,
                        history_stride=frame_stride,
                    )
                    val_recall1 = float(val_metrics["recall@1"])
                    val_recall5 = float(val_metrics["recall@5"])
                    val_mean_rank = float(val_metrics["mean_rank"])
                else:
                    val_recall1 = float("nan")
                    val_recall5 = float("nan")
                    val_mean_rank = float("nan")
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_batch_recall@1": train_recall1,
                    "train_batch_recall@5": train_recall5,
                    "train_batch_mean_rank": train_mean_rank,
                    "train_bank_filled_frac": train_bank_filled,
                    "train_bank_hard_neg_logit_mean": train_hard_neg_logit,
                    "train_batch_vs_positive_gap": train_pos_gap,
                    "val_recall@1": val_recall1,
                    "val_recall@5": val_recall5,
                    "val_mean_rank": val_mean_rank,
                }
                writer.writerow(row)
                history_file.flush()
                checkpoint_metrics = {
                    "train_loss": train_loss,
                    "train_batch_recall@1": train_recall1,
                    "train_batch_recall@5": train_recall5,
                    "train_batch_mean_rank": train_mean_rank,
                    "train_bank_filled_frac": train_bank_filled,
                    "train_bank_hard_neg_logit_mean": train_hard_neg_logit,
                    "train_batch_vs_positive_gap": train_pos_gap,
                    "val_recall@1": val_recall1,
                    "val_recall@5": val_recall5,
                    "val_mean_rank": val_mean_rank,
                }
                save_checkpoint(
                    output_dir / "last",
                    params,
                    epoch=epoch,
                    metrics=checkpoint_metrics,
                    model_id=clip_model_id,
                    config=model_config,
                    extra_metadata=data_metadata,
                    opt_state=opt_state,
                    memory_bank=memory_bank if memory_bank_size > 0 else None,
                )
                if run_eval and (
                    val_recall1 > best_recall_at_1
                    or (val_recall1 == best_recall_at_1 and val_mean_rank < best_mean_rank)
                ):
                    best_recall_at_1 = val_recall1
                    best_mean_rank = val_mean_rank
                    best_epoch = epoch
                    epochs_since_improve = 0
                    save_checkpoint(
                        output_dir / "best",
                        params,
                        epoch=epoch,
                        metrics=checkpoint_metrics,
                        model_id=clip_model_id,
                        config=model_config,
                        extra_metadata=data_metadata,
                        opt_state=opt_state,
                        memory_bank=memory_bank if memory_bank_size > 0 else None,
                    )
                elif run_eval:
                    epochs_since_improve = epoch - best_epoch if best_epoch > 0 else epochs_since_improve + eval_every
                if run_eval:
                    print(
                        f"epoch={epoch}/{epochs} train_loss={train_loss:.6f} "
                        f"val_recall@1={val_recall1:.6f} val_recall@5={val_recall5:.6f} "
                        f"val_mean_rank={val_mean_rank:.2f}"
                        + (
                            f" early_stop={epochs_since_improve}/{early_stop_patience}"
                            if early_stop_patience > 0
                            else ""
                        )
                    )
                else:
                    print(
                        f"epoch={epoch}/{epochs} train_loss={train_loss:.6f} "
                        f"(skipped val; next eval every {eval_every} epochs)"
                    )
                if (
                    early_stop_patience > 0
                    and run_eval
                    and best_epoch > 0
                    and epochs_since_improve >= early_stop_patience
                ):
                    print(
                        f"early stopping at epoch={epoch}: no val improvement for "
                        f"{epochs_since_improve} epochs since best_epoch={best_epoch} "
                        f"(patience={early_stop_patience})",
                        flush=True,
                    )
                    stopped_early = True
                    break
    finally:
        if mp_loader is not None:
            mp_loader.close()
    if stopped_early:
        print(f"stopped_early=true best_epoch={best_epoch}")
    print(f"best_val_recall@1={best_recall_at_1:.6f} best_val_mean_rank={best_mean_rank:.2f}")
    print(f"checkpoints={output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train tactile ResNet18 + frozen CLIP future contrastive pretraining."
    )
    parser.add_argument(
        "--dataset-repo-id",
        action="append",
        required=True,
        dest="dataset_repo_ids",
        help=(
            "LeRobot dataset repo id (repeatable). "
            "Example: --dataset-repo-id org/dataset_a --dataset-repo-id org/dataset_b"
        ),
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Optional run label stored in checkpoint metadata (defaults to repo id).",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--clip-model-id", default=DEFAULT_CLIP_MODEL_ID)
    parser.add_argument(
        "--masked-rgb-key",
        default="",
        help="Optional RGB key used instead of each side's current camera for masking.",
    )
    parser.add_argument("--future-offset", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument(
        "--positive-temporal-window",
        type=int,
        default=None,
        help=(
            "Treat future frames within +/- this many dataset indices (same episode "
            "and wrist side) as additional positives in the contrastive loss. "
            "Defaults to --frame-stride. Use 0 to disable (exact target only)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Contrastive batch size. Each sample encodes 4 images (2 RGB + 2 tactile).",
    )
    parser.add_argument(
        "--clip-microbatch",
        type=int,
        default=64,
        help="Max frozen RGB images encoded through CLIP at once.",
    )
    parser.add_argument(
        "--loader",
        choices=("thread", "mp"),
        default="mp",
        help="Batch decode backend: spawn process workers (mp) or threads (thread).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Process workers (loader=mp) or thread workers (loader=thread). Ignored when preloaded.",
    )
    parser.add_argument(
        "--pair-threads",
        type=int,
        default=8,
        help="Threads per mp worker for parallel load_pair within a batch.",
    )
    parser.add_argument(
        "--pipeline-prefetch",
        type=int,
        default=4,
        help="Main-process batches to decode/unpickle ahead while GPU trains.",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=12,
        help="How many batches mp workers decode ahead of the GPU train step.",
    )
    parser.add_argument(
        "--image-cache-size",
        type=int,
        default=65536,
        help="Total LRU budget of decoded frames across workers (split per worker).",
    )
    parser.add_argument(
        "--preload-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Decode all frames into RAM before training (best GPU utilization).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--rgb-mask-patch-size", type=int, default=16)
    parser.add_argument("--rgb-mask-ratio", type=float, default=0.5)
    parser.add_argument("--eval-mask-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
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
        "--eval-every",
        type=int,
        default=5,
        help="Run full validation every N epochs (always evaluates on the final epoch).",
    )
    parser.add_argument(
        "--memory-bank-size",
        type=int,
        default=0,
        help=(
            "Circular queue of past future-RGB embeddings used as extra contrastive "
            "negatives (MoCo-style). 0 disables the memory bank. Try 8192 or 16384."
        ),
    )
    parser.add_argument(
        "--hard-negatives-k",
        type=int,
        default=0,
        help=(
            "If >0 and --memory-bank-size >0, each query only uses the top-K hardest "
            "cross-episode bank negatives in the InfoNCE denominator (plus all "
            "in-batch logits and exact bank positives). Same-episode bank keys are "
            "never used as negatives. 0 keeps all cross-episode bank keys. Try 256."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help=(
            "Stop after this many epochs without val_recall@1 improvement "
            "(checked on eval epochs). 0 disables early stopping."
        ),
    )
    parser.add_argument(
        "--tactile-history",
        type=int,
        default=0,
        help=(
            "Number of past tactile timesteps for shared-GRU temporal encoding "
            "(lookback step = --frame-stride). 0 keeps the original "
            "vision + current two tactile ResNet embeddings. "
            ">0 encodes each finger stream with a shared GRU (hidden=256) and "
            "concatenates vision + two GRU hiddens into the future FC."
        ),
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train(
        dataset_repo_ids=args.dataset_repo_ids,
        output_dir=args.output_dir,
        clip_model_id=args.clip_model_id,
        masked_rgb_key=args.masked_rgb_key or None,
        future_offset=args.future_offset,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        frame_stride=args.frame_stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        warmup_epochs=args.warmup_epochs,
        min_learning_rate_ratio=args.min_lr_ratio,
        cosine_decay=args.lr_schedule == "cosine",
        rgb_mask_patch_size=args.rgb_mask_patch_size,
        rgb_mask_ratio=args.rgb_mask_ratio,
        eval_mask_seed=args.eval_mask_seed,
        seed=args.seed,
        clip_microbatch=args.clip_microbatch,
        num_workers=args.num_workers,
        prefetch_batches=args.prefetch_batches,
        image_cache_size=args.image_cache_size,
        preload_images=args.preload_images,
        loader=args.loader,
        pair_threads=args.pair_threads,
        pipeline_prefetch=args.pipeline_prefetch,
        resume=args.resume,
        resume_from=args.resume_from,
        eval_every=args.eval_every,
        positive_temporal_window=(
            args.frame_stride
            if args.positive_temporal_window is None
            else args.positive_temporal_window
        ),
        memory_bank_size=args.memory_bank_size,
        hard_negatives_k=args.hard_negatives_k,
        early_stop_patience=args.early_stop_patience,
        tactile_history=args.tactile_history,
        config_name=args.config_name,
    )


if __name__ == "__main__":
    main()
