"""Train one joint 20D FRS; source Pi0.5 models and tactile ResNet stay frozen."""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from pathlib import Path

import numpy as np


def train_shared(
    config: dict,
    sources: list[dict],
    *,
    resume: bool = False,
    evaluate_only: bool = False,
    evaluate_task: str | None = None,
) -> None:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from train_pi05_frs.pi05_cache.cache import atomic_write_json
    from train_pi05_frs.shared_data import SharedTaskData, encoder_fingerprint
    from train_pi05_frs.shared_model import decode_shared, make_shared_train_step
    from train_pi05_frs.utils.checkpoint import (
        load_checkpoint,
        load_optimizer_state,
        restore_optimizer_state,
        save_checkpoint,
    )
    from train_pi05_frs.utils.model import DecoderConfig, TactileConditionedFlowDecoder, make_optimizer

    training = config["training"]
    seed = int(config["seed"])
    horizon = int(config["action_horizon"])
    window = int(training["tactile_window"])
    data = SharedTaskData(
        sources,
        encoder_sha256=encoder_fingerprint(config["tactile_encoder"]),
        tactile_window=window,
        history_stride=int(training["history_stride"]),
        num_train_episodes=int(config["num_train_episodes"]),
        num_val_episodes=int(config["num_val_episodes"]),
        seed=seed,
    )
    if data.horizon != horizon:
        raise ValueError("Configured horizon differs from native action caches")
    if evaluate_task is not None and evaluate_task not in data.selection:
        raise ValueError(f"Unknown evaluation task: {evaluate_task}")
    output = Path(config["output"])
    decoder_config = DecoderConfig(
        action_dim=20,
        action_horizon=horizon,
        tactile_window=window,
        resnet_embedding_dim=data.embedding_dim,
        model_dim=int(training["model_dim"]),
        depth=int(training["depth"]),
        num_heads=int(training["num_heads"]),
        mlp_ratio=int(training["mlp_ratio"]),
        state_conditioning=False,
        state_dim=0,
        num_tactile_tokens=4,
        tactile_token_identity=True,
    )
    provenance = {
        "shared_frs_interface_version": 1,
        "cache_sha256": data.fingerprint,
        "loss_mode": training["loss_mode"],
        "training_configuration": training,
        "canonical_action_order": ["left_10d", "right_10d"],
        "inactive_wrist_policy": "masked_every_ode_step_and_attention",
        "state_conditioning": False,
        "sources": sources,
        "episode_selection": data.selection,
        "tactile_encoder": config["tactile_encoder"],
        "tactile_encoder_repo": config.get("tactile_encoder_repo"),
        "tactile_encoder_revision": config.get("tactile_encoder_revision"),
    }
    start_epoch = 1
    checkpoint = output / ("best" if evaluate_only else "last")
    if resume or evaluate_only:
        model, metadata = load_checkpoint(checkpoint)
        extra = metadata.get("extra_metadata", {})
        if extra.get("shared_frs_interface_version") != 1 or extra.get("cache_sha256") != data.fingerprint:
            raise ValueError("Shared checkpoint/cache provenance differs")
        if extra.get("training_configuration") != training or metadata[
            "decoder_config"
        ] != dataclasses.asdict(decoder_config):
            raise ValueError("Shared checkpoint/training configuration differs")
        start_epoch = int(metadata["epoch"]) + 1
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Training output is not empty: {output}; use --resume or a new output")
        model = TactileConditionedFlowDecoder(decoder_config, rngs=nnx.Rngs(seed))
    output.mkdir(parents=True, exist_ok=True)
    batch_size = int(training["batch_size"])
    tau, temperature = float(training["gate_tau"]), float(training["gate_temperature"])
    print(
        f"shared_frs=one_model action_dim=20 tactile_tokens=4 frozen_vlas={len(sources)} "
        f"state_conditioning=false train_samples={len(data.rows['train'])} val_samples={len(data.rows['val'])}",
        flush=True,
    )
    for source in sources:
        selected = data.selection[source["name"]]
        print(
            f"task={source['name']} train_episodes={len(selected['train'])} val_episodes={len(selected['val'])}",
            flush=True,
        )

    @nnx.jit
    def decode(candidate, batch):
        return decode_shared(
            candidate, batch, num_steps=int(training["validation_steps"]), solver=training["decode_solver"]
        )

    def validate():
        values = {
            source["name"]: {"gt": [], "vla": [], "baseline": [], "wrists": [], "gates": []}
            for source in sources
        }
        for batch_np in data.batches(
            "val", batch_size=batch_size, seed=seed, tau=tau, temperature=temperature
        ):
            batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
            repaired = np.asarray(decode(model, batch))

            def per_wrist(x, y):
                return np.square(x - y).reshape(len(x), horizon, 2, 10).mean(axis=(1, 3))

            gt = per_wrist(repaired, batch_np["gt"])
            vla = per_wrist(repaired, batch_np["predicted"])
            baseline = per_wrist(batch_np["predicted"], batch_np["gt"])
            active = batch_np["action_mask"].reshape(-1, 2, 10).any(axis=-1)
            for source_i, source in enumerate(sources):
                selected = batch_np["source"] == source_i
                for key, value in (
                    ("gt", gt),
                    ("vla", vla),
                    ("baseline", baseline),
                    ("wrists", active),
                    ("gates", batch_np["gates"]),
                ):
                    values[source["name"]][key].extend(value[selected])
        metrics = {}
        task_mses, violations = [], []
        for name, values_np in values.items():
            arrays = {key: np.asarray(value) for key, value in values_np.items()}
            active = arrays["wrists"].astype(bool)
            if not active.any():
                raise ValueError(f"No active validation samples for {name}")
            mse = float(arrays["gt"][active].mean())
            metrics[f"val/{name}/mse_gt"] = mse
            metrics[f"val/{name}/vla_mse_gt"] = float(arrays["baseline"][active].mean())
            metrics[f"val/{name}/gain"] = metrics[f"val/{name}/vla_mse_gt"] - mse
            metrics[f"val/{name}/sample_count"] = len(active)
            task_mses.append(mse)
            for wrist_i, wrist in enumerate(("left", "right")):
                mask = active[:, wrist_i]
                if not mask.any():
                    continue
                low = mask & (arrays["gates"][:, wrist_i] <= float(training["low_gate_threshold"]))
                high = mask & (arrays["gates"][:, wrist_i] >= float(training["high_gate_threshold"]))
                for key in ("gt", "vla", "baseline"):
                    metrics[f"val/{name}/{wrist}/{key}_mse"] = float(arrays[key][mask, wrist_i].mean())
                if low.any():
                    unsafe = float(
                        (
                            np.minimum(arrays["gt"][low, wrist_i], arrays["vla"][low, wrist_i])
                            > float(training["low_gate_safety_margin"])
                        ).mean()
                    )
                    metrics[f"val/{name}/{wrist}/low_unsafe_frac"] = unsafe
                    violations.append(max(unsafe - float(training["best_max_low_gate_unsafe_frac"]), 0))
                if high.any():
                    gain = float((arrays["baseline"][high, wrist_i] - arrays["gt"][high, wrist_i]).mean())
                    rank = float(
                        (
                            arrays["gt"][high, wrist_i] + float(training["rank_margin"])
                            <= arrays["vla"][high, wrist_i]
                        ).mean()
                    )
                    metrics[f"val/{name}/{wrist}/high_gain"] = gain
                    metrics[f"val/{name}/{wrist}/high_rank_frac"] = rank
                    violations.extend(
                        (
                            max(float(training["best_min_high_gate_gain"]) - gain, 0),
                            max(float(training["best_min_high_gate_rank_satisfied_frac"]) - rank, 0),
                        )
                    )
        metrics["val_mse"] = float(np.mean(task_mses))
        metrics["val_worst_task_mse"] = max(task_mses)
        metrics["val_constraint_violation"] = sum(violations)
        return metrics

    if evaluate_only:
        metrics = validate()
        if evaluate_task:
            metrics = {
                key: value for key, value in metrics.items() if key.startswith(f"val/{evaluate_task}/")
            }
        atomic_write_json(output / f"evaluation_{evaluate_task or 'all'}.json", metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        return
    epochs = int(training["epochs"])
    steps = math.ceil(len(data.rows["train"]) / batch_size)
    optimizer = make_optimizer(
        model,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        grad_clip_norm=float(training["grad_clip_norm"]),
        warmup_steps=min(int(training["warmup_epochs"]) * steps, max(epochs * steps - 1, 0)),
        total_steps=epochs * steps,
        min_learning_rate_ratio=float(training["min_lr_ratio"]),
    )
    if resume:
        state, step = load_optimizer_state(checkpoint)
        if state is None:
            raise ValueError("Cannot resume shared FRS without optimizer state")
        restore_optimizer_state(optimizer, opt_state=state, step=step)
    loss_options = {
        key: training[key]
        for key in (
            "loss_mode",
            "aux_decode_weight",
            "aux_decode_steps",
            "low_gate_safety_weight",
            "low_gate_safety_margin",
            "rank_weight",
            "rank_margin",
            "repair_weight",
            "repair_margin",
        )
    }
    loss_options.update(
        low_threshold=training["low_gate_threshold"],
        high_threshold=training["high_gate_threshold"],
        aux_decode_solver=training["decode_solver"],
    )
    step_fn = make_shared_train_step(**loss_options)
    counts = np.bincount([source for source, _ in data.rows["train"]], minlength=len(sources))
    weights = len(data.rows["train"]) / (len(sources) * counts)
    best_key = (float("inf"),) * 3
    if resume and (output / "best").exists():
        _, best = load_checkpoint(output / "best")
        m = best["metrics"]
        best_key = (m["val_constraint_violation"], m["val_worst_task_mse"], m["val_mse"])
    atomic_write_json(output / "shared_training_manifest.json", provenance)
    history = output / "history.csv"
    history_fields = ["epoch", "train_loss", "val_mse", "val_worst_task_mse", "val_constraint_violation"]
    if resume and history.exists():
        with history.open() as stream:
            if next(csv.reader(stream), []) != history_fields:
                raise ValueError("Shared history header differs")
    with history.open("a" if resume else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history_fields)
        if not resume or history.stat().st_size == 0:
            writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            loss_sum, samples = 0.0, 0
            for batch_i, batch_np in enumerate(
                data.batches(
                    "train", batch_size=batch_size, seed=seed + epoch, tau=tau, temperature=temperature
                )
            ):
                batch_np["sample_weight"] = (
                    weights[batch_np["source"]].astype(np.float32)
                    if training["balance_tasks"]
                    else np.ones(len(batch_np["source"]), dtype=np.float32)
                )
                batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
                key = jax.random.fold_in(jax.random.key(seed), (epoch - 1) * steps + batch_i)
                loss, _ = step_fn(model, optimizer, batch, key)
                if not np.isfinite(float(loss)):
                    raise FloatingPointError(f"Non-finite shared FRS loss at epoch={epoch}, batch={batch_i}")
                n = len(batch_np["source"])
                loss_sum += float(loss) * n
                samples += n
                if batch_i == 0 or (batch_i + 1) % 100 == 0:
                    print(
                        f"epoch={epoch}/{epochs} step={batch_i + 1}/{steps} loss={float(loss):.6f}",
                        flush=True,
                    )
            metrics = {"train_loss": loss_sum / samples}
            if epoch % int(training["eval_every"]) == 0 or epoch == epochs:
                metrics.update(validate())
                atomic_write_json(output / f"validation_epoch_{epoch:04d}.json", metrics)
            save_checkpoint(
                output / "last",
                model,
                epoch=epoch,
                metrics=metrics,
                extra_metadata=provenance,
                optimizer=optimizer,
            )
            if "val_mse" in metrics:
                candidate = (
                    metrics["val_constraint_violation"],
                    metrics["val_worst_task_mse"],
                    metrics["val_mse"],
                )
                if candidate < best_key:
                    best_key = candidate
                    save_checkpoint(
                        output / "best",
                        model,
                        epoch=epoch,
                        metrics=metrics,
                        extra_metadata=provenance,
                        optimizer=optimizer,
                    )
            writer.writerow(
                {"epoch": epoch, **{field: metrics.get(field, float("nan")) for field in history_fields[1:]}}
            )
            stream.flush()
            print(f"epoch={epoch} metrics={json.dumps(metrics)}", flush=True)
