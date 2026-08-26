"""Checkpoint resume compatibility rules without model/runtime dependencies."""

import json


RESUME_CONFIG_KEYS = (
    "model_type",
    "dataset_id",
    "hidden_dim",
    "layers",
    "heads",
    "image_size",
    "inference_steps",
    "rope_height",
    "rope_width",
    "use_task_condition",
    "num_tasks",
    "action_dim",
    "chunk_size",
    "world_size",
    "batch_size",
    "seed",
    "train_samples",
    "steps_per_epoch",
    "objective_version",
    "validation_metric_version",
    "validation_seed",
    # validation_noise_seeds is NOT a training-state field — it only controls
    # how many noise seeds validation runs (an eval-only setting). Changing it
    # does not affect optimizer/scheduler/epoch state, so it must not block an
    # otherwise-exact resume. Removed from the strict-check keys so a run can
    # resume after lowering VALIDATION_NOISE_SEEDS (e.g. 3 -> 1 to speed up the
    # 187GB validation-set cold-read).
    "lr",
    "lr_final",
    "backbone_lr",
    "backbone_lr_final",
    "weight_decay",
    "warmup_steps",
    "cosine_t_max_steps",
    "backbone_freeze_steps",
    "backbone_bn_eval",
    "scheduler_type",
    "optimizer_group_names",
    "rank_seed_scheme",
    "augmentation",
    "early_stopping_min_delta",
)

FINETUNE_RESUME_OVERRIDE_KEYS = frozenset(
    {
        "lr",
        "lr_final",
        "backbone_lr",
        "backbone_lr_final",
        "warmup_steps",
        "cosine_t_max_steps",
        "scheduler_type",
    }
)


def validate_resume_config(
    checkpoint_config: dict,
    current_config: dict,
    resume_mode: str = "exact",
    expected_training_state_version: int = 2,
    allowed_overrides: frozenset[str] | set[str] = frozenset(),
) -> None:
    if checkpoint_config.get("training_state_version") != expected_training_state_version:
        raise ValueError(
            "Checkpoint uses an incompatible training state version. Legacy one-group/"
            "epoch-scheduler checkpoints may be used for inference, but cannot "
            "state-resume this optimizer."
        )
    if resume_mode not in {"exact", "finetune"}:
        raise ValueError(f"Unknown resume mode: {resume_mode!r}")
    mode_allowed_overrides = (
        FINETUNE_RESUME_OVERRIDE_KEYS
        if resume_mode == "finetune"
        else frozenset()
    )
    allowed_overrides = frozenset(allowed_overrides) | mode_allowed_overrides
    mismatches = [
        key
        for key in RESUME_CONFIG_KEYS
        if key not in allowed_overrides
        if checkpoint_config.get(key) != current_config.get(key)
    ]
    if mismatches:
        details = {
            key: {
                "checkpoint": checkpoint_config.get(key),
                "current": current_config.get(key),
            }
            for key in mismatches
        }
        raise ValueError(f"Resume configuration mismatch: {json.dumps(details)}")
