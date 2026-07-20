"""Compatibility re-exports for SmolVLA evaluation helpers.

Prefer importing from ``modalities_eval_scripts.utils`` directly.
"""

from __future__ import annotations

from modalities_eval_scripts.utils import (  # noqa: F401
    EpisodeData,
    EvalObservation,
    SmolVLAEvalModel,
    VelocityContext,
    _add_batch_dim,
    _as_scalar,
    _batch_actions,
    _batch_observation,
    _scalar,
    _stack_observations,
    ablate_modality_observation,
    add_eval_data_arguments,
    create_velocity_context,
    load_episode,
    load_model,
    load_model_from_args,
    parse_rename_map,
    predict_velocity_with_context,
)

__all__ = [
    "EpisodeData",
    "EvalObservation",
    "SmolVLAEvalModel",
    "VelocityContext",
    "_add_batch_dim",
    "_as_scalar",
    "_batch_actions",
    "_batch_observation",
    "_scalar",
    "_stack_observations",
    "ablate_modality_observation",
    "add_eval_data_arguments",
    "create_velocity_context",
    "load_episode",
    "load_model",
    "load_model_from_args",
    "parse_rename_map",
    "predict_velocity_with_context",
]
