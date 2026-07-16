#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, TypedDict, Unpack

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.processor import (
    PolicyProcessorPipeline,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.types import PolicyAction
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .pretrained import PreTrainedPolicy
from .smolvla.configuration_smolvla import SmolVLAConfig


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """Return the policy class for the given registered name."""
    if name == "smolvla":
        from .smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    raise ValueError(f"Policy type '{name}' is not available. Only 'smolvla' is supported.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    """Instantiate a policy configuration object."""
    if policy_type == "smolvla":
        return SmolVLAConfig(**kwargs)
    raise ValueError(f"Policy type '{policy_type}' is not available. Only 'smolvla' is supported.")


class ProcessorConfigKwargs(TypedDict, total=False):
    preprocessor_config_filename: str | None
    postprocessor_config_filename: str | None
    preprocessor_overrides: dict[str, Any] | None
    postprocessor_overrides: dict[str, Any] | None
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    pretrained_revision: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create or load pre- and post-processor pipelines for SmolVLA."""
    if not isinstance(policy_cfg, SmolVLAConfig):
        raise ValueError(f"Only SmolVLA is supported, got {type(policy_cfg).__name__}.")

    if pretrained_path:
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
            ),
            overrides=kwargs.get("preprocessor_overrides", {}),
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
            revision=pretrained_revision,
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
            ),
            overrides=kwargs.get("postprocessor_overrides", {}),
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
            revision=pretrained_revision,
        )
        return preprocessor, postprocessor

    from .smolvla.processor_smolvla import make_smolvla_pre_post_processors

    return make_smolvla_pre_post_processors(
        config=policy_cfg,
        dataset_stats=kwargs.get("dataset_stats"),
    )
