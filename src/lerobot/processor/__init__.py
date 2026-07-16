#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.types import PolicyAction, TransitionKey

from .batch_processor import AddBatchDimensionProcessorStep
from .converters import (
    batch_to_transition,
    create_transition,
    identity_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from .device_processor import DeviceProcessorStep
from .newline_task_processor import NewLineTaskProcessorStep
from .normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep, hotswap_stats
from .pipeline import (
    DataProcessorPipeline,
    PolicyProcessorPipeline,
    ProcessorKwargs,
    ProcessorStep,
    ProcessorStepRegistry,
)
from .relative_action_processor import RelativeActionsProcessorStep
from .rename_processor import RenameObservationsProcessorStep, rename_stats
from .tokenizer_processor import TokenizerProcessorStep

__all__ = [
    "AddBatchDimensionProcessorStep",
    "DataProcessorPipeline",
    "DeviceProcessorStep",
    "NewLineTaskProcessorStep",
    "NormalizerProcessorStep",
    "PolicyAction",
    "PolicyProcessorPipeline",
    "ProcessorKwargs",
    "ProcessorStep",
    "ProcessorStepRegistry",
    "RelativeActionsProcessorStep",
    "RenameObservationsProcessorStep",
    "TokenizerProcessorStep",
    "TransitionKey",
    "batch_to_transition",
    "create_transition",
    "hotswap_stats",
    "identity_transition",
    "policy_action_to_transition",
    "rename_stats",
    "transition_to_batch",
    "transition_to_policy_action",
    "UnnormalizerProcessorStep",
]
