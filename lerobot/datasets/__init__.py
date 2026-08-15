#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team.
# All rights reserved.
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

"""Minimal LeRobot dataset read API for SmolVLA inference workflows."""

from lerobot.utils.import_utils import require_package

require_package("datasets", extra="dataset")
require_package("av", extra="dataset")

from .compute_stats import aggregate_stats, get_feature_stats
from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .pyav_utils import check_video_encoder_parameters_pyav, detect_available_encoders_pyav

__all__ = [
    "CODEBASE_VERSION",
    "LeRobotDataset",
    "LeRobotDatasetMetadata",
    "aggregate_stats",
    "check_video_encoder_parameters_pyav",
    "detect_available_encoders_pyav",
    "get_feature_stats",
]
