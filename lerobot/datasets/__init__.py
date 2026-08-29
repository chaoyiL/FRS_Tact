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

__all__ = [
    "CODEBASE_VERSION",
    "LeRobotDataset",
    "LeRobotDatasetMetadata",
    "aggregate_stats",
    "check_video_encoder_parameters_pyav",
    "detect_available_encoders_pyav",
    "get_feature_stats",
]


def __getattr__(name: str):
    """Load optional dataset surfaces only when a caller actually uses them."""

    if name in {"aggregate_stats", "get_feature_stats"}:
        from .compute_stats import aggregate_stats, get_feature_stats

        value = {"aggregate_stats": aggregate_stats, "get_feature_stats": get_feature_stats}[name]
    elif name in {"CODEBASE_VERSION", "LeRobotDatasetMetadata"}:
        from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata

        value = {
            "CODEBASE_VERSION": CODEBASE_VERSION,
            "LeRobotDatasetMetadata": LeRobotDatasetMetadata,
        }[name]
    elif name == "LeRobotDataset":
        from .lerobot_dataset import LeRobotDataset

        value = LeRobotDataset
    elif name in {"check_video_encoder_parameters_pyav", "detect_available_encoders_pyav"}:
        from .pyav_utils import check_video_encoder_parameters_pyav, detect_available_encoders_pyav

        value = {
            "check_video_encoder_parameters_pyav": check_video_encoder_parameters_pyav,
            "detect_available_encoders_pyav": detect_available_encoders_pyav,
        }[name]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
