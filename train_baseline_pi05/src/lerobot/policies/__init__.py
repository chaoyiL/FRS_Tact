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

# Deliberately empty of re-exports. This used to do
# `from .smolvla_jax import JaxSmolVLA, JaxSmolVLAConfig, JaxSmolVLAPolicy`, which meant every
# `import lerobot.policies.pi05_jax` first executed this file and pulled in SmolVLA's entire model
# stack -- on a branch whose jax/flax/transformers pins come from openpi and are not expected to
# keep SmolVLA working. Any import failure there would have taken pi0.5 down with it, before it
# did anything. Import policies from their own subpackage (`lerobot.policies.pi05_jax`) instead.

__all__: list[str] = []
