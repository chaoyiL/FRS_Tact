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

from importlib import import_module
from typing import Any

_LAZY_IMPORTS = {
    "JaxSmolVLA": (".smolvla_jax", "JaxSmolVLA"),
    "JaxSmolVLAConfig": (".smolvla_jax", "JaxSmolVLAConfig"),
    "JaxSmolVLAPolicy": (".smolvla_jax", "JaxSmolVLAPolicy"),
}

__all__ = [
    "JaxSmolVLA",
    "JaxSmolVLAConfig",
    "JaxSmolVLAPolicy",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
