"""Pure-JAX SmolVLA implementation.

The JAX backend intentionally keeps checkpoint tensors in their original
PyTorch/safetensors layout.  This makes conversion lossless and lets the same
parameter tree round-trip between the reference and JAX implementations.
"""

from importlib import import_module
from typing import Any

from .validation import (
    CheckpointContract,
    CheckpointValidationReport,
    contract_from_config,
    validate_checkpoint,
)

_LAZY_IMPORTS = {
    "JaxSmolVLA": (".modeling", "JaxSmolVLA"),
    "JaxSmolVLAConfig": (".configuration", "JaxSmolVLAConfig"),
    "JaxSmolVLAPolicy": (".policy", "JaxSmolVLAPolicy"),
}

__all__ = [
    "CheckpointContract",
    "CheckpointValidationReport",
    "JaxSmolVLA",
    "JaxSmolVLAConfig",
    "JaxSmolVLAPolicy",
    "contract_from_config",
    "validate_checkpoint",
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
