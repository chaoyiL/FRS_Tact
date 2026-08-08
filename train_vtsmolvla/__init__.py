"""Vision-tactile extensions for the standalone visual JAX SmolVLA core."""

from importlib import import_module
from typing import Any

_LAZY_IMPORTS = {
    "VTSmolVLAConfig": (".configuration", "VTSmolVLAConfig"),
    "VTJaxSmolVLA": (".modeling", "VTJaxSmolVLA"),
    "VTJaxSmolVLAPolicy": (".policy", "VTJaxSmolVLAPolicy"),
    "VTJaxSmolVLAPreprocessor": (".preprocessing", "VTJaxSmolVLAPreprocessor"),
    "VTJaxSmolVLATrainer": (".training", "VTJaxSmolVLATrainer"),
    "VTLeRobotJaxDataLoader": (".data", "VTLeRobotJaxDataLoader"),
    "VTJaxSmolVLADataLoader": (".data", "VTJaxSmolVLADataLoader"),
}

__all__ = sorted(_LAZY_IMPORTS)


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
