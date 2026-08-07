"""Standalone visual-only JAX SmolVLA implementation."""

from importlib import import_module
from typing import Any

_LAZY_IMPORTS = {
    "JaxSmolVLA": (".modeling", "JaxSmolVLA"),
    "JaxSmolVLAConfig": (".configuration", "JaxSmolVLAConfig"),
    "JaxSmolVLAPolicy": (".policy", "JaxSmolVLAPolicy"),
}

__all__ = ["JaxSmolVLA", "JaxSmolVLAConfig", "JaxSmolVLAPolicy"]


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
