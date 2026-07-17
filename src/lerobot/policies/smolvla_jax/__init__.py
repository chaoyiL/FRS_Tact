"""Pure-JAX SmolVLA implementation.

The JAX backend intentionally keeps checkpoint tensors in their original
PyTorch/safetensors layout.  This makes conversion lossless and lets the same
parameter tree round-trip between the reference and JAX implementations.
"""

from .configuration import JaxSmolVLAConfig
from .modeling import JaxSmolVLA
from .policy import JaxSmolVLAPolicy

__all__ = ["JaxSmolVLA", "JaxSmolVLAConfig", "JaxSmolVLAPolicy"]
