from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from train_smolvla.configuration import JaxSmolVLAConfig


@dataclass(frozen=True)
class VTSmolVLAConfig(JaxSmolVLAConfig):
    """Vision-tactile settings layered on the visual SmolVLA configuration."""

    use_tactile_encoder: bool = False
    tactile_encoder_path: str | None = None
    freeze_tactile_encoder: bool = True
    tactile_keys: tuple[str, ...] = ()
    tactile_embedding_dim: int = 512
    tactile_num_tokens: int = 4
    tactile_image_size: int = 224

    @classmethod
    def _validate_pretrained_metadata(cls, raw: dict[str, Any]) -> None:
        del raw

    @classmethod
    def _pretrained_extension_fields(cls, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "use_tactile_encoder": bool(raw.get("use_tactile_encoder", False)),
            "tactile_encoder_path": raw.get("tactile_encoder_path"),
            "freeze_tactile_encoder": bool(raw.get("freeze_tactile_encoder", True)),
            "tactile_keys": tuple(raw.get("tactile_keys") or ()),
            "tactile_embedding_dim": int(raw.get("tactile_embedding_dim", 512)),
            "tactile_num_tokens": int(raw.get("tactile_num_tokens", 4)),
            "tactile_image_size": int(raw.get("tactile_image_size", 224)),
        }

    @classmethod
    def _tuple_override_fields(cls) -> frozenset[str]:
        return super()._tuple_override_fields() | {"tactile_keys"}

    def _validate_extension_overrides(self) -> None:
        if not self.use_tactile_encoder:
            return
        if not self.tactile_encoder_path:
            raise ValueError("tactile_encoder_path is required when use_tactile_encoder=True")
        if not self.tactile_keys:
            raise ValueError("tactile_keys is required when use_tactile_encoder=True")
        if len(self.tactile_keys) != int(self.tactile_num_tokens):
            raise ValueError(
                "tactile_keys length must match tactile_num_tokens "
                f"({len(self.tactile_keys)} != {self.tactile_num_tokens})"
            )
        overlap = sorted(set(self.image_keys) & set(self.tactile_keys))
        if overlap:
            raise ValueError(f"tactile_keys must not also appear in image_keys: {overlap}")
