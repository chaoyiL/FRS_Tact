from __future__ import annotations

import dataclasses
import math
from typing import Any

CLIP_IMAGE_SIZE = 224

DEFAULT_GRU_HIDDEN_DIM = 256


DEFAULT_CONTRASTIVE_TEMPERATURE = 0.07


@dataclasses.dataclass(frozen=True)
class TactileClipConfig:
    embedding_dim: int = 512
    tactile_image_count: int = 2
    tactile_history: int = 0
    gru_hidden_dim: int = DEFAULT_GRU_HIDDEN_DIM
    # Fixed InfoNCE temperature (not learnable).
    temperature: float = DEFAULT_CONTRASTIVE_TEMPERATURE
    tactile_image_size: int = CLIP_IMAGE_SIZE

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}.")

    @property
    def temporal_length(self) -> int:
        """Number of tactile timesteps including the current frame."""

        if self.tactile_history < 0:
            raise ValueError(f"tactile_history must be non-negative, got {self.tactile_history}.")
        return 1 + self.tactile_history

    @property
    def uses_gru(self) -> bool:
        return self.tactile_history > 0

    @property
    def projection_in_dim(self) -> int:
        """Vision embedding plus tactile features for the future projection."""

        if self.uses_gru:
            return self.embedding_dim + self.tactile_image_count * self.gru_hidden_dim
        return self.embedding_dim * (1 + self.tactile_image_count)

    @property
    def logit_scale(self) -> float:
        """Constant multiplicative scale ``1 / temperature`` for contrastive logits."""

        return 1.0 / self.temperature


def tactile_clip_config_from_dict(data: dict[str, Any]) -> TactileClipConfig:
    """Build config from checkpoint metadata, ignoring obsolete fields."""

    known = {field.name for field in dataclasses.fields(TactileClipConfig)}
    filtered = {key: value for key, value in data.items() if key in known}
    if "temperature" not in filtered and "logit_scale_init" in data:
        # Legacy learnable-temperature init stored ``log(1 / T)``.
        filtered["temperature"] = float(math.exp(-float(data["logit_scale_init"])))
    return TactileClipConfig(**filtered)
