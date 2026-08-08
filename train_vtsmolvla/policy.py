from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from train_smolvla.policy import JaxSmolVLAPolicy

from .checkpoint import load_config
from .modeling import VTJaxSmolVLA
from .preprocessing import VTJaxSmolVLAPreprocessor


class VTJaxSmolVLAPolicy(JaxSmolVLAPolicy):
    """Stateful VT policy reusing the visual policy lifecycle and queueing."""

    def _load_config(self, checkpoint: Path):
        return load_config(checkpoint)

    def _make_model(self, config):
        return VTJaxSmolVLA(config)

    def _make_preprocessor(
        self,
        checkpoint: Path,
        config,
        *,
        rename_map: Mapping[str, str] | None,
        local_files_only: bool,
    ):
        return VTJaxSmolVLAPreprocessor(
            checkpoint,
            config,
            rename_map=rename_map,
            local_files_only=local_files_only,
        )

    def _sample_prepared_batch(self, params, batch, rng, **kwargs):
        return self.model.sample_actions(
            params,
            batch["images"],
            batch["image_masks"],
            batch["language_tokens"],
            batch["language_masks"],
            batch["state"],
            rng,
            tactile_images=batch.get("tactile_images"),
            tactile_embeddings=batch.get("tactile_embeddings"),
            tactile_masks=batch.get("tactile_masks"),
            **kwargs,
        )
