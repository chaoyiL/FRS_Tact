from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import jax

from train_smolvla.training import (
    JaxSmolVLATrainer,
    partition_params as partition_visual_params,
    promote_trainable_params_to_fp32 as promote_visual_trainable_params_to_fp32,
)

from .checkpoint import initialize_tactile_fusion_params, write_effective_config
from .configuration import VTSmolVLAConfig
from .lora import initialize_lora_params, is_trainable_parameter

Array = jax.Array
Params = dict[str, Array]


def partition_params(
    params: Mapping[str, Array],
    config: VTSmolVLAConfig,
) -> tuple[Params, Params]:
    return partition_visual_params(
        params,
        config,
        classifier=is_trainable_parameter,
    )


def promote_trainable_params_to_fp32(
    params: Mapping[str, Array],
    config: VTSmolVLAConfig,
) -> Params:
    return promote_visual_trainable_params_to_fp32(
        params,
        config,
        classifier=is_trainable_parameter,
    )


class VTJaxSmolVLATrainer(JaxSmolVLATrainer):
    """Visual trainer with VT parameter initialization and partition hooks."""

    config: VTSmolVLAConfig

    def _prepare_parameter_partition(
        self,
        params: Mapping[str, Array],
    ) -> tuple[Params, Params]:
        params = initialize_tactile_fusion_params(params, self.config, seed=self.seed)
        params = initialize_lora_params(params, self.config, seed=self.seed)
        params = promote_trainable_params_to_fp32(params, self.config)
        return partition_params(params, self.config)

    def _write_effective_config(self, destination: str | Path) -> Path:
        return write_effective_config(destination, self.config)


__all__ = [
    "VTJaxSmolVLATrainer",
    "partition_params",
    "promote_trainable_params_to_fp32",
]
