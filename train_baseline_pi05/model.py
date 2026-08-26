"""A small, direct tactile-conditioned action decoder implemented in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from train_baseline_pi05.config import DecoderTrainConfig, TACTILE_KEYS


@dataclass(frozen=True)
class DirectDecoderConfig:
    """The fixed architecture contract for direct Pi0.5 decoder experiments."""

    action_horizon: int = 50
    action_dim: int = 20
    tactile_dim: int = 512
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    tactile_keys: tuple[str, ...] = TACTILE_KEYS

    @classmethod
    def from_train_config(cls, config: DecoderTrainConfig) -> "DirectDecoderConfig":
        result = cls(
            action_horizon=config.action_horizon,
            action_dim=config.action_dim,
            tactile_dim=config.tactile_dim,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            tactile_keys=config.tactile_keys,
        )
        result.validate()
        return result

    def validate(self) -> None:
        expected_integers = {
            "action_horizon": 50,
            "action_dim": 20,
            "tactile_dim": 512,
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
        }
        for name, required in expected_integers.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer for the direct decoder contract.")
            if value != required:
                raise ValueError(f"{name} must be {required!r} for the direct decoder contract.")
        if not isinstance(self.dropout, float):
            raise ValueError("dropout must be a float for the direct decoder contract.")
        if self.dropout != 0.1:
            raise ValueError("dropout must be 0.1 for the direct decoder contract.")
        if not isinstance(self.tactile_keys, tuple) or any(
            not isinstance(key, str) for key in self.tactile_keys
        ):
            raise ValueError("tactile_keys must be a tuple of strings for the direct decoder contract.")
        if self.tactile_keys != TACTILE_KEYS:
            raise ValueError("tactile_keys must use canonical order for the direct decoder contract.")

    def to_primitive(self) -> dict[str, int | float | list[str]]:
        self.validate()
        return {
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "tactile_dim": self.tactile_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "tactile_keys": list(self.tactile_keys),
        }

    @classmethod
    def from_primitive(cls, raw: object) -> "DirectDecoderConfig":
        if not isinstance(raw, dict):
            raise ValueError("decoder_config must be a dictionary.")
        expected_keys = {
            "action_horizon",
            "action_dim",
            "tactile_dim",
            "d_model",
            "nhead",
            "num_layers",
            "dim_feedforward",
            "dropout",
            "tactile_keys",
        }
        if set(raw) != expected_keys:
            raise ValueError("decoder_config has an invalid schema.")
        try:
            config = cls(
                action_horizon=raw["action_horizon"],
                action_dim=raw["action_dim"],
                tactile_dim=raw["tactile_dim"],
                d_model=raw["d_model"],
                nhead=raw["nhead"],
                num_layers=raw["num_layers"],
                dim_feedforward=raw["dim_feedforward"],
                dropout=raw["dropout"],
                tactile_keys=tuple(raw["tactile_keys"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("decoder_config has invalid values.") from exc
        config.validate()
        return config


class DirectTactileActionDecoder(nn.Module):
    """Predict full action chunks from coarse actions and current tactile tokens."""

    def __init__(self, config: DirectDecoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.coarse_projection = nn.Linear(config.action_dim, config.d_model)
        self.tactile_projection = nn.Linear(config.tactile_dim, config.d_model)
        self.action_position = nn.Parameter(torch.empty(config.action_horizon, config.d_model))
        self.sensor_identity = nn.Parameter(torch.empty(len(config.tactile_keys), config.d_model))
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.action_dim)
        nn.init.normal_(self.action_position, std=0.02)
        nn.init.normal_(self.sensor_identity, std=0.02)

    @staticmethod
    def normalize_tactile(tactile: Tensor) -> Tensor:
        """RMS-normalize each sensor token independently before projection."""
        working = tactile.to(dtype=torch.float32)
        rms = working.square().mean(dim=-1, keepdim=True).add(1e-8).sqrt()
        return working / rms

    def forward(self, coarse: Tensor, tactile: Tensor) -> Tensor:
        if coarse.ndim != 3 or tuple(coarse.shape[1:]) != (
            self.config.action_horizon,
            self.config.action_dim,
        ):
            raise ValueError(
                f"coarse must have shape [B,{self.config.action_horizon},{self.config.action_dim}]."
            )
        if tactile.ndim != 3 or tuple(tactile.shape[1:]) != (
            len(self.config.tactile_keys),
            self.config.tactile_dim,
        ):
            raise ValueError(
                f"tactile must have shape [B,{len(self.config.tactile_keys)},{self.config.tactile_dim}]."
            )
        if coarse.shape[0] != tactile.shape[0]:
            raise ValueError("coarse and tactile batch dimensions must agree.")

        target = self.coarse_projection(coarse.to(dtype=self.coarse_projection.weight.dtype))
        target = target + self.action_position.unsqueeze(0)
        tactile_tokens = self.normalize_tactile(tactile).to(dtype=self.tactile_projection.weight.dtype)
        memory = self.tactile_projection(tactile_tokens)
        memory = memory + self.sensor_identity.unsqueeze(0)
        predicted = self.output_head(self.output_norm(self.decoder(target, memory)))
        if not torch.isfinite(predicted).all():
            raise ValueError("decoder output must be finite.")
        return predicted


def masked_smooth_l1(predicted: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    """Average Smooth L1 across only valid action tokens or elements."""
    if predicted.shape != target.shape:
        raise ValueError("predicted and target must have identical shapes.")
    if valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must have boolean dtype.")
    if valid_mask.shape == predicted.shape[:-1]:
        expanded_mask = valid_mask.unsqueeze(-1).expand_as(predicted)
    elif valid_mask.shape == predicted.shape:
        expanded_mask = valid_mask
    else:
        raise ValueError("valid_mask must match action tokens or action elements.")
    count = expanded_mask.sum()
    if count.item() == 0:
        raise ValueError("valid_mask must contain at least one valid element.")
    losses = F.smooth_l1_loss(predicted, target, reduction="none")
    return (losses * expanded_mask).sum() / count
