"""Arm-wise PCA projection for the four pick-tube tactile embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


TACTILE_SENSOR_ORDER = (
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)
SENSOR_EMBEDDING_DIM = 512
SENSORS_PER_ARM = 2
ARM_COUNT = 2
ARM_INPUT_DIM = SENSOR_EMBEDDING_DIM * SENSORS_PER_ARM
COMPONENTS_PER_ARM = 15
RAW_TACTILE_DIM = ARM_INPUT_DIM * ARM_COUNT
REDUCED_TACTILE_DIM = COMPONENTS_PER_ARM * ARM_COUNT
PCA_FORMAT_VERSION = 1


def _contains_complex_values(values: np.ndarray) -> bool:
    return np.iscomplexobj(values) or (
        values.dtype == object and any(np.iscomplexobj(value) for value in values.flat)
    )


def group_tactile_embeddings(values: np.ndarray) -> np.ndarray:
    """Return ``[..., 2, 1024]`` values grouped by robot arm.

    The input follows ``TACTILE_SENSOR_ORDER``. ``left_0 + right_0`` form the
    robot-0 (left-arm) vector and ``left_1 + right_1`` form the robot-1
    (right-arm) vector.
    """
    values = np.asarray(values)
    if values.shape[-2:] == (len(TACTILE_SENSOR_ORDER), SENSOR_EMBEDDING_DIM):
        arm_0 = np.concatenate((values[..., 0, :], values[..., 1, :]), axis=-1)
        arm_1 = np.concatenate((values[..., 2, :], values[..., 3, :]), axis=-1)
    elif values.shape[-1:] == (RAW_TACTILE_DIM,):
        reshaped = values.reshape(
            *values.shape[:-1], len(TACTILE_SENSOR_ORDER), SENSOR_EMBEDDING_DIM
        )
        return group_tactile_embeddings(reshaped)
    else:
        raise ValueError(
            "Expected tactile embeddings ending in [4,512] or [2048], "
            f"got {values.shape}"
        )
    return np.stack((arm_0, arm_1), axis=-2)


class BimanualTactilePCA(nn.Module):
    """Apply two independent arm-wise PCA projections.

    The number of components is inferred from the artifact. Existing 2x15
    artifacts therefore remain compatible, while ablations can use 2x8,
    2x30, or another positive component count.
    """

    def __init__(self, means: np.ndarray, components: np.ndarray) -> None:
        super().__init__()
        means = np.asarray(means)
        components = np.asarray(components)
        if _contains_complex_values(means):
            raise ValueError("PCA means must contain real values")
        if _contains_complex_values(components):
            raise ValueError("PCA components must contain real values")
        means = np.asarray(means, dtype=np.float32)
        components = np.asarray(components, dtype=np.float32)
        if means.shape != (ARM_COUNT, ARM_INPUT_DIM):
            raise ValueError(
                f"PCA means must be [{ARM_COUNT},{ARM_INPUT_DIM}], got {means.shape}"
            )
        if (
            components.ndim != 3
            or components.shape[0] != ARM_COUNT
            or components.shape[1] < 1
            or components.shape[2] != ARM_INPUT_DIM
        ):
            raise ValueError(
                "PCA components must be "
                f"[{ARM_COUNT},N,{ARM_INPUT_DIM}] with N >= 1, "
                f"got {components.shape}"
            )
        if not np.isfinite(means).all():
            raise ValueError("PCA means must contain only finite values")
        if not np.isfinite(components).all():
            raise ValueError("PCA components must contain only finite values")
        self.register_buffer("means", torch.from_numpy(means))
        self.register_buffer("components", torch.from_numpy(components))

    @property
    def components_per_arm(self) -> int:
        return int(self.components.shape[1])

    @property
    def output_dim(self) -> int:
        return self.components_per_arm * ARM_COUNT

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "BimanualTactilePCA":
        path = Path(path)
        with np.load(path, allow_pickle=False) as artifact:
            version = int(np.asarray(artifact["format_version"]).item())
            if version != PCA_FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported tactile PCA format {version}; expected {PCA_FORMAT_VERSION}"
                )
            sensor_order = tuple(str(value) for value in artifact["sensor_order"].tolist())
            if sensor_order != TACTILE_SENSOR_ORDER:
                raise ValueError(
                    f"Tactile PCA sensor order {sensor_order} does not match {TACTILE_SENSOR_ORDER}"
                )
            model = cls(artifact["means"], artifact["components"])
        return model.eval().to(device)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-2:] == (len(TACTILE_SENSOR_ORDER), SENSOR_EMBEDDING_DIM):
            arm_0 = torch.cat((values[..., 0, :], values[..., 1, :]), dim=-1)
            arm_1 = torch.cat((values[..., 2, :], values[..., 3, :]), dim=-1)
            grouped = torch.stack((arm_0, arm_1), dim=-2)
        elif values.shape[-1:] == (RAW_TACTILE_DIM,):
            reshaped = values.reshape(
                *values.shape[:-1], len(TACTILE_SENSOR_ORDER), SENSOR_EMBEDDING_DIM
            )
            arm_0 = torch.cat((reshaped[..., 0, :], reshaped[..., 1, :]), dim=-1)
            arm_1 = torch.cat((reshaped[..., 2, :], reshaped[..., 3, :]), dim=-1)
            grouped = torch.stack((arm_0, arm_1), dim=-2)
        else:
            raise ValueError(
                "Expected tactile embeddings ending in [4,512] or [2048], "
                f"got {tuple(values.shape)}"
            )
        grouped = grouped.to(device=self.means.device, dtype=self.means.dtype)
        reduced = torch.einsum(
            "...ad,acd->...ac",
            grouped - self.means,
            self.components,
        )
        return reduced.flatten(start_dim=-2)

    def transform_numpy(self, values: np.ndarray) -> np.ndarray:
        grouped = group_tactile_embeddings(values).astype(np.float32, copy=False)
        means = self.means.detach().cpu().numpy()
        components = self.components.detach().cpu().numpy()
        reduced = np.einsum(
            "...ad,acd->...ac",
            grouped - means,
            components,
            optimize=True,
        )
        return reduced.reshape(*reduced.shape[:-2], self.output_dim)


def save_tactile_pca(
    path: str | Path,
    *,
    means: np.ndarray,
    components: np.ndarray,
    explained_variance_ratio: np.ndarray,
    sample_count: int,
) -> None:
    """Save a portable PCA artifact consumed by conversion and deployment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.asarray(PCA_FORMAT_VERSION, dtype=np.int64),
        sensor_order=np.asarray(TACTILE_SENSOR_ORDER),
        means=np.asarray(means, dtype=np.float32),
        components=np.asarray(components, dtype=np.float32),
        explained_variance_ratio=np.asarray(explained_variance_ratio, dtype=np.float32),
        sample_count=np.asarray(sample_count, dtype=np.int64),
    )
