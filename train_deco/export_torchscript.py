import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import torch
from torch import nn

from .input_adapter import (
    IMAGE_MEAN,
    IMAGE_STD,
    letterbox_and_normalize,
    letterbox_tactile_images,
    select_deco_observation,
)
from .lerobot_vision_dataset import TACTILE_NAMES
from .model_factory import (
    MODEL_TYPE,
    STAGE2_MODEL_TYPE,
    build_model,
    build_stage2_model,
)
from .stage2_initialization import configure_stage2_trainability


EXPORT_FORMAT = "sudo-upstream-deco-stage1-torchscript-v1"
STAGE2_EXPORT_FORMAT = "sudo-upstream-deco-stage2-torchscript-v1"
STAGE2_CHECKPOINT_SCHEMA_VERSION = 1
TACTILE_TARGET_SIZE = (224, 224)


class UpstreamDECODeployment(nn.Module):
    """Raw source-data boundary around the unmodified upstream DECO policy."""

    def __init__(self, policy: nn.Module, stats: dict, config: dict):
        super().__init__()
        self.policy = policy
        self.image_size = int(config["image_size"])
        self.num_cameras = len(config["camera_names"])
        self.register_buffer(
            "observation_mean",
            torch.as_tensor(stats["observation_mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "observation_std",
            torch.as_tensor(stats["observation_std"], dtype=torch.float32),
        )
        self.register_buffer(
            "observation_indices",
            torch.as_tensor(config["observation_indices"], dtype=torch.long),
        )
        self.register_buffer(
            "action_mean", torch.as_tensor(stats["action_mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "action_std", torch.as_tensor(stats["action_std"], dtype=torch.float32)
        )

    def forward(self, images: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        normalized_source = (
            observation - self.observation_mean
        ) / self.observation_std
        deco_observation = select_deco_observation(
            normalized_source, self.observation_indices
        )
        normalized_images = letterbox_and_normalize(images, self.image_size)
        if self.num_cameras == 2:
            normalized_action = self.policy(
                normalized_images[:, 0],
                normalized_images[:, 1],
                obs=deco_observation,
                training=False,
            )
        else:
            normalized_action = self.policy(
                normalized_images[:, 0],
                normalized_images[:, 1],
                normalized_images[:, 2],
                obs=deco_observation,
                training=False,
            )
        return normalized_action * self.action_std + self.action_mean


class Stage2DECODeployment(UpstreamDECODeployment):
    """Raw six-image boundary around the tactile-image Stage2 policy."""

    def forward(
        self,
        images: torch.Tensor,
        tactile_images: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        normalized_source = (
            observation - self.observation_mean
        ) / self.observation_std
        deco_observation = select_deco_observation(
            normalized_source, self.observation_indices
        )
        normalized_images = letterbox_and_normalize(images, self.image_size)
        normalized_tactile = letterbox_tactile_images(
            tactile_images, TACTILE_TARGET_SIZE
        )
        normalized_action = self.policy(
            normalized_images[:, 0],
            normalized_images[:, 1],
            obs=deco_observation,
            training=False,
            tactile_images=normalized_tactile,
        )
        return normalized_action * self.action_std + self.action_mean


@torch.jit.interface
class _Stage2DeploymentInterface(nn.Module):
    def forward(
        self,
        images: torch.Tensor,
        tactile_images: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        pass


class _FixedShapeStage2Deployment(nn.Module):
    """Scripted guard around a traced Stage2 deployment graph."""

    deployment: _Stage2DeploymentInterface

    def __init__(
        self,
        deployment: nn.Module,
        image_height: int,
        image_width: int,
    ) -> None:
        super().__init__()
        self.deployment = deployment
        self.image_height = image_height
        self.image_width = image_width

    def forward(
        self,
        images: torch.Tensor,
        tactile_images: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        if (
            images.size(-2) != self.image_height
            or images.size(-1) != self.image_width
        ):
            raise RuntimeError(
                "Stage2 visual spatial shape does not match the fixed "
                "TorchScript export contract"
            )
        if (
            tactile_images.size(-2) != self.image_height
            or tactile_images.size(-1) != self.image_width
        ):
            raise RuntimeError(
                "Stage2 tactile spatial shape does not match the fixed "
                "TorchScript export contract"
            )
        return self.deployment(images, tactile_images, observation)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_config(config: dict) -> None:
    model_type = config.get("model_type")
    if model_type not in (MODEL_TYPE, STAGE2_MODEL_TYPE):
        raise ValueError(
            f"Expected {MODEL_TYPE!r} or {STAGE2_MODEL_TYPE!r}, got {model_type!r}"
        )
    if config.get("use_task_condition", False):
        raise ValueError(
            "The TorchScript deployment contract does not support "
            "task-conditioned policies"
        )
    camera_count = len(config.get("camera_names", []))
    if model_type == STAGE2_MODEL_TYPE and camera_count != 2:
        raise ValueError("Stage2 TorchScript export requires exactly two cameras")
    if model_type == MODEL_TYPE and camera_count not in (2, 3):
        raise ValueError("TorchScript export requires two or three camera names")
    action_mode = config.get("action_mode", "absolute")
    if action_mode not in {
        "absolute",
        "delta",
        "tcp_delta_absolute_gripper",
    }:
        raise ValueError(
            f"Unsupported TorchScript action_mode: {config.get('action_mode')!r}"
        )


def _validate_stage2_checkpoint(checkpoint: dict, config: dict) -> dict:
    if (
        checkpoint.get("stage") != 2
        or checkpoint.get("model_type") != STAGE2_MODEL_TYPE
        or config.get("model_type") != STAGE2_MODEL_TYPE
    ):
        raise ValueError("Stage2 export requires a Stage2 tactile-image checkpoint")
    schema = checkpoint.get("checkpoint_schema_version")
    if schema != STAGE2_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Stage2 checkpoint schema is incompatible: "
            f"expected {STAGE2_CHECKPOINT_SCHEMA_VERSION}, got {schema!r}"
        )
    metadata = checkpoint.get("stage2_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Stage2 checkpoint is missing stage2_metadata")
    required = {
        "model_type",
        "tactile_field_order",
        "tactile_encoder",
        "tactile_adapter_rank",
        "gate_values",
        "parameter_categories",
        "parameter_counts",
        "stage1_checkpoint",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Stage2 checkpoint metadata is missing: {missing}")
    if metadata["model_type"] != STAGE2_MODEL_TYPE:
        raise ValueError("Stage2 checkpoint metadata model_type is incompatible")
    tactile_order = list(TACTILE_NAMES)
    if (
        metadata["tactile_field_order"] != tactile_order
        or config.get("tactile_field_order") != tactile_order
    ):
        raise ValueError("Stage2 checkpoint tactile field order is incompatible")
    adapter_rank = metadata["tactile_adapter_rank"]
    if (
        isinstance(adapter_rank, bool)
        or not isinstance(adapter_rank, int)
        or adapter_rank <= 0
        or int(config.get("tactile_adapter_rank", 0)) != adapter_rank
    ):
        raise ValueError("Stage2 checkpoint adapter rank is incompatible")
    encoder = metadata["tactile_encoder"]
    if (
        not isinstance(encoder, dict)
        or encoder.get("architecture") != "resnet18"
        or encoder.get("embedding_dim") != 512
    ):
        raise ValueError("Stage2 checkpoint tactile encoder contract is incompatible")
    for name in ("source_sha256", "artifact_sha256"):
        digest = encoder.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Stage2 checkpoint tactile encoder {name} is invalid")
    for name in ("artifact_path", "metadata_path"):
        path = encoder.get(name)
        if not isinstance(path, str) or not path.strip():
            raise ValueError(
                f"Stage2 checkpoint tactile encoder {name} is invalid"
            )
    gates = metadata["gate_values"]
    if not isinstance(gates, dict) or not gates:
        raise ValueError("Stage2 checkpoint gate values are missing")
    if any(
        not isinstance(name, str)
        or not name.endswith(".tactile_gate")
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for name, value in gates.items()
    ):
        raise ValueError("Stage2 checkpoint gate values are invalid")
    stage1 = metadata["stage1_checkpoint"]
    if not isinstance(stage1, dict):
        raise ValueError("Stage2 checkpoint Stage1 provenance is invalid")
    stage1_path = stage1.get("path")
    stage1_digest = stage1.get("sha256")
    if not isinstance(stage1_path, str) or not stage1_path.strip():
        raise ValueError("Stage2 checkpoint Stage1 provenance path is invalid")
    if (
        not isinstance(stage1_digest, str)
        or len(stage1_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in stage1_digest
        )
    ):
        raise ValueError("Stage2 checkpoint Stage1 provenance digest is invalid")
    categories = metadata["parameter_categories"]
    if not isinstance(categories, dict):
        raise ValueError("Stage2 checkpoint parameter_categories is invalid")
    for boundary in ("trainable", "frozen"):
        grouped = categories.get(boundary)
        if not isinstance(grouped, dict):
            raise ValueError(
                f"Stage2 checkpoint parameter_categories.{boundary} is invalid"
            )
        for names in grouped.values():
            if (
                not isinstance(names, list)
                or not all(isinstance(name, str) and name for name in names)
                or len(names) != len(set(names))
            ):
                raise ValueError(
                    f"Stage2 checkpoint parameter_categories.{boundary} is invalid"
                )
    counts = metadata["parameter_counts"]
    if not isinstance(counts, dict):
        raise ValueError("Stage2 checkpoint parameter_counts is invalid")
    total = counts.get("total")
    trainable = counts.get("trainable")
    if (
        isinstance(total, bool)
        or isinstance(trainable, bool)
        or not isinstance(total, int)
        or not isinstance(trainable, int)
        or total <= 0
        or trainable <= 0
        or trainable >= total
    ):
        raise ValueError("Stage2 checkpoint parameter_counts is invalid")
    return metadata


def _build_policy(config: dict) -> nn.Module:
    if config.get("model_type") == STAGE2_MODEL_TYPE:
        return build_stage2_model(config, load_backbone=False)
    return build_model(config, load_backbone=False)


def _validate_stage2_gates(policy: nn.Module, metadata: dict) -> None:
    actual = {
        name: float(parameter.detach().cpu())
        for name, parameter in policy.named_parameters()
        if name.endswith(".tactile_gate")
    }
    expected = metadata["gate_values"]
    if set(actual) != set(expected):
        raise ValueError("Stage2 checkpoint gate names disagree with the model")
    for name, value in expected.items():
        if not isinstance(value, (int, float)) or not torch.isclose(
            torch.tensor(actual[name]),
            torch.tensor(float(value)),
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError(f"Stage2 checkpoint gate value disagrees for {name!r}")


def _validate_stage2_parameter_inventory(
    policy: nn.Module,
    metadata: dict,
) -> None:
    report = configure_stage2_trainability(policy)
    expected_categories = {
        "trainable": {
            category: list(names)
            for category, names in report.trainable_by_category.items()
        },
        "frozen": {
            category: list(names)
            for category, names in report.frozen_by_category.items()
        },
    }
    if metadata["parameter_categories"] != expected_categories:
        raise ValueError(
            "Stage2 checkpoint parameter categories disagree with the model"
        )
    expected_counts = {
        "total": report.total_parameters,
        "trainable": report.trainable_parameters,
    }
    if metadata["parameter_counts"] != expected_counts:
        raise ValueError(
            "Stage2 checkpoint parameter counts disagree with the model"
        )


def _atomic_save_torchscript(module, output_path: Path, extra_files: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        torch.jit.save(module, str(temporary), _extra_files=extra_files)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _snapshot(policy: nn.Module, config: dict) -> nn.Module:
    # Build and trace an independent copy without moving or switching the mode
    # of the live training policy. Tracing must use the deployment device because
    # TorchScript specializes input-derived device expressions into literals.
    device = next(policy.parameters()).device
    snapshot = _build_policy(config).to(device)
    snapshot.load_state_dict(
        {
            key: value.detach().to(device)
            for key, value in policy.state_dict().items()
        },
        strict=True,
    )
    return snapshot.eval()


def _trace(policy, stats, config, image_height, image_width):
    device = next(policy.parameters()).device
    stage2 = config.get("model_type") == STAGE2_MODEL_TYPE
    deployment_type = Stage2DECODeployment if stage2 else UpstreamDECODeployment
    deployment = deployment_type(policy, stats, config).eval().to(device)
    images = torch.zeros(
        1, len(config["camera_names"]), 3, image_height, image_width,
        device=device,
    )
    observation = torch.zeros(
        1, int(config["source_obs_dim"]), device=device
    )
    inputs = (images, observation)
    if stage2:
        tactile_images = torch.zeros(
            1, 4, 3, image_height, image_width, device=device
        )
        inputs = (images, tactile_images, observation)
    with torch.inference_mode():
        traced = torch.jit.trace(
            deployment, inputs, check_trace=False, strict=True
        )
        traced = torch.jit.freeze(traced.eval())
        if stage2:
            traced = torch.jit.script(
                _FixedShapeStage2Deployment(
                    traced,
                    image_height,
                    image_width,
                )
            )
            traced = torch.jit.freeze(traced.eval())
    return traced, inputs


def _metadata(
    config,
    stats,
    image_height,
    image_width,
    epoch,
    val_loss,
    source,
    *,
    checkpoint_schema_version=None,
    stage2_metadata=None,
):
    action_mode = config.get("action_mode", "absolute")
    stage2 = config.get("model_type") == STAGE2_MODEL_TYPE
    metadata = {
        "format": STAGE2_EXPORT_FORMAT if stage2 else EXPORT_FORMAT,
        "source": source,
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "dataset_id": config.get("dataset_id"),
        "state_action_profile": config.get("state_action_profile"),
        "controlled_arms": config.get("controlled_arms"),
        "upstream_model": "train_deco.models.deco.deco.DECO",
        "camera_names": config["camera_names"],
        "input": {
            "images": [
                1, len(config["camera_names"]), 3, image_height, image_width
            ],
            "images_dtype": "float32",
            "images_range": [0.0, 1.0],
            "observation": [1, int(config["source_obs_dim"])],
            "observation_space": "raw source state; mapping and normalization embedded",
            "state_layout": config.get("state_layout"),
            "state_columns": config.get("state_columns"),
        },
        "output": {
            "action": [1, int(config["chunk_size"]), int(config["action_dim"])],
            "action_mode": action_mode,
            "action_space": {
                "delta": "denormalized delta robot action (target-current)",
                "absolute": "denormalized absolute robot action",
                "tcp_delta_absolute_gripper": (
                    "denormalized per-arm TCP-frame delta xyz + Rotation-6D "
                    "matrix columns + absolute gripper width"
                ),
            }[action_mode],
            "rotation_representation": config.get("rotation_representation"),
            "terminal_action_policy": config.get("terminal_action_policy"),
            "action_columns": config.get("action_columns"),
            "gripper_mode": config.get("gripper_mode"),
        },
        "normalization": {
            "embedded": True,
            "statistics_source": config.get("statistics_source"),
            "statistics": stats,
        },
        "expected_sample_hz": config.get("expected_sample_hz"),
        "inference_steps": int(config["inference_steps"]),
        "stochastic": True,
    }
    if stage2:
        if stage2_metadata is None:
            raise ValueError("Stage2 export requires checkpoint provenance metadata")
        metadata["input"]["tactile_images"] = [
            1, 4, 3, image_height, image_width
        ]
        metadata["input"]["tactile_images_dtype"] = "float32"
        metadata["input"]["tactile_images_range"] = [0.0, 1.0]
        metadata["input"]["stream_order"] = [
            *config["camera_names"],
            *stage2_metadata["tactile_field_order"],
        ]
        metadata["input"]["spatial_shape_contract"] = "fixed"
        metadata["preprocessing"] = {
            "visual": {
                "resize": "aspect-preserving-letterbox",
                "target_size": [int(config["image_size"])] * 2,
                "padding_value": 128.0 / 255.0,
                "normalization": "imagenet",
                "mean": list(IMAGE_MEAN),
                "std": list(IMAGE_STD),
            },
            "tactile": {
                "resize": "aspect-preserving-letterbox",
                "target_size": list(TACTILE_TARGET_SIZE),
                "padding_value": 0.0,
                "normalization": None,
                "range": [0.0, 1.0],
            },
        }
        metadata["checkpoint_schema_version"] = int(checkpoint_schema_version)
        for name in (
            "tactile_field_order",
            "tactile_encoder",
            "tactile_adapter_rank",
            "gate_values",
            "stage1_checkpoint",
        ):
            metadata[name] = stage2_metadata[name]
    return metadata


def _save_and_validate(traced, output_path, metadata, inputs):
    metadata_json = json.dumps(metadata, indent=2, sort_keys=True)
    _atomic_save_torchscript(
        traced, output_path, {"deco_metadata.json": metadata_json}
    )
    extra = {"deco_metadata.json": ""}
    target_device = inputs[0].device
    loaded = torch.jit.load(
        str(output_path), _extra_files=extra, map_location=target_device
    ).eval()
    with torch.inference_mode():
        output = loaded(*inputs)
    expected = tuple(metadata["output"]["action"])
    if tuple(output.shape) != expected or not torch.isfinite(output).all():
        raise ValueError(
            f"TorchScript validation failed: shape={tuple(output.shape)}, expected={expected}"
        )
    metadata["torchscript_sha256"] = sha256_file(output_path)
    metadata["output_path"] = str(output_path)
    _atomic_write_text(
        output_path.with_suffix(output_path.suffix + ".json"),
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )
    return metadata


def export_policy(
    policy,
    stats: dict,
    config: dict,
    output_path: str | Path,
    image_height: int,
    image_width: int,
    epoch: int,
    val_loss: float,
    *,
    checkpoint_schema_version: int | None = None,
    stage2_metadata: dict | None = None,
) -> dict:
    _validate_config(config)
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_height and image_width must be positive")
    output_path = Path(output_path)
    snapshot = _snapshot(policy, config)
    try:
        traced, inputs = _trace(
            snapshot, stats, config, image_height, image_width
        )
        metadata = _metadata(
            config,
            stats,
            image_height,
            image_width,
            epoch,
            val_loss,
            "training_loop",
            checkpoint_schema_version=checkpoint_schema_version,
            stage2_metadata=stage2_metadata,
        )
        return _save_and_validate(
            traced, output_path, metadata, inputs
        )
    finally:
        del snapshot


def copy_torchscript_artifact(source: str | Path, destination: str | Path) -> None:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = source.with_suffix(source.suffix + ".json")
    if sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        metadata["output_path"] = str(destination)
        _atomic_write_text(
            destination.with_suffix(destination.suffix + ".json"),
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        )


def export_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path,
    image_height: int,
    image_width: int,
    device: str = "cpu",
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA export device {target_device} is unavailable")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    required = {"model", "config", "stats"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing)}")
    config = checkpoint["config"]
    _validate_config(config)
    stage2_metadata = None
    if config.get("model_type") == STAGE2_MODEL_TYPE:
        stage2_metadata = _validate_stage2_checkpoint(checkpoint, config)
    policy = _build_policy(config).to(target_device)
    policy.load_state_dict(checkpoint["model"], strict=True)
    if stage2_metadata is not None:
        _validate_stage2_gates(policy, stage2_metadata)
        _validate_stage2_parameter_inventory(policy, stage2_metadata)
    traced, inputs = _trace(
        policy.eval(), checkpoint["stats"], config, image_height, image_width
    )
    metadata = _metadata(
        config,
        checkpoint["stats"],
        image_height,
        image_width,
        checkpoint.get("epoch", 0),
        checkpoint.get("val_loss", float("nan")),
        checkpoint_path.name,
        checkpoint_schema_version=checkpoint.get("checkpoint_schema_version"),
        stage2_metadata=stage2_metadata,
    )
    metadata["source_checkpoint_sha256"] = sha256_file(checkpoint_path)
    return _save_and_validate(
        traced, Path(output_path), metadata, inputs
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-height", type=int, default=208)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps({
        "event": "torchscript_export_complete",
        **export_checkpoint(
            args.checkpoint, args.output, args.image_height,
            args.image_width, args.device,
        ),
    }))


if __name__ == "__main__":
    main()
