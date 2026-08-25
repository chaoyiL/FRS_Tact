import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from torch import nn

from .input_adapter import letterbox_and_normalize, select_deco_observation
from .model_factory import MODEL_TYPE, build_model


EXPORT_FORMAT = "sudo-upstream-deco-stage1-torchscript-v1"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_config(config: dict) -> None:
    if config.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"Expected {MODEL_TYPE!r}, got {config.get('model_type')!r}"
        )
    if config.get("use_task_condition", False):
        raise ValueError(
            "The two-input TorchScript deployment contract does not support "
            "task-conditioned policies"
        )
    if len(config.get("camera_names", [])) not in (2, 3):
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
    # Build a clean eval copy on the SAME device as the training policy so the
    # exported TorchScript keeps GPU-resident weights (the previous .cpu() call
    # forced every tensor to CPU during save, so loading always landed on CPU).
    device = next(policy.parameters()).device
    snapshot = build_model(config, load_backbone=False).to(device)
    snapshot.load_state_dict(
        {key: value for key, value in policy.state_dict().items()},
        strict=True,
    )
    return snapshot.eval()


def _trace(policy, stats, config, image_height, image_width):
    # Place trace inputs on the same device as the policy so the traced graph and
    # its saved tensors stay on GPU (CPU inputs would force the traced module to CPU).
    device = next(policy.parameters()).device
    deployment = UpstreamDECODeployment(policy, stats, config).eval().to(device)
    images = torch.zeros(
        1, len(config["camera_names"]), 3, image_height, image_width,
        device=device,
    )
    observation = torch.zeros(1, int(config["source_obs_dim"]), device=device)
    with torch.inference_mode():
        traced = torch.jit.trace(
            deployment, (images, observation), check_trace=False, strict=True
        )
        traced = torch.jit.freeze(traced.eval())
    return traced, images, observation


def _metadata(config, image_height, image_width, epoch, val_loss, source):
    action_mode = config.get("action_mode", "absolute")
    return {
        "format": EXPORT_FORMAT,
        "source": source,
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "dataset_id": config.get("dataset_id"),
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
        },
        "expected_sample_hz": config.get("expected_sample_hz"),
        "inference_steps": int(config["inference_steps"]),
        "stochastic": True,
    }


def _save_and_validate(traced, output_path, metadata, images, observation):
    metadata_json = json.dumps(metadata, indent=2, sort_keys=True)
    _atomic_save_torchscript(
        traced, output_path, {"deco_metadata.json": metadata_json}
    )
    extra = {"deco_metadata.json": ""}
    # Map the reloaded TorchScript onto the same device as the validation inputs
    # so GPU-resident weights stay on GPU (torch.jit.load defaults to CPU, which
    # would mismatch the GPU images/observation used for validation).
    device = images.device
    loaded = torch.jit.load(str(output_path), _extra_files=extra, map_location=device).eval()
    with torch.inference_mode():
        output = loaded(images, observation)
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
) -> dict:
    _validate_config(config)
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_height and image_width must be positive")
    output_path = Path(output_path)
    snapshot = _snapshot(policy, config)
    try:
        traced, images, observation = _trace(
            snapshot, stats, config, image_height, image_width
        )
        metadata = _metadata(
            config, image_height, image_width, epoch, val_loss, "training_loop"
        )
        return _save_and_validate(
            traced, output_path, metadata, images, observation
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
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    required = {"model", "config", "stats"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing)}")
    config = checkpoint["config"]
    _validate_config(config)
    policy = build_model(config, load_backbone=False)
    policy.load_state_dict(checkpoint["model"], strict=True)
    # Keep the policy on GPU when available so the exported TorchScript carries
    # GPU-resident tensors (the previous path left it on CPU after load).
    if torch.cuda.is_available():
        policy = policy.cuda()
    traced, images, observation = _trace(
        policy.eval(), checkpoint["stats"], config, image_height, image_width
    )
    metadata = _metadata(
        config,
        image_height,
        image_width,
        checkpoint.get("epoch", 0),
        checkpoint.get("val_loss", float("nan")),
        checkpoint_path.name,
    )
    metadata["source_checkpoint_sha256"] = sha256_file(checkpoint_path)
    return _save_and_validate(
        traced, Path(output_path), metadata, images, observation
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
