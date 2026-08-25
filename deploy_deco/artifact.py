"""Validation and loading for self-contained DECO TorchScript artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

EXPORT_FORMAT = "sudo-upstream-deco-stage1-torchscript-v1"
ACTION_MODE = "tcp_delta_absolute_gripper"
ROTATION_REPRESENTATION = "rotation_6d_matrix_columns"
STATE_LAYOUT = "relative_start_pose6d_gripper_plus_left_relative_right"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecar_path(torchscript_path: Path) -> Path:
    return torchscript_path.with_suffix(torchscript_path.suffix + ".json")


def _shape(metadata: Mapping[str, Any], section: str, name: str) -> tuple[int, ...]:
    value = metadata.get(section)
    if not isinstance(value, Mapping):
        raise ValueError(f"DECO metadata is missing mapping {section!r}")
    raw = value.get(name)
    if not isinstance(raw, list) or not raw or any(type(item) is not int for item in raw):
        raise ValueError(f"DECO metadata {section}.{name} must be an integer list")
    return tuple(raw)


def validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fixed VB3 DECO Stage 1 deployment contract."""
    if metadata.get("format") != EXPORT_FORMAT:
        raise ValueError(f"unsupported DECO artifact format: {metadata.get('format')!r}")
    cameras = metadata.get("camera_names")
    expected_cameras = [
        "observation.images.camera0",
        "observation.images.camera1",
    ]
    if cameras != expected_cameras:
        raise ValueError(f"DECO camera order must be {expected_cameras}, got {cameras!r}")
    images = _shape(metadata, "input", "images")
    observation = _shape(metadata, "input", "observation")
    action = _shape(metadata, "output", "action")
    if len(images) != 5 or images[:3] != (1, 2, 3) or min(images[3:]) <= 0:
        raise ValueError(f"DECO images must have shape [1,2,3,H,W], got {images}")
    if observation != (1, 20):
        raise ValueError(f"DECO observation must have shape [1,20], got {observation}")
    if len(action) != 3 or action[0] != 1 or action[1] <= 0 or action[2] != 20:
        raise ValueError(f"DECO action must have shape [1,H,20], got {action}")
    input_contract = metadata["input"]
    output_contract = metadata["output"]
    if input_contract.get("images_dtype") != "float32":
        raise ValueError("DECO TorchScript images_dtype must be float32")
    if input_contract.get("images_range") != [0.0, 1.0]:
        raise ValueError("DECO TorchScript images_range must be [0.0, 1.0]")
    if input_contract.get("state_layout") != STATE_LAYOUT:
        raise ValueError("DECO state layout does not match the VB3 7+7+6 contract")
    if output_contract.get("action_mode") != ACTION_MODE:
        raise ValueError("DECO action_mode must be tcp_delta_absolute_gripper")
    if output_contract.get("rotation_representation") != ROTATION_REPRESENTATION:
        raise ValueError("DECO rotation representation must use matrix columns")
    if output_contract.get("gripper_mode") != "absolute":
        raise ValueError("DECO gripper mode must be absolute")
    sample_hz = metadata.get("expected_sample_hz")
    if isinstance(sample_hz, bool) or not isinstance(sample_hz, (int, float)) or sample_hz <= 0:
        raise ValueError("DECO expected_sample_hz must be positive")
    if metadata.get("normalization", {}).get("embedded") is not True:
        raise ValueError("DECO deployment requires embedded normalization")
    if metadata.get("stochastic") is not True:
        raise ValueError("DECO Stage 1 metadata must declare stochastic inference")
    return dict(metadata)


def load_sidecar(torchscript_path: str | Path, *, verify_hash: bool = True) -> dict[str, Any]:
    path = Path(torchscript_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DECO TorchScript not found: {path}")
    sidecar = sidecar_path(path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"DECO metadata sidecar not found: {sidecar}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("DECO metadata root must be a mapping")
    validated = validate_metadata(metadata)
    expected_hash = validated.get("torchscript_sha256")
    if verify_hash:
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError("DECO metadata is missing torchscript_sha256")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"DECO TorchScript SHA256 mismatch: expected {expected_hash}, got {actual_hash}"
            )
    return validated


def _embedded_contract(metadata: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {"torchscript_sha256", "output_path", "source_checkpoint_sha256"}
    return {key: value for key, value in metadata.items() if key not in ignored}


def load_torchscript(
    torchscript_path: str | Path,
    *,
    device: str,
    verify_hash: bool = True,
):
    """Load a TorchScript model and prove its embedded contract matches the sidecar."""
    path = Path(torchscript_path).expanduser().resolve()
    metadata = load_sidecar(path, verify_hash=verify_hash)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to load the DECO TorchScript artifact") from error
    extra = {"deco_metadata.json": ""}
    model = torch.jit.load(str(path), map_location=device, _extra_files=extra).eval()
    try:
        embedded = json.loads(extra["deco_metadata.json"])
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("DECO TorchScript is missing valid embedded metadata") from error
    validate_metadata(embedded)
    if _embedded_contract(embedded) != _embedded_contract(metadata):
        raise ValueError("DECO sidecar metadata does not match embedded TorchScript metadata")
    return model, metadata
