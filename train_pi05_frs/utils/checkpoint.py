from __future__ import annotations

from collections.abc import Mapping
import ctypes
import dataclasses
import hashlib
import json
import os
import pathlib
import pickle
from typing import Any
import uuid

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax import traverse_util

from train_pi05_frs.utils.model import DecoderConfig
from train_pi05_frs.utils.model import TactileConditionedFlowDecoder

PARAMS_NAME = "params.npz"
OPT_STATE_NAME = "opt_state.npz"
OPT_STATE_TREEDEF_NAME = "opt_state.treedef.pkl"
CHECKPOINT_NAME = "checkpoint.json"
GENERATION_ROOT_NAME = ".checkpoint-generations"


def _checkpoint_fault(stage: str) -> None:
    """Test hook for simulating interruption at every publication stage."""


def _full_parameter_state(model: TactileConditionedFlowDecoder):
    state = nnx.state(model, nnx.Param)
    return state, traverse_util.flatten_dict(state.to_pure_dict())


def _flat_parameter_state(model: TactileConditionedFlowDecoder):
    state, flat = _full_parameter_state(model)
    return state, {path: value for path, value in flat.items() if value is not None}


def _path_name(path: tuple[Any, ...]) -> str:
    return "/".join(f"{type(part).__name__}:{part}" for part in path)


def _fsync_file(path: pathlib.Path) -> None:
    with path.open("rb") as file:
        os.fsync(file.fileno())


def _fsync_directory(path: pathlib.Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _savez_fsync(path: pathlib.Path, arrays: Mapping[str, np.ndarray]) -> None:
    np.savez(path, **arrays)
    _fsync_file(path)


def _write_bytes_fsync(path: pathlib.Path, payload: bytes) -> None:
    with path.open("xb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_record(path: pathlib.Path, *, generation: str) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "generation": generation,
        "size": len(payload),
        "sha256": _sha256(payload),
    }


def _metadata_payload_for_checksum(metadata: Mapping[str, Any]) -> bytes:
    canonical = json.loads(json.dumps(metadata))
    canonical["files"][CHECKPOINT_NAME]["sha256"] = ""
    return _canonical_json(canonical)


def _finish_metadata_self_record(metadata: dict[str, Any]) -> bytes:
    record = metadata["files"][CHECKPOINT_NAME]
    while True:
        payload = _metadata_payload_for_checksum(metadata)
        size = len(payload)
        if record["size"] == size:
            break
        record["size"] = size
    record["sha256"] = _sha256(payload)
    return _canonical_json(metadata)


def _leaf_to_numpy(leaf: Any) -> np.ndarray:
    if isinstance(leaf, (np.bool_, bool)):
        return np.asarray(leaf, dtype=np.bool_)
    if isinstance(leaf, (np.integer, int)):
        return np.asarray(leaf, dtype=np.int32)
    if isinstance(leaf, (np.floating, float)):
        return np.asarray(leaf, dtype=np.float32)
    dtype = getattr(leaf, "dtype", None)
    if dtype is not None and str(dtype) == "bfloat16":
        return np.asarray(jnp.asarray(leaf).astype(jnp.float32))
    array = np.asarray(jax.device_get(leaf))
    if array.dtype.kind == "V":
        raise TypeError(f"Cannot serialize void-dtype leaf with shape {array.shape}.")
    if np.issubdtype(array.dtype, np.floating):
        return np.asarray(array, dtype=np.float32)
    if np.issubdtype(array.dtype, np.integer):
        return np.asarray(array, dtype=np.int32 if array.dtype != np.int64 else np.int64)
    if array.dtype == np.bool_:
        return array
    raise TypeError(f"Unsupported opt_state leaf dtype {array.dtype} shape {array.shape}.")


def _numpy_to_leaf(array: np.ndarray) -> Any:
    array = np.asarray(array)
    if array.dtype.kind == "V":
        raise TypeError(f"Unsupported void dtype {array.dtype} in checkpoint leaf.")
    return jnp.asarray(array)


def _optimizer_step_value(optimizer: nnx.Optimizer) -> int:
    return int(np.asarray(jax.device_get(optimizer.step[...])))


def _serialize_metric_value(value: Any) -> float | str | int | bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    return float(value)


def _checkpoint_alias(path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path).expanduser()
    return path.parent.resolve(strict=False) / path.name


def _rename_exchange(left: pathlib.Path, right: pathlib.Path) -> None:
    """Atomically exchange two directory entries on Linux."""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError(
            "cannot atomically upgrade a legacy checkpoint directory on this platform"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(left),
        -100,
        os.fsencode(right),
        2,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(right))


def _publish_pointer(
    alias: pathlib.Path,
    *,
    snapshot: pathlib.Path,
    generation_root: pathlib.Path,
    generation: str,
) -> None:
    relative_target = os.path.relpath(snapshot, alias.parent)
    temporary_pointer = alias.parent / f".{alias.name}.{generation}.pointer"
    os.symlink(relative_target, temporary_pointer)
    _fsync_directory(alias.parent)
    _checkpoint_fault("after_pointer_prepare")

    if not os.path.lexists(alias) or alias.is_symlink():
        os.replace(temporary_pointer, alias)
    elif alias.is_dir():
        _rename_exchange(temporary_pointer, alias)
        retired = generation_root / f"legacy-{generation}"
        os.replace(temporary_pointer, retired)
        _fsync_directory(generation_root)
    else:
        raise ValueError(f"checkpoint target is neither a directory nor symlink: {alias}")
    _checkpoint_fault("after_pointer_publish")
    _fsync_directory(alias.parent)
    _checkpoint_fault("after_pointer_parent_fsync")


def _snapshot_metadata(snapshot: pathlib.Path) -> dict[str, Any]:
    try:
        with (snapshot / CHECKPOINT_NAME).open(encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read checkpoint metadata from {snapshot}: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"checkpoint metadata must be a mapping: {snapshot}")
    return metadata


def _validated_v3_metadata(
    snapshot: pathlib.Path,
    metadata: dict[str, Any],
    *,
    expected_generation: str | None = None,
) -> None:
    generation = metadata.get("generation")
    expected_generation = expected_generation or snapshot.name
    if not isinstance(generation, str) or generation != expected_generation:
        raise ValueError(
            "checkpoint generation mismatch: "
            f"metadata={generation!r}, directory={expected_generation!r}"
        )
    files = metadata.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("checkpoint v3 metadata is missing file checksums")
    expected_files = {PARAMS_NAME, CHECKPOINT_NAME}
    if metadata.get("has_opt_state"):
        expected_files.update((OPT_STATE_NAME, OPT_STATE_TREEDEF_NAME))
    if set(files) != expected_files:
        raise ValueError(
            f"checkpoint file manifest mismatch: {sorted(files)} != {sorted(expected_files)}"
        )
    for name in sorted(expected_files):
        record = files.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"checkpoint file record is invalid: {name}")
        if record.get("generation") != generation:
            raise ValueError(f"checkpoint file generation mismatch: {name}")
        path = snapshot / name
        if not path.is_file():
            raise ValueError(f"checkpoint file is missing: {name}")
        payload = (
            _metadata_payload_for_checksum(metadata)
            if name == CHECKPOINT_NAME
            else path.read_bytes()
        )
        if record.get("size") != len(payload):
            raise ValueError(f"checkpoint file size mismatch: {name}")
        if record.get("sha256") != _sha256(payload):
            raise ValueError(f"checkpoint file checksum mismatch: {name}")

    parameter_count = len(metadata.get("parameter_paths", ()))
    try:
        with np.load(snapshot / PARAMS_NAME, allow_pickle=False) as archive:
            expected = {f"p{index:05d}" for index in range(parameter_count)}
            if set(archive.files) != expected:
                raise ValueError("checkpoint parameter array set is incomplete")
            for name in sorted(expected):
                _ = archive[name]
        if metadata.get("has_opt_state"):
            leaf_count = int(metadata["opt_state_leaf_count"])
            with np.load(snapshot / OPT_STATE_NAME, allow_pickle=False) as archive:
                expected = {f"p{index:05d}" for index in range(leaf_count)}
                if set(archive.files) != expected:
                    raise ValueError("checkpoint optimizer array set is incomplete")
                for name in sorted(expected):
                    _ = archive[name]
            with (snapshot / OPT_STATE_TREEDEF_NAME).open("rb") as file:
                pickle.load(file)
    except (KeyError, OSError, pickle.PickleError, ValueError) as error:
        raise ValueError(f"checkpoint generation is invalid: {error}") from error


def resolve_checkpoint_snapshot(directory: pathlib.Path) -> pathlib.Path:
    """Pin an alias to one immutable snapshot for the whole restore operation."""
    try:
        snapshot = pathlib.Path(directory).expanduser().resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError(f"checkpoint does not exist: {directory}") from error
    if not snapshot.is_dir():
        raise ValueError(f"checkpoint snapshot is not a directory: {snapshot}")
    return snapshot


def _load_metadata(directory: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    snapshot = resolve_checkpoint_snapshot(directory)
    metadata = _snapshot_metadata(snapshot)
    version = int(metadata.get("version", 1))
    if version >= 3:
        _validated_v3_metadata(snapshot, metadata)
    return snapshot, metadata


def save_checkpoint(
    directory: pathlib.Path,
    model: TactileConditionedFlowDecoder,
    *,
    epoch: int,
    metrics: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
    optimizer: nnx.Optimizer | None = None,
) -> None:
    alias = _checkpoint_alias(directory)
    alias.parent.mkdir(parents=True, exist_ok=True)
    generation_root = alias.parent / GENERATION_ROOT_NAME
    generation_root.mkdir(exist_ok=True)
    generation = uuid.uuid4().hex
    staging = generation_root / f".{generation}.writing"
    snapshot = generation_root / generation
    staging.mkdir()

    _, flat = _flat_parameter_state(model)
    ordered = sorted(flat.items(), key=lambda item: _path_name(item[0]))
    arrays = {f"p{index:05d}": np.asarray(value) for index, (_, value) in enumerate(ordered)}
    _savez_fsync(staging / PARAMS_NAME, arrays)
    _checkpoint_fault("after_params_fsync")

    metadata: dict[str, Any] = {
        "version": 3,
        "generation": generation,
        "epoch": int(epoch),
        "metrics": {key: _serialize_metric_value(value) for key, value in metrics.items()},
        "decoder_config": dataclasses.asdict(model.config),
        "parameter_paths": [_path_name(path) for path, _ in ordered],
        "has_opt_state": optimizer is not None,
    }
    if optimizer is not None:
        optimizer_state = nnx.state(optimizer, type(optimizer.step))
        pure_opt_state = optimizer_state["opt_state"].to_pure_dict()
        leaves, treedef = jax.tree_util.tree_flatten(pure_opt_state)
        opt_arrays = {
            f"p{index:05d}": _leaf_to_numpy(leaf)
            for index, leaf in enumerate(leaves)
        }
        _savez_fsync(staging / OPT_STATE_NAME, opt_arrays)
        _checkpoint_fault("after_opt_state_fsync")
        with (staging / OPT_STATE_TREEDEF_NAME).open("xb") as file:
            pickle.dump(treedef, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        _checkpoint_fault("after_opt_treedef_fsync")
        metadata["opt_state_leaf_count"] = len(leaves)
        metadata["optimizer_step"] = _optimizer_step_value(optimizer)
    if extra_metadata is not None:
        metadata["extra_metadata"] = extra_metadata

    file_names = [PARAMS_NAME]
    if optimizer is not None:
        file_names.extend((OPT_STATE_NAME, OPT_STATE_TREEDEF_NAME))
    metadata["files"] = {
        name: _file_record(staging / name, generation=generation)
        for name in file_names
    }
    metadata["files"][CHECKPOINT_NAME] = {
        "generation": generation,
        "size": 0,
        "sha256": "",
    }
    _write_bytes_fsync(
        staging / CHECKPOINT_NAME, _finish_metadata_self_record(metadata)
    )
    _checkpoint_fault("after_metadata_fsync")
    _validated_v3_metadata(
        staging, _snapshot_metadata(staging), expected_generation=generation
    )
    _checkpoint_fault("after_snapshot_validation")
    _fsync_directory(staging)
    _checkpoint_fault("after_snapshot_dir_fsync")
    os.replace(staging, snapshot)
    _fsync_directory(generation_root)
    _checkpoint_fault("after_generation_publish")
    _publish_pointer(
        alias,
        snapshot=snapshot,
        generation_root=generation_root,
        generation=generation,
    )


def load_checkpoint(directory: pathlib.Path) -> tuple[TactileConditionedFlowDecoder, dict[str, Any]]:
    snapshot, metadata = _load_metadata(directory)
    config = DecoderConfig(**metadata["decoder_config"])
    model = TactileConditionedFlowDecoder(config, rngs=nnx.Rngs(0))
    state, full_template = _full_parameter_state(model)
    ordered_full = sorted(full_template.items(), key=lambda item: _path_name(item[0]))
    ordered_filtered = [(path, value) for path, value in ordered_full if value is not None]
    metadata_names = metadata["parameter_paths"]
    if metadata_names == [_path_name(path) for path, _ in ordered_filtered]:
        indexed_paths = [(index, path) for index, (path, _) in enumerate(ordered_filtered)]
    elif metadata_names == [_path_name(path) for path, _ in ordered_full]:
        indexed_paths = [
            (index, path)
            for index, (path, value) in enumerate(ordered_full)
            if value is not None
        ]
    else:
        raise ValueError("Checkpoint parameter structure does not match the decoder implementation.")

    with np.load(snapshot / PARAMS_NAME, allow_pickle=False) as archive:
        restored_flat = {
            path: jnp.asarray(archive[f"p{index:05d}"])
            for index, path in indexed_paths
        }
    state.replace_by_pure_dict(traverse_util.unflatten_dict(restored_flat))
    nnx.update(model, state)
    return model, metadata


def load_optimizer_state(directory: pathlib.Path) -> tuple[Any | None, int | None]:
    """Load optimizer pytree and step count when present; otherwise ``(None, None)``."""
    snapshot, metadata = _load_metadata(directory)
    opt_state_path = snapshot / OPT_STATE_NAME
    treedef_path = snapshot / OPT_STATE_TREEDEF_NAME
    if not (metadata.get("has_opt_state") and opt_state_path.exists() and treedef_path.exists()):
        return None, None
    try:
        with treedef_path.open("rb") as file:
            treedef = pickle.load(file)
        with np.load(opt_state_path, allow_pickle=False) as archive:
            leaf_count = int(metadata.get("opt_state_leaf_count", len(archive.files)))
            leaves = [_numpy_to_leaf(archive[f"p{index:05d}"]) for index in range(leaf_count)]
        opt_state = jax.tree_util.tree_unflatten(treedef, leaves)
        step = metadata.get("optimizer_step")
        return opt_state, (int(step) if step is not None else None)
    except Exception as error:
        if int(metadata.get("version", 1)) >= 3:
            raise ValueError(
                f"failed to restore optimizer state from checkpoint generation: {error}"
            ) from error
        print(
            f"warning: failed to restore optimizer state from {snapshot}: {error}; "
            "reinitializing Adam state.",
            flush=True,
        )
        return None, None


def restore_optimizer_state(
    optimizer: nnx.Optimizer,
    *,
    opt_state: Any,
    step: int | None,
) -> None:
    optimizer_state = nnx.state(optimizer, type(optimizer.step))
    optimizer_state["opt_state"].replace_by_pure_dict(opt_state)
    nnx.update(optimizer, optimizer_state)
    if step is not None:
        optimizer.step.value = jnp.asarray(step, dtype=jnp.uint32)
