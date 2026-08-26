"""Validate and cache conversions of the Flax tactile ResNet18 checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from contextlib import contextmanager
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from train_deco.models.tactile_resnet import RESNET18_EMBEDDING_DIM, TactileResNet18


CONVERSION_VERSION = "1"
ARCHITECTURE = "resnet18"
_WEIGHTS_NAME = "encoder.safetensors"
_METADATA_NAME = "encoder.json"
_DIRECT_SIDECAR_SUFFIX = ".json"
_PARITY_SEED = 1729
_PARITY_INPUT_SHAPE = (4, 224, 224, 3)
_PARITY_RTOL = 2e-3
_PARITY_ATOL = 2e-4


@dataclass(frozen=True)
class ResolvedTactileEncoder:
    weights_path: Path
    metadata_path: Path
    source_sha256: str
    architecture: str
    embedding_dim: int


@dataclass(frozen=True)
class _Source:
    path: Path
    params_path: Path | None
    checkpoint: dict[str, Any] | None
    digest: str
    framework: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_digest(checkpoint_path: Path, params_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (checkpoint_path, params_path):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _resolve_source(source: str | Path) -> _Source:
    path = Path(source).expanduser().resolve()
    if path.is_file() and path.suffix == ".safetensors":
        return _Source(path, None, None, _sha256_file(path), "pytorch")
    if not path.is_dir():
        raise FileNotFoundError(
            "tactile encoder source must be a .safetensors file or a checkpoint directory: "
            f"{path}"
        )
    checkpoint_path = path / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"tactile encoder checkpoint directory is missing checkpoint.json: {path}")
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid tactile encoder checkpoint.json: {checkpoint_path}") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("tactile encoder checkpoint.json must contain a JSON object")
    requested = checkpoint.get("params_file")
    if requested is None:
        candidates = sorted(path.glob("params-*.npz"))
        if len(candidates) != 1:
            raise ValueError(
                "tactile encoder checkpoint directory must contain exactly one params-*.npz "
                "when checkpoint.json has no params_file"
            )
        params_path = candidates[0]
    else:
        if not isinstance(requested, str) or Path(requested).name != requested:
            raise ValueError("checkpoint.json params_file must be a simple params-*.npz filename")
        params_path = path / requested
        if not params_path.is_file():
            raise FileNotFoundError(f"checkpoint.json references missing parameter archive: {params_path}")
    if params_path.suffix != ".npz" or not params_path.name.startswith("params-"):
        raise ValueError(f"tactile encoder params_file must be params-*.npz, got {params_path.name!r}")
    return _Source(path, params_path, checkpoint, _directory_digest(checkpoint_path, params_path), "flax")


def _flax_path_for_torch(name: str) -> str | None:
    if name.endswith("num_batches_tracked"):
        return None
    parts = name.split(".")
    if parts[0].startswith("layer"):
        block = f"block{parts[0][-1]}_{parts[1]}"
        tail = "/".join(parts[2:])
    else:
        block = ""
        tail = "/".join(parts)
    is_stat = tail.endswith(("running_mean", "running_var"))
    if is_stat:
        tail = tail.replace("running_mean", "mean").replace("running_var", "var")
        prefix = "tactile_resnet/batch_stats"
    else:
        if tail.endswith("weight"):
            module_name = name.split(".")[-2]
            is_convolution = module_name.startswith("conv") or module_name == "proj_conv"
            tail = tail[:-len("weight")] + ("kernel" if is_convolution else "scale")
        prefix = "tactile_resnet/params"
    return "/".join(part for part in (prefix, block, tail) if part)


def _flax_shape_for_torch(tensor: torch.Tensor) -> tuple[int, ...]:
    if tensor.ndim == 4:
        return (tensor.shape[2], tensor.shape[3], tensor.shape[1], tensor.shape[0])
    return tuple(tensor.shape)


def _load_flax_state(source: _Source) -> dict[str, torch.Tensor]:
    assert source.params_path is not None and source.checkpoint is not None
    paths = source.checkpoint.get("parameter_paths")
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise ValueError("checkpoint.json parameter_paths must be a list of Flax leaf paths")
    if len(paths) != len(set(paths)):
        raise ValueError("checkpoint.json parameter_paths contains duplicate leaves")
    archive_by_path: dict[str, np.ndarray] = {}
    try:
        with np.load(source.params_path, allow_pickle=False) as archive:
            for index, path in enumerate(paths):
                archive_name = f"p{index:05d}"
                if archive_name not in archive:
                    raise ValueError(f"missing archive leaf {archive_name} for parameter path {path!r}")
                archive_by_path[path] = np.asarray(archive[archive_name])
    except ValueError:
        raise
    except Exception as error:  # corrupt NPZ errors vary by NumPy release
        raise ValueError(f"failed to read tactile encoder parameter archive: {source.params_path}") from error

    reference = TactileResNet18().state_dict()
    expected = {name: _flax_path_for_torch(name) for name in reference}
    expected_paths = {path for path in expected.values() if path is not None}
    present_paths = {path for path in archive_by_path if path.startswith("tactile_resnet/")}
    missing = sorted(expected_paths - present_paths)
    extra = sorted(present_paths - expected_paths)
    if missing:
        raise ValueError(f"missing required tactile ResNet leaves: {missing}")
    if extra:
        raise ValueError(f"unexpected tactile ResNet leaves: {extra}")

    converted = dict(reference)
    for name, reference_tensor in reference.items():
        flax_path = expected[name]
        if flax_path is None:
            continue
        array = archive_by_path[flax_path]
        expected_shape = _flax_shape_for_torch(reference_tensor)
        if tuple(array.shape) != expected_shape:
            raise ValueError(
                f"shape mismatch for {name}: expected Flax {expected_shape}, got {tuple(array.shape)}"
            )
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"tactile encoder leaf {flax_path!r} must contain finite numeric values")
        if reference_tensor.ndim == 4:
            array = np.transpose(array, (3, 2, 0, 1))
        converted[name] = torch.from_numpy(np.ascontiguousarray(array)).to(dtype=reference_tensor.dtype)
    model = TactileResNet18()
    model.load_state_dict(converted, strict=True)
    return converted


def _validate_pytorch_state(state: dict[str, torch.Tensor], *, source: Path) -> dict[str, torch.Tensor]:
    model = TactileResNet18()
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"converted tactile encoder does not strictly match ResNet18: {source}") from error
    for name, tensor in state.items():
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"converted tactile encoder tensor {name!r} is non-finite")
    return state


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_source_unchanged(source: _Source) -> None:
    if _resolve_source(source.path).digest != source.digest:
        raise RuntimeError("tactile encoder source changed during conversion; retry resolution")

def import_jax_flax_for_cpu() -> tuple[Any, Any]:
    """Lazily import conversion-only JAX/Flax with CPU selected before import."""

    existing_jax = sys.modules.get("jax")
    if existing_jax is not None:
        configured_platform = getattr(existing_jax.config, "jax_platform_name", "")
        if configured_platform not in (None, "", "cpu"):
            raise RuntimeError(
                "JAX is already imported with a non-CPU platform setting "
                f"({configured_platform!r}); cannot safely switch to CPU"
            )
        try:
            existing_backend = existing_jax.default_backend()
        except Exception as error:  # JAX may have failed while initializing elsewhere.
            raise RuntimeError("JAX is already imported but its backend cannot be inspected") from error
        if existing_backend != "cpu":
            raise RuntimeError(
                "JAX is already initialized on backend "
                f"{existing_backend!r}; cannot safely switch to CPU"
            )

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    try:
        import jax

        jax.config.update("jax_platform_name", "cpu")
        if jax.default_backend() != "cpu":
            raise RuntimeError("JAX did not initialize the requested CPU backend")
    except ImportError:
        raise
    except Exception as error:
        raise RuntimeError("failed to initialize JAX on CPU") from error
    import flax
    return jax, flax



def _direct_sidecar_path(weights_path: Path) -> Path:
    return weights_path.with_suffix(weights_path.suffix + _DIRECT_SIDECAR_SUFFIX)


def _validated_parity_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("tactile encoder metadata is missing a parity record")
    if value.get("status") != "passed":
        raise ValueError("tactile encoder parity status must be passed")
    if value.get("seed") != _PARITY_SEED:
        raise ValueError("tactile encoder parity seed does not match the conversion contract")
    if value.get("input_shape") != list(_PARITY_INPUT_SHAPE):
        raise ValueError("tactile encoder parity input shape does not match [4,224,224,3]")
    for key, expected in (("rtol", _PARITY_RTOL), ("atol", _PARITY_ATOL)):
        if value.get(key) != expected:
            raise ValueError(f"tactile encoder parity {key} does not match the conversion contract")
    for key in ("max_abs", "max_rel"):
        number = value.get(key)
        if not isinstance(number, (int, float)) or not np.isfinite(number) or number < 0:
            raise ValueError(f"tactile encoder parity {key} must be a finite non-negative number")
    return dict(value)


def _verify_flax_pytorch_parity(source: _Source, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Run the conversion proof on a fixed CPU JAX/PyTorch input."""

    if source.framework != "flax":
        raise ValueError("Flax parity verification requires a checkpoint directory")
    jax, _ = import_jax_flax_for_cpu()
    import jax.numpy as jnp
    from train_encoder.utils.checkpoint import load_checkpoint
    from train_encoder.utils.resnet import encode_resnet18

    rng = np.random.default_rng(_PARITY_SEED)
    images = rng.random(_PARITY_INPUT_SHAPE, dtype=np.float32)
    params, _ = load_checkpoint(source.path)
    try:
        jax_embeddings, _ = encode_resnet18(
            params["tactile_resnet"], jnp.asarray(images), train=False
        )
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("failed to run Flax tactile ResNet parity verification") from error
    model = TactileResNet18().eval()
    model.load_state_dict(state, strict=True)
    with torch.inference_mode():
        torch_embeddings = model(torch.from_numpy(images).permute(0, 3, 1, 2)).cpu().numpy()
    jax_values = np.asarray(jax.device_get(jax_embeddings), dtype=np.float32)
    difference = np.abs(torch_embeddings - jax_values)
    max_abs = float(difference.max(initial=0.0))
    max_rel = float((difference / np.maximum(np.abs(jax_values), np.finfo(np.float32).tiny)).max(initial=0.0))
    if not np.allclose(torch_embeddings, jax_values, rtol=_PARITY_RTOL, atol=_PARITY_ATOL):
        raise ValueError(
            "JAX-to-PyTorch tactile encoder parity failed: "
            f"max_abs={max_abs:.6g}, max_rel={max_rel:.6g}, "
            f"rtol={_PARITY_RTOL}, atol={_PARITY_ATOL}"
        )
    return {
        "status": "passed",
        "seed": _PARITY_SEED,
        "input_shape": list(_PARITY_INPUT_SHAPE),
        "rtol": _PARITY_RTOL,
        "atol": _PARITY_ATOL,
        "max_abs": max_abs,
        "max_rel": max_rel,
    }


@contextmanager
def _cache_lock(path: Path):
    """Hold an advisory cross-process lock for one content-addressed artifact."""

    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _source_file_inventory(source: _Source) -> list[dict[str, str]]:
    if source.framework == "pytorch":
        paths = (source.path,)
    else:
        assert source.params_path is not None
        paths = (source.path / "checkpoint.json", source.params_path)
    return [
        {"path": path.name, "sha256": _sha256_file(path)}
        for path in paths
    ]


def _mapped_key_count() -> int:
    return sum(
        _flax_path_for_torch(name) is not None
        for name in TactileResNet18().state_dict()
    )


def _strict_conversion_sidecar(source: _Source) -> dict[str, Any]:
    """Require an adjacent, complete proof emitted by a prior conversion."""

    sidecar_path = source.path.with_suffix(".json")
    if not sidecar_path.is_file():
        raise ValueError(
            "direct tactile encoder .safetensors requires an existing conversion sidecar "
            f"at {sidecar_path}"
        )
    try:
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid direct tactile encoder conversion sidecar: {sidecar_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError("direct tactile encoder conversion sidecar must be a JSON object")
    if (
        metadata.get("conversion_version") != CONVERSION_VERSION
        or metadata.get("architecture") != ARCHITECTURE
        or metadata.get("embedding_dim") != RESNET18_EMBEDDING_DIM
        or metadata.get("target_framework") != "pytorch"
        or metadata.get("weights_sha256") != source.digest
        or not isinstance(metadata.get("source_sha256"), str)
        or not isinstance(metadata.get("source_file_inventory"), list)
        or metadata.get("mapped_key_count") != _mapped_key_count()
        or not isinstance(metadata.get("tensor_shapes"), dict)
    ):
        raise ValueError("direct tactile encoder conversion sidecar is incomplete or does not match source bytes")
    _validated_parity_record(metadata.get("parity"))
    return metadata


def _artifact_metadata_v3(
    source: _Source,
    state: dict[str, torch.Tensor],
    weights_path: Path,
    parity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "conversion_version": CONVERSION_VERSION,
        "embedding_dim": RESNET18_EMBEDDING_DIM,
        "source_framework": source.framework,
        "source_path": str(source.path),
        "source_sha256": source.digest,
        "source_file_inventory": _source_file_inventory(source),
        "target_framework": "pytorch",
        "mapped_key_count": _mapped_key_count(),
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in state.items()},
        "weights_sha256": _sha256_file(weights_path),
        "parity": _validated_parity_record(parity),
    }


def verify_resolved_tactile_encoder(artifact: ResolvedTactileEncoder) -> None:
    """Verify source, parity proof, cache metadata, hashes, and strict loading."""

    try:
        metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid tactile encoder metadata: {artifact.metadata_path}") from error
    if not isinstance(metadata, dict) or not isinstance(metadata.get("source_path"), str):
        raise ValueError("tactile encoder metadata is missing source_path")
    source = _resolve_source(metadata["source_path"])
    if (
        artifact.source_sha256 != source.digest
        or metadata.get("source_sha256") != source.digest
        or metadata.get("architecture") != ARCHITECTURE
        or metadata.get("conversion_version") != CONVERSION_VERSION
        or metadata.get("embedding_dim") != RESNET18_EMBEDDING_DIM
        or metadata.get("source_framework") != source.framework
        or metadata.get("target_framework") != "pytorch"
        or metadata.get("source_file_inventory") != _source_file_inventory(source)
        or metadata.get("mapped_key_count") != _mapped_key_count()
    ):
        raise ValueError("tactile encoder metadata does not match the resolved source contract")
    if metadata.get("weights_sha256") != _sha256_file(artifact.weights_path):
        raise ValueError("tactile encoder artifact SHA256 does not match metadata")
    parity = _validated_parity_record(metadata.get("parity"))
    if source.framework == "pytorch" and parity != _strict_conversion_sidecar(source).get("parity"):
        raise ValueError("tactile encoder parity record does not match its conversion sidecar")
    try:
        state = _validate_pytorch_state(load_file(str(artifact.weights_path)), source=artifact.weights_path)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"failed to load tactile encoder artifact: {artifact.weights_path}") from error
    if metadata.get("tensor_shapes") != {name: list(tensor.shape) for name, tensor in state.items()}:
        raise ValueError("tactile encoder tensor-shape inventory does not match artifact")


def _generation_artifact(slot: Path, source: _Source) -> ResolvedTactileEncoder | None:
    current_path = slot / "current.json"
    try:
        pointer = json.loads(current_path.read_text(encoding="utf-8"))
        generation = pointer.get("generation")
        if not isinstance(generation, str) or Path(generation).name != generation:
            return None
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    generation_path = slot / "generations" / generation
    artifact = ResolvedTactileEncoder(
        generation_path / _WEIGHTS_NAME,
        generation_path / _METADATA_NAME,
        source.digest,
        ARCHITECTURE,
        RESNET18_EMBEDDING_DIM,
    )
    try:
        verify_resolved_tactile_encoder(artifact)
    except (OSError, RuntimeError, ValueError):
        return None
    return artifact


def resolve_tactile_encoder(source: str | Path, cache_root: str | Path) -> ResolvedTactileEncoder:
    """Resolve a locked, generation-published, verified tactile encoder artifact."""

    initial = _resolve_source(source)
    version_root = Path(cache_root).expanduser().resolve() / f"v{CONVERSION_VERSION}"
    slot = version_root / initial.digest
    with _cache_lock(version_root / f".{initial.digest}.lock"):
        resolved_source = _resolve_source(source)
        if resolved_source.digest != initial.digest:
            raise RuntimeError("tactile encoder source changed while waiting for conversion lock; retry resolution")
        cached = _generation_artifact(slot, resolved_source)
        if cached is not None:
            return cached
        if resolved_source.framework == "flax":
            state = _load_flax_state(resolved_source)
            parity = _verify_flax_pytorch_parity(resolved_source, state)
        else:
            sidecar = _strict_conversion_sidecar(resolved_source)
            parity = _validated_parity_record(sidecar.get("parity"))
            state = _validate_pytorch_state(load_file(str(resolved_source.path)), source=resolved_source.path)
        _assert_source_unchanged(resolved_source)
        generations = slot / "generations"
        generations.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        staging = generations / f".{generation}.tmp"
        published = generations / generation
        staging.mkdir()
        try:
            staging_weights = staging / _WEIGHTS_NAME
            if resolved_source.framework == "flax":
                save_file(state, str(staging_weights))
            else:
                shutil.copyfile(resolved_source.path, staging_weights)
            _validate_pytorch_state(load_file(str(staging_weights)), source=staging_weights)
            _atomic_json(staging / _METADATA_NAME, _artifact_metadata_v3(resolved_source, state, staging_weights, parity))
            _assert_source_unchanged(resolved_source)
            os.replace(staging, published)
            try:
                _atomic_json(slot / "current.json", {"generation": generation})
            except Exception:
                shutil.rmtree(published, ignore_errors=True)
                raise
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    artifact = ResolvedTactileEncoder(
        published / _WEIGHTS_NAME,
        published / _METADATA_NAME,
        resolved_source.digest,
        ARCHITECTURE,
        RESNET18_EMBEDDING_DIM,
    )
    verify_resolved_tactile_encoder(artifact)
    return artifact


def load_tactile_encoder_weights(module: torch.nn.Module, artifact: ResolvedTactileEncoder) -> None:
    verify_resolved_tactile_encoder(artifact)
    try:
        module.load_state_dict(load_file(str(artifact.weights_path)), strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"failed to strictly load tactile encoder artifact: {artifact.weights_path}") from error
