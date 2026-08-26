from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from train_deco.models.tactile_resnet import TactileResNet18
import train_deco.tactile_encoder_conversion as conversion
from train_deco.tactile_encoder_conversion import (
    CONVERSION_VERSION,
    create_trusted_tactile_encoder_sidecar,
    load_tactile_encoder_weights,
    resolve_tactile_encoder,
    verify_resolved_tactile_encoder,
)


def _flax_paths_and_arrays() -> tuple[list[str], dict[str, np.ndarray]]:
    """Build a complete, deterministic Flax-format ResNet18 checkpoint."""

    paths: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    for name, tensor in TactileResNet18().state_dict().items():
        if name.endswith("num_batches_tracked"):
            continue
        parts = name.split(".")
        if parts[0].startswith("layer"):
            block = f"block{parts[0][-1]}_{parts[1]}"
            remainder = "/".join(parts[2:])
        else:
            block = ""
            remainder = "/".join(parts)
        is_stat = remainder.endswith(("running_mean", "running_var"))
        if is_stat:
            leaf = remainder.replace("running_mean", "mean").replace("running_var", "var")
            prefix = "tactile_resnet/batch_stats"
        else:
            leaf = remainder.replace("weight", "scale").replace("bias", "bias")
            if leaf.endswith("kernel"):
                raise AssertionError("PyTorch tactile ResNet must name convolutions as weight")
            if name.endswith("weight") and tensor.ndim == 4:
                leaf = remainder[:-len("weight")] + "kernel"
            prefix = "tactile_resnet/params"
        path = "/".join(part for part in (prefix, block, leaf) if part)
        array = tensor.detach().cpu().numpy().astype(np.float32)
        if tensor.ndim == 4:
            array = np.transpose(array, (2, 3, 1, 0))
        arrays[path] = array
        paths.append(path)
    # A non-symmetric sentinel proves that the HWIO -> OIHW transpose is real.
    arrays["tactile_resnet/params/conv1/kernel"] = np.arange(
        np.prod(arrays["tactile_resnet/params/conv1/kernel"].shape), dtype=np.float32
    ).reshape(arrays["tactile_resnet/params/conv1/kernel"].shape)
    return sorted(paths), arrays


def _write_flax_checkpoint(directory: Path, *, seed: int = 0) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    paths, arrays = _flax_paths_and_arrays()
    if seed:
        arrays["tactile_resnet/params/bn1/bias"] = np.full((64,), seed, dtype=np.float32)
    params_name = "params-synthetic.npz"
    np.savez(directory / params_name, **{f"p{index:05d}": arrays[path] for index, path in enumerate(paths)})
    (directory / "checkpoint.json").write_text(
        json.dumps(
            {
                "version": 4,
                "tactile_backbone": "resnet18",
                "params_file": params_name,
                "parameter_paths": paths,
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_tactile_resnet18_returns_512_embeddings_with_stable_parameter_names() -> None:
    model = TactileResNet18().eval()

    output = model(torch.zeros(2, 3, 224, 224))

    assert output.shape == (2, 512)
    assert not any("projection" in name or "classifier" in name for name in model.state_dict())
    assert tuple(model.state_dict())[:5] == (
        "conv1.weight",
        "bn1.weight",
        "bn1.bias",
        "bn1.running_mean",
        "bn1.running_var",
    )
    assert "layer4.1.bn2.running_var" in model.state_dict()
    with pytest.raises(ValueError, match=r"\[N, 3, 224, 224\]"):
        model(torch.zeros(2, 3, 128, 128))


def test_conversion_rejects_missing_and_shape_mismatched_flax_leaves(tmp_path: Path) -> None:
    source = _write_flax_checkpoint(tmp_path / "source")
    metadata = json.loads((source / "checkpoint.json").read_text(encoding="utf-8"))
    params = source / metadata["params_file"]
    with np.load(params, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    conv_archive_name = f"p{metadata['parameter_paths'].index('tactile_resnet/params/conv1/kernel'):05d}"
    arrays.pop(conv_archive_name)
    np.savez(params, **arrays)
    with pytest.raises(ValueError, match=rf"missing archive leaf.*{conv_archive_name}"):
        resolve_tactile_encoder(source, tmp_path / "cache")

    _write_flax_checkpoint(source)
    with np.load(params, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays[conv_archive_name] = np.zeros((1,), dtype=np.float32)
    np.savez(params, **arrays)
    with pytest.raises(ValueError, match="shape mismatch.*conv1.weight"):
        resolve_tactile_encoder(source, tmp_path / "cache")


def test_resolution_is_content_addressed_and_writes_complete_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        conversion,
        "_verify_flax_pytorch_parity",
        lambda source, state: {
            "status": "passed",
            "seed": 1729,
            "input_shape": [4, 224, 224, 3],
            "rtol": 2e-3,
            "atol": 2e-4,
            "max_abs": 0.0,
            "max_rel": 0.0,
        },
    )
    source = _write_flax_checkpoint(tmp_path / "source")
    cache_root = tmp_path / "cache"

    first = resolve_tactile_encoder(source, cache_root)
    first_weight_stat = first.weights_path.stat()
    second = resolve_tactile_encoder(source, cache_root)

    assert first == second
    assert second.weights_path.stat().st_mtime_ns == first_weight_stat.st_mtime_ns
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_path"] == str(source.resolve())
    assert metadata["source_sha256"] == first.source_sha256
    assert metadata["conversion_version"] == CONVERSION_VERSION
    assert metadata["architecture"] == "resnet18"
    assert metadata["embedding_dim"] == 512
    assert metadata["source_framework"] == "flax"
    assert metadata["target_framework"] == "pytorch"
    assert metadata["tensor_shapes"]["conv1.weight"] == [64, 3, 7, 7]
    assert metadata["parity"]["status"] == "passed"
    verify_resolved_tactile_encoder(first)

    _write_flax_checkpoint(source, seed=1)
    changed = resolve_tactile_encoder(source, cache_root)
    assert changed.weights_path != first.weights_path
    assert changed.source_sha256 != first.source_sha256

    module = TactileResNet18()
    load_tactile_encoder_weights(module, changed)
    assert torch.equal(
        module.conv1.weight,
        torch.from_numpy(
            np.arange(7 * 7 * 3 * 64, dtype=np.float32).reshape(7, 7, 3, 64).transpose(3, 2, 0, 1)
        ),
    )


def test_resolution_requires_explicit_trusted_sidecar_for_direct_safetensors(tmp_path: Path) -> None:
    source = tmp_path / "encoder.safetensors"
    save_file(TactileResNet18().state_dict(), str(source))

    with pytest.raises(ValueError, match="trusted sidecar"):
        resolve_tactile_encoder(source, tmp_path / "cache")

    create_trusted_tactile_encoder_sidecar(source)
    artifact = resolve_tactile_encoder(source, tmp_path / "cache")

    assert artifact.weights_path.exists()
    assert artifact.metadata_path.exists()
    module = TactileResNet18()
    verify_resolved_tactile_encoder(artifact)
    load_tactile_encoder_weights(module, artifact)



def test_cpu_jax_import_overrides_conflicting_platform_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "gpu")
    from train_deco.tactile_encoder_conversion import import_jax_flax_for_cpu

    jax, _ = import_jax_flax_for_cpu()

    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["JAX_PLATFORM_NAME"] == "cpu"
    assert jax.default_backend() == "cpu"

@pytest.mark.skipif(
    not Path("/home/typhon/FRS_Tact/checkpoints/encoder/encoder_ckpt_0824").is_dir(),
    reason="local tactile encoder checkpoint is unavailable",
)
def test_local_jax_and_converted_pytorch_encoders_match(tmp_path: Path) -> None:
    try:
        from train_deco.tactile_encoder_conversion import import_jax_flax_for_cpu

        import_jax_flax_for_cpu()
    except ImportError:
        pytest.skip("JAX/Flax conversion dependencies are unavailable")
    from train_encoder.utils.checkpoint import load_checkpoint
    from train_encoder.utils.resnet import encode_resnet18

    source = Path("/home/typhon/FRS_Tact/checkpoints/encoder/encoder_ckpt_0824")
    artifact = resolve_tactile_encoder(source, tmp_path / "cache")
    torch_model = TactileResNet18().eval()
    load_tactile_encoder_weights(torch_model, artifact)
    images_nhwc = np.linspace(0.0, 1.0, num=224 * 224 * 3, dtype=np.float32).reshape(1, 224, 224, 3)
    params, _ = load_checkpoint(source)
    jax_embedding, _ = encode_resnet18(
        params["tactile_resnet"], images_nhwc, train=False
    )
    with torch.inference_mode():
        torch_embedding = torch_model(torch.from_numpy(images_nhwc).permute(0, 3, 1, 2)).numpy()

    np.testing.assert_allclose(torch_embedding, np.asarray(jax_embedding), rtol=2e-3, atol=2e-4)



def test_direct_safetensors_source_bytes_produce_a_new_cache_artifact(tmp_path: Path) -> None:
    source = tmp_path / "encoder.safetensors"
    save_file(TactileResNet18().state_dict(), str(source))
    create_trusted_tactile_encoder_sidecar(source)
    first = resolve_tactile_encoder(source, tmp_path / "cache")

    changed_module = TactileResNet18()
    with torch.no_grad():
        changed_module.bn1.bias.fill_(1.0)
    save_file(changed_module.state_dict(), str(source))
    create_trusted_tactile_encoder_sidecar(source)
    changed = resolve_tactile_encoder(source, tmp_path / "cache")

    assert changed.weights_path != first.weights_path
    assert changed.source_sha256 != first.source_sha256


def test_checkpoint_directory_rejects_ambiguous_or_unsafe_parameter_discovery(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "params-one.npz").touch()
    (source / "params-two.npz").touch()
    (source / "checkpoint.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=r"exactly one params-\*\.npz"):
        resolve_tactile_encoder(source, tmp_path / "cache")

    (source / "checkpoint.json").write_text(
        json.dumps({"params_file": "../params-one.npz"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"simple params-\*\.npz filename"):
        resolve_tactile_encoder(source, tmp_path / "cache")



def test_cache_hit_rebuilds_when_parity_provenance_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fake_parity(source, state):
        nonlocal calls
        calls += 1
        return {
            "status": "passed",
            "seed": 1729,
            "input_shape": [4, 224, 224, 3],
            "rtol": 2e-3,
            "atol": 2e-4,
            "max_abs": 0.0,
            "max_rel": 0.0,
        }

    monkeypatch.setattr(conversion, "_verify_flax_pytorch_parity", fake_parity)
    source = _write_flax_checkpoint(tmp_path / "source")
    artifact = resolve_tactile_encoder(source, tmp_path / "cache")
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    metadata["parity"]["status"] = "missing"
    artifact.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    repaired = resolve_tactile_encoder(source, tmp_path / "cache")

    assert calls == 2
    verify_resolved_tactile_encoder(repaired)


def test_verify_rejects_artifact_when_its_source_digest_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        conversion,
        "_verify_flax_pytorch_parity",
        lambda source, state: {
            "status": "passed",
            "seed": 1729,
            "input_shape": [4, 224, 224, 3],
            "rtol": 2e-3,
            "atol": 2e-4,
            "max_abs": 0.0,
            "max_rel": 0.0,
        },
    )
    source = _write_flax_checkpoint(tmp_path / "source")
    artifact = resolve_tactile_encoder(source, tmp_path / "cache")
    _write_flax_checkpoint(source, seed=2)

    with pytest.raises(ValueError, match="resolved source contract"):
        verify_resolved_tactile_encoder(artifact)



def test_direct_resolution_accepts_a_converted_artifacts_verified_encoder_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        conversion,
        "_verify_flax_pytorch_parity",
        lambda source, state: {
            "status": "passed",
            "seed": 1729,
            "input_shape": [4, 224, 224, 3],
            "rtol": 2e-3,
            "atol": 2e-4,
            "max_abs": 0.0,
            "max_rel": 0.0,
        },
    )
    converted = resolve_tactile_encoder(_write_flax_checkpoint(tmp_path / "source"), tmp_path / "cache")

    direct = resolve_tactile_encoder(converted.weights_path, tmp_path / "direct-cache")

    verify_resolved_tactile_encoder(direct)



def test_verify_rejects_artifact_sha256_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        conversion,
        "_verify_flax_pytorch_parity",
        lambda source, state: {
            "status": "passed",
            "seed": 1729,
            "input_shape": [4, 224, 224, 3],
            "rtol": 2e-3,
            "atol": 2e-4,
            "max_abs": 0.0,
            "max_rel": 0.0,
        },
    )
    artifact = resolve_tactile_encoder(_write_flax_checkpoint(tmp_path / "source"), tmp_path / "cache")
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    metadata["weights_sha256"] = "not-the-artifact-digest"
    artifact.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256"):
        verify_resolved_tactile_encoder(artifact)
