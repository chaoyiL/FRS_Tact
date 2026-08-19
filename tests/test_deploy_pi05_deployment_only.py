from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy_pi05"


def test_deploy_pi05_contains_no_training_or_analysis_trees() -> None:
    forbidden = {
        "modalities_eval",
        "tactile_encoder",
        "tests",
        "tools",
        "train_pi05_frs",
    }
    present = {path.name for path in DEPLOY.iterdir() if path.is_dir()}
    assert forbidden.isdisjoint(present)

    forbidden_files = {
        "pi05_frs_plan.md",
        "pi05_tools_manifest.sha256",
        "prepare_pi05.py",
        "vendor_manifest.sha256",
    }
    assert all(not (DEPLOY / name).exists() for name in forbidden_files)


def test_only_deployment_configs_and_launchers_remain() -> None:
    assert {path.name for path in (DEPLOY / "configs").glob("*.yaml")} == {
        "deploy_pi05.yaml",
        "deploy_pi05_frs.yaml",
    }
    assert {path.name for path in (DEPLOY / "scripts").glob("*.sh")} == {
        "start_pi05.sh",
        "start_pi05_frs.sh",
        "start_remote_client.sh",
    }


def test_deployment_sources_do_not_import_removed_training_packages() -> None:
    forbidden_roots = {"modalities_eval", "tactile_encoder", "train_pi05_frs"}
    for source in DEPLOY.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        assert forbidden_roots.isdisjoint(imported), source


def test_minimal_frs_runtime_helpers_preserve_tactile_math() -> None:
    from deploy_pi05.frs_inference.tactile import (
        resolve_tactile_window,
        tactile_change_from_tokens,
    )

    assert resolve_tactile_window(action_horizon=50, window_divisor=5) == 10
    baseline = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    current = np.asarray([[[0.0, 1.0], [0.0, 1.0]]], dtype=np.float32)
    np.testing.assert_allclose(tactile_change_from_tokens(current, baseline), [0.5])


def test_tactile_checkpoint_loader_keeps_hashed_file_support(tmp_path: Path) -> None:
    from deploy_pi05.frs_inference.encoder_checkpoint import load_tactile_encoder

    metadata = {
        "parameter_paths": ["tactile_resnet/params/kernel"],
        "tactile_clip_config": {"embedding_dim": 512, "tactile_image_size": 224},
    }
    (tmp_path / "checkpoint.json").write_text(json.dumps(metadata), encoding="utf-8")
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.savez(tmp_path / "params-deadbeef.npz", p00000=expected)

    bundle = load_tactile_encoder(tmp_path)

    np.testing.assert_array_equal(bundle.params["tactile_resnet"]["params"]["kernel"], expected)
    assert bundle.metadata == metadata


def test_tactile_preprocessing_is_deployment_local() -> None:
    from deploy_pi05.frs_inference.preprocess import parse_image_to_unit

    image = np.full((2, 2, 3), 255, dtype=np.uint8)
    output = parse_image_to_unit(image, image_size=2)
    assert output.shape == (2, 2, 3)
    assert output.dtype == np.float32
    np.testing.assert_array_equal(output, np.ones_like(output))


def test_frs_inference_module_exposes_no_training_entrypoints() -> None:
    from deploy_pi05.frs_inference import decoder

    forbidden = {
        "flow_matching_loss_per_sample",
        "make_optimizer",
        "save_checkpoint",
        "train_step",
    }
    assert forbidden.isdisjoint(vars(decoder))


def test_frs_decoder_constructs_with_deployment_flax() -> None:
    from flax import nnx
    import jax.numpy as jnp

    from deploy_pi05.frs_inference.decoder import (
        DecoderConfig,
        TactileConditionedFlowDecoder,
    )

    config = DecoderConfig(
        action_dim=20,
        action_horizon=2,
        tactile_window=1,
        gru_hidden_dim=8,
        resnet_embedding_dim=8,
        model_dim=8,
        depth=1,
        num_heads=2,
        num_tactile_tokens=4,
    )
    model = TactileConditionedFlowDecoder(config, rngs=nnx.Rngs(0))
    assert len(model.blocks) == 1
    output = model(
        jnp.zeros((1, 2, 20), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.zeros((1, 1, 4, 8), dtype=jnp.float32),
    )
    assert output.shape == (1, 2, 20)


def test_frs_decoder_checkpoint_loader_uses_runtime_parameter_tree(tmp_path: Path) -> None:
    import dataclasses

    import jax
    from flax import nnx

    from deploy_pi05.frs_inference.decoder import (
        DecoderConfig,
        TactileConditionedFlowDecoder,
    )
    from deploy_pi05.frs_inference.decoder_checkpoint import (
        _flat_parameter_state,
        _path_name,
        load_checkpoint,
    )

    config = DecoderConfig(
        action_dim=20,
        action_horizon=2,
        tactile_window=1,
        gru_hidden_dim=8,
        resnet_embedding_dim=8,
        model_dim=8,
        depth=1,
        num_heads=2,
        num_tactile_tokens=4,
    )
    model = TactileConditionedFlowDecoder(config, rngs=nnx.Rngs(0))
    _, flat = _flat_parameter_state(model)
    ordered = sorted(flat.items(), key=lambda item: _path_name(item[0]))
    np.savez(
        tmp_path / "params.npz",
        **{
            f"p{index:05d}": np.asarray(jax.device_get(value))
            for index, (_, value) in enumerate(ordered)
        },
    )
    metadata = {
        "decoder_config": dataclasses.asdict(config),
        "parameter_paths": [_path_name(path) for path, _ in ordered],
    }
    (tmp_path / "checkpoint.json").write_text(json.dumps(metadata), encoding="utf-8")

    restored, restored_metadata = load_checkpoint(tmp_path)

    assert restored.config == config
    assert restored_metadata == metadata


def test_frs_decoder_loader_accepts_legacy_none_parameter_slot(tmp_path: Path) -> None:
    import dataclasses

    import jax
    from flax import nnx, traverse_util

    from deploy_pi05.frs_inference.decoder import (
        DecoderConfig,
        TactileConditionedFlowDecoder,
    )
    from deploy_pi05.frs_inference.decoder_checkpoint import _path_name, load_checkpoint

    config = DecoderConfig(
        action_dim=20,
        action_horizon=2,
        tactile_window=1,
        gru_hidden_dim=8,
        resnet_embedding_dim=8,
        model_dim=8,
        depth=1,
        num_heads=2,
        num_tactile_tokens=4,
    )
    model = TactileConditionedFlowDecoder(config, rngs=nnx.Rngs(0))
    state = nnx.state(model, nnx.Param)
    full = traverse_util.flatten_dict(state.to_pure_dict())
    ordered = sorted(full.items(), key=lambda item: _path_name(item[0]))
    assert sum(value is None for _, value in ordered) == 1
    np.savez(
        tmp_path / "params.npz",
        **{
            f"p{index:05d}": (
                np.asarray(None) if value is None else np.asarray(jax.device_get(value))
            )
            for index, (_, value) in enumerate(ordered)
        },
    )
    metadata = {
        "decoder_config": dataclasses.asdict(config),
        "parameter_paths": [_path_name(path) for path, _ in ordered],
    }
    (tmp_path / "checkpoint.json").write_text(json.dumps(metadata), encoding="utf-8")

    restored, restored_metadata = load_checkpoint(tmp_path)

    assert restored.config == config
    assert restored_metadata == metadata


def test_pi05_model_tree_is_inference_only() -> None:
    training = DEPLOY / "src/lerobot/policies/pi05_jax/training"
    assert not training.exists()
    assert (DEPLOY / "src/lerobot/policies/pi05_jax/sharding.py").is_file()
    assert not (DEPLOY / "src/lerobot/datasets").exists()
    assert not (DEPLOY / "src/lerobot/processor").exists()
