from __future__ import annotations

import ast
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy_pi05"


def test_pi05_right_arm_profile_accepts_7d_state_and_10d_action() -> None:
    import yaml

    from deploy_pi05.deployment import load_deployment_config_bytes

    config = yaml.safe_load((DEPLOY / "configs/deploy_pi05.yaml").read_bytes())
    config = copy.deepcopy(config)
    config["model"].update(
        state_action_profile="single-right-arm-7x10",
        state_dim=7,
        robot_action_dim=10,
        action_dim=10,
        camera_map={
            "right_wrist_0_rgb": "observation.images.camera1",
        },
    )
    config["observation"].update(
        single_arm_mode=True,
        controlled_arm="right",
    )

    loaded = load_deployment_config_bytes(
        yaml.safe_dump(config).encode(), mode="pi05"
    )

    assert loaded["model"]["state_dim"] == 7
    assert loaded["model"]["robot_action_dim"] == 10


def test_checked_in_pi05_right_config_uses_only_real_right_wrist_camera() -> None:
    from deploy_pi05.deployment import load_deployment_config

    loaded = load_deployment_config(
        DEPLOY / "configs/deploy_pi05_right.yaml", mode="pi05"
    )

    assert loaded["model"]["state_action_profile"] == "single-right-arm-7x10"
    assert loaded["model"]["camera_map"] == {
        "right_wrist_0_rgb": "observation.images.camera1",
    }
    assert "black_camera0" not in loaded["observation"]


def test_pi05_rejects_deprecated_black_camera0() -> None:
    import yaml

    from deploy_pi05.deployment import load_deployment_config_bytes

    config = yaml.safe_load((DEPLOY / "configs/deploy_pi05.yaml").read_bytes())
    config["observation"]["black_camera0"] = True

    with pytest.raises(ValueError, match="black_camera0.*not supported"):
        load_deployment_config_bytes(yaml.safe_dump(config).encode(), mode="pi05")


def test_single_right_pi05_rejects_extra_left_camera_slot() -> None:
    import yaml

    from deploy_pi05.deployment import load_deployment_config_bytes

    config = yaml.safe_load(
        (DEPLOY / "configs/deploy_pi05_right.yaml").read_bytes()
    )
    config["model"]["camera_map"] = {
        "left_wrist_0_rgb": "observation.images.camera0",
        "right_wrist_0_rgb": "observation.images.camera1",
    }

    with pytest.raises(ValueError, match="single-right-arm.*only.*right_wrist_0_rgb"):
        load_deployment_config_bytes(yaml.safe_dump(config).encode(), mode="pi05")


def test_single_right_server_projection_keeps_bimanual_wire_mode() -> None:
    from deploy_pi05.deployment import load_deployment_config, make_server_config

    config = load_deployment_config(
        DEPLOY / "configs/deploy_pi05_right.yaml", mode="pi05"
    )

    projected = make_server_config(config, mode="pi05")

    assert projected["single_arm_mode"] is False
    assert projected["action_horizon"] == 50


def test_pi05_right_arm_profile_rejects_dual_arm_dimensions() -> None:
    import yaml

    from deploy_pi05.deployment import load_deployment_config_bytes

    config = yaml.safe_load((DEPLOY / "configs/deploy_pi05.yaml").read_bytes())
    config["model"]["state_action_profile"] = "single-right-arm-7x10"
    config["observation"].update(single_arm_mode=True, controlled_arm="right")

    with pytest.raises(ValueError, match="model.state_dim must be 7"):
        load_deployment_config_bytes(yaml.safe_dump(config).encode(), mode="pi05")


def test_pi05_frs_right_arm_uses_both_tactile_faces_of_arm_1() -> None:
    import yaml

    from deploy_pi05.deployment import load_deployment_config_bytes

    config = yaml.safe_load((DEPLOY / "configs/deploy_pi05_frs.yaml").read_bytes())
    config["task"] = 0
    config["model"].update(
        state_action_profile="single-right-arm-7x10",
        state_dim=7,
        robot_action_dim=10,
        action_dim=10,
        camera_map={
            "right_wrist_0_rgb": "observation.images.camera1",
        },
    )
    config["observation"].update(single_arm_mode=True, controlled_arm="right")
    config["frs"]["tactile_keys"] = [
        "observation.images.tactile_left_1",
        "observation.images.tactile_right_1",
    ]

    loaded = load_deployment_config_bytes(
        yaml.safe_dump(config).encode(), mode="frs"
    )

    assert loaded["frs"]["tactile_keys"] == config["frs"]["tactile_keys"]


def test_checked_in_pi05_frs_right_config_uses_new_two_face_checkpoint() -> None:
    from deploy_pi05.deployment import load_deployment_config

    loaded = load_deployment_config(
        DEPLOY / "configs/deploy_pi05_frs_right.yaml", mode="frs"
    )

    assert loaded["frs"]["tactile_keys"] == [
        "observation.images.tactile_left_1",
        "observation.images.tactile_right_1",
    ]
    assert loaded["frs"]["checkpoint"] == (
        "/home/typhon/FRS_Tact/checkpoints/model/"
        "pi05_frs_right_insert_01_two_face"
    )


def test_pi05_frs_right_arm_rejects_legacy_cross_arm_tactile_pair() -> None:
    import yaml

    from deploy_pi05.deployment import load_deployment_config_bytes

    config = yaml.safe_load(
        (DEPLOY / "configs/deploy_pi05_frs_right.yaml").read_bytes()
    )
    config["frs"]["tactile_keys"] = [
        "observation.images.tactile_right_0",
        "observation.images.tactile_right_1",
    ]

    with pytest.raises(ValueError, match="tactile_left_1.*tactile_right_1"):
        load_deployment_config_bytes(yaml.safe_dump(config).encode(), mode="frs")


def test_pi05_frs_right_arm_rejects_noncanonical_tactile_key_prefix() -> None:
    import yaml

    from deploy_pi05.deployment import load_deployment_config_bytes

    config = yaml.safe_load(
        (DEPLOY / "configs/deploy_pi05_frs_right.yaml").read_bytes()
    )
    config["frs"]["tactile_keys"] = [
        "bogus.tactile_left_1",
        "bogus.tactile_right_1",
    ]

    with pytest.raises(ValueError, match="tactile_left_1.*tactile_right_1"):
        load_deployment_config_bytes(yaml.safe_dump(config).encode(), mode="frs")


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
        "deploy_pi05_frs_right.yaml",
        "deploy_pi05_right.yaml",
    }
    assert {path.name for path in (DEPLOY / "scripts").glob("*.sh")} == {
        "start_pi05.sh",
        "start_pi05_frs.sh",
        "start_pi05_right_frs.sh",
        "start_pi05_right.sh",
        "start_remote_client.sh",
    }


def test_pi05_right_frs_launcher_selects_checked_in_right_config() -> None:
    environment = os.environ.copy()
    environment.pop("PI05_DEPLOY_CONFIG", None)
    environment.pop("PI05_FRS_DEPLOY_CONFIG", None)

    completed = subprocess.run(
        [
            "bash",
            str(DEPLOY / "scripts/start_pi05_right_frs.sh"),
            "--check",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "mode=frs" in completed.stdout
    assert f"config={DEPLOY / 'configs/deploy_pi05_frs_right.yaml'}" in completed.stdout


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


def test_plain_pi05_server_config_requests_fixed_224_observation_profile() -> None:
    from deploy_pi05.deployment import make_server_config

    server_config = make_server_config(
        {
            "observation": {
                "data_type": "vision",
                "language_prompt": "pick up the tubes",
                "single_arm_mode": False,
                "no_state_obs_mode": False,
            },
            "control": {
                "control_frequency": 10.0,
                "controller_frequency": 80.0,
                "steps_per_inference": 50,
                "action_horizon": 50,
            },
        },
        mode="pi05",
    )

    assert server_config["observation_profile"] == "pi05_vision_224"


def test_pi05_frs_server_config_requests_fixed_224_vitac_profile() -> None:
    from deploy_pi05.deployment import make_server_config

    server_config = make_server_config(
        {
            "observation": {
                "data_type": "vitac",
                "language_prompt": "pick up the tubes",
                "single_arm_mode": False,
                "no_state_obs_mode": False,
            },
            "control": {
                "control_frequency": 10.0,
                "controller_frequency": 80.0,
                "steps_per_inference": 50,
                "action_horizon": 50,
            },
        },
        mode="frs",
        frs_runtime=SimpleNamespace(
            config=SimpleNamespace(steering_protection_interval_s=None),
            tactile_keys=("left", "right"),
        ),
    )

    assert server_config["observation_profile"] == "pi05_vitac_224"


def _policy_for_model_input() -> object:
    from deploy_pi05.policy import Pi05DeploymentConfig, Pi05RemotePolicy

    policy = object.__new__(Pi05RemotePolicy)
    policy.config = Pi05DeploymentConfig(
        checkpoint="checkpoint",
        assets_dir="assets",
        asset_id="asset",
        camera_map={
            "left_wrist_0_rgb": "observation.images.camera0",
            "right_wrist_0_rgb": "observation.images.camera1",
        },
        empty_cameras=(),
    )
    return policy


def test_pi05_model_input_accepts_two_final_224_uint8_images() -> None:
    policy = _policy_for_model_input()
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    result = policy._model_input(
        {
            "observation.state": np.zeros(20, dtype=np.float32),
            "observation.images.camera0": image,
            "observation.images.camera1": image,
        },
        "pick up the tubes",
    )

    assert set(result["image"]) == {"left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(value.shape == (224, 224, 3) for value in result["image"].values())
    assert all(value.dtype == np.uint8 for value in result["image"].values())


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((256, 256, 3), dtype=np.uint8),
        np.zeros((224, 224, 3), dtype=np.float32),
    ],
)
@pytest.mark.parametrize("camera_key", ["observation.images.camera0", "observation.images.camera1"])
def test_pi05_model_input_rejects_nonfinal_image_contract(
    image: np.ndarray, camera_key: str
) -> None:
    policy = _policy_for_model_input()
    observation = {
        "observation.state": np.zeros(20, dtype=np.float32),
        "observation.images.camera0": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation.images.camera1": np.zeros((224, 224, 3), dtype=np.uint8),
    }
    observation[camera_key] = image

    with pytest.raises(ValueError, match=r"\(224, 224, 3\).*uint8"):
        policy._model_input(observation, "pick up the tubes")


def test_pi05_deployment_input_transform_does_not_resize_final_images() -> None:
    from deploy_pi05.policy import Pi05RemotePolicy

    assert "ResizeImages" not in inspect.getsource(Pi05RemotePolicy.__init__)
