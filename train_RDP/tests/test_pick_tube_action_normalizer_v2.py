import numpy as np
from hydra import compose, initialize_config_dir
from pathlib import Path

from reactive_diffusion_policy.common.normalize_util import get_action_normalizer
from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer
from reactive_diffusion_policy.dataset.real_image_tactile_dataset import RealImageTactileDataset


IDENTITY_6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)


def _actions(count=257):
    rng = np.random.default_rng(7)
    actions = np.zeros((count, 20), dtype=np.float64)
    actions[:, :3] = rng.normal(size=(count, 3)) * np.array([0.002, 0.004, 0.008])
    actions[:, 3:9] = IDENTITY_6D + rng.normal(scale=0.02, size=(count, 6))
    actions[:, 9] = np.linspace(0.012, 0.078, count)
    actions[:, 10:13] = rng.normal(size=(count, 3)) * np.array([0.003, 0.006, 0.009])
    actions[:, 13:19] = IDENTITY_6D + rng.normal(scale=0.02, size=(count, 6))
    actions[:, 19] = np.linspace(0.021, 0.067, count)
    return actions


def test_zero_centered_v2_maps_bimanual_physical_noop_to_exact_zero():
    actions = _actions()
    normalizer = get_action_normalizer(
        actions,
        bimanual_contiguous=True,
        version="zero_centered_v2",
    )
    noop = np.zeros(20, dtype=np.float64)
    noop[3:9] = IDENTITY_6D
    noop[13:19] = IDENTITY_6D
    noop[[9, 19]] = actions[0, [9, 19]]

    normalized = normalizer.normalize(noop).detach().cpu().numpy()

    np.testing.assert_array_equal(normalized[:9], np.zeros(9))
    np.testing.assert_array_equal(normalized[10:19], np.zeros(9))


def test_zero_centered_v2_round_trip_and_parameters_match_contract():
    actions = _actions()
    normalizer = get_action_normalizer(
        actions,
        bimanual_contiguous=True,
        version="zero_centered_v2",
    )

    normalized = normalizer.normalize(actions)
    restored = normalizer.unnormalize(normalized).detach().cpu().numpy()

    assert np.max(np.abs(restored - actions)) < 1e-7
    scale = normalizer.params_dict["scale"].detach().cpu().numpy()
    offset = normalizer.params_dict["offset"].detach().cpu().numpy()
    for position_slice in (slice(0, 3), slice(10, 13)):
        expected = 1.0 / np.maximum(
            np.quantile(np.abs(actions[:, position_slice]), 0.995, axis=0),
            1e-7,
        )
        np.testing.assert_allclose(scale[position_slice], expected, rtol=1e-12, atol=0)
        np.testing.assert_array_equal(offset[position_slice], np.zeros(3))
    np.testing.assert_array_equal(scale[3:9], np.ones(6))
    np.testing.assert_array_equal(offset[3:9], -IDENTITY_6D)
    np.testing.assert_array_equal(scale[13:19], np.ones(6))
    np.testing.assert_array_equal(offset[13:19], -IDENTITY_6D)


def test_zero_centered_v2_keeps_grippers_range_normalized():
    actions = _actions()
    normalizer = get_action_normalizer(
        actions,
        bimanual_contiguous=True,
        version="zero_centered_v2",
    )
    normalized = normalizer.normalize(actions).detach().cpu().numpy()

    for gripper_index in (9, 19):
        assert np.isclose(normalized[:, gripper_index].min(), -1.0)
        assert np.isclose(normalized[:, gripper_index].max(), 1.0)


def test_legacy_v1_remains_the_default():
    actions = _actions()

    default = get_action_normalizer(actions, bimanual_contiguous=True)
    explicit = get_action_normalizer(
        actions,
        bimanual_contiguous=True,
        version="legacy_v1",
    )

    np.testing.assert_array_equal(
        default.params_dict["scale"].detach().cpu().numpy(),
        explicit.params_dict["scale"].detach().cpu().numpy(),
    )
    np.testing.assert_array_equal(
        default.params_dict["offset"].detach().cpu().numpy(),
        explicit.params_dict["offset"].detach().cpu().numpy(),
    )


def test_zero_centered_v2_rejects_non_bimanual_action_layout():
    actions = np.zeros((8, 10), dtype=np.float32)

    try:
        get_action_normalizer(actions, version="zero_centered_v2")
    except ValueError as exc:
        assert "20D bimanual" in str(exc)
    else:
        raise AssertionError("zero_centered_v2 accepted a non-20D action layout")


def test_pick_tube_training_configs_select_v2_normalization_and_physical_loss():
    config_dir = str(
        Path(__file__).resolve().parents[1] / "reactive_diffusion_policy" / "config"
    )
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        at_cfg = compose(config_name="train_pick_tube_at_workspace")
        ldp_cfg = compose(config_name="train_pick_tube_ldp_workspace")

    assert at_cfg.task.dataset.action_normalizer_version == "zero_centered_v2"
    assert ldp_cfg.task.dataset.action_normalizer_version == "zero_centered_v2"
    assert at_cfg.policy.action_loss_version == "physical_v2"
    assert ldp_cfg.policy.at.action_loss_version == "physical_v2"
    assert ldp_cfg.task.dataset.at.action_loss_version == "physical_v2"
    for vae_cfg in (at_cfg.policy, ldp_cfg.policy.at, ldp_cfg.task.dataset.at):
        assert vae_cfg.idle_weight == 1.0
        assert vae_cfg.position_scale == 1e-3
        assert vae_cfg.rotation_scale == np.deg2rad(1.0)
        assert vae_cfg.gripper_scale == 5e-3
        assert vae_cfg.idle_position_scale == 1e-4
        assert vae_cfg.idle_rotation_scale == np.deg2rad(0.05)


def test_dataset_v2_normalizer_excludes_invalid_action_targets(tmp_path):
    dataset_path = tmp_path / "v2"
    dataset_path.mkdir()
    actions = np.zeros((4, 20), dtype=np.float64)
    actions[:, 3:9] = IDENTITY_6D
    actions[:, 13:19] = IDENTITY_6D
    actions[:, :3] = [[0.001, 0.002, 0.003], [0.002, 0.004, 0.006], [1, 1, 1], [2, 2, 2]]
    actions[:, 10:13] = actions[:, :3] * 2
    actions[:, 9] = [0.01, 0.02, 0.03, 0.04]
    actions[:, 19] = [0.02, 0.03, 0.04, 0.05]
    replay_buffer = ReplayBuffer.create_empty_numpy()
    replay_buffer.add_episode({
        "action": actions,
        "action_valid": np.array([True, True, False, False]),
        "idle_arm_mask": np.zeros((4, 2), dtype=bool),
    })
    replay_buffer.save_to_path(dataset_path / "replay_buffer.zarr")
    dataset = RealImageTactileDataset(
        shape_meta={"obs": {}, "action": {"shape": [20]}},
        dataset_path=str(dataset_path),
        horizon=1,
        use_episode_repeats=False,
        bimanual_contiguous_action=True,
        action_normalizer_version="zero_centered_v2",
    )

    normalizer = dataset.get_normalizer()["action"]
    scale = normalizer.params_dict["scale"].detach().cpu().numpy()
    expected = 1.0 / np.quantile(np.abs(actions[:2, :3]), 0.995, axis=0)

    np.testing.assert_allclose(scale[:3], expected)
