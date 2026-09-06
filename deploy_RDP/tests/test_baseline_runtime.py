"""CPU coverage for fixed-chunk relative baseline deployment."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from baseline_runtime import BaselineRDPRuntime, matrix_to_pose9, pose9_to_matrix
from deploy_pick_tube_rdp import CAMERA_KEYS, TACTILE_KEYS, STATE_KEY


class Encoder:
    def __call__(self, images):
        return torch.zeros((4, 512))


class PCA:
    output_dim = 30
    def __call__(self, embeddings):
        return torch.arange(30, dtype=torch.float32)


class Policy:
    def __init__(self, arms):
        self.arms = arms
        self.observations = []
    def predict_action(self, obs, **kwargs):
        self.observations.append(obs)
        return {'action': torch.zeros((1, 1, 4))}
    def predict_from_latent_action(self, *args, **kwargs):
        action = np.tile([.1, 0, 0, 1, 0, 0, 0, 1, 0, .04], len(self.arms))
        return {'action': torch.tensor(action).reshape(1, 1, -1)}


def observation(x=0., angle=0.):
    state = np.zeros(20, dtype=np.float32)
    state[7] = x
    state[12] = angle
    state[6] = .02
    state[13] = .03
    return {STATE_KEY: state, **{key: np.zeros((224, 224, 3), dtype=np.uint8) for key in (*CAMERA_KEYS, *TACTILE_KEYS)}}


def runtime(arms='right'):
    return BaselineRDPRuntime(policy=Policy(('left', 'right') if arms == 'both' else ('right',)), tactile_encoder=Encoder(), device=torch.device('cpu'), tactile_pca=PCA(), slow_update_interval=4, dataset_obs_temporal_downsample_ratio=2, n_obs_steps=2, arms=arms)


def test_right_observations_select_exact_second_pca_arm():
    rt = runtime()
    obs, tactile = rt._prepare_observation(observation())
    assert set(obs) == {'camera2', 'right_robot_tcp_pose', 'right_robot_gripper_width', 'tactile_embedding'}
    np.testing.assert_array_equal(tactile.numpy().ravel(), np.arange(15, 30))


def test_dual_observations_keep_both_pca_arms():
    rt = runtime('both')
    obs, tactile = rt._prepare_observation(observation())
    assert 'camera1' in obs and 'left_robot_tcp_pose' in obs
    assert tactile.shape == (1, 1, 30)


def test_relative_history_and_fixed_slow_base():
    rt = runtime()
    rt.observation_history = [rt._prepare_observation(observation(x))[0] for x in (1., 2., 3., 4.)]
    slow = rt._slow_policy_observation()
    np.testing.assert_allclose(slow['right_robot_tcp_pose'][0, :, 0], [-2, 0])
    relative = np.array([.1, 0, 0, 1, 0, 0, 0, 1, 0, .04], dtype=np.float32)
    rt._prepare_observation(observation(5.))
    np.testing.assert_allclose(rt.absolute_action_target(relative)[:3], [4.1, 0, 0], atol=1e-6)
    rt.reset()
    with pytest.raises(RuntimeError, match='base'):
        rt.absolute_action_target(relative)


def test_absolute_target_composes_translation_and_rotation():
    rt = runtime()
    rt.observation_history = [rt._prepare_observation(observation(1., np.pi/2))[0]]
    rt._slow_policy_observation()
    delta = np.eye(4)
    delta[:3, 3] = [.2, 0, 0]
    delta[:3, :3] = Rotation.from_euler('x', .3).as_matrix()
    action = np.r_[matrix_to_pose9(delta), .07]
    absolute = rt.absolute_action_target(action)
    base = np.eye(4)
    base[:3, 3] = [1, 0, 0]
    base[:3, :3] = Rotation.from_euler('z', np.pi/2).as_matrix()
    np.testing.assert_allclose(pose9_to_matrix(absolute[:9]), base @ delta, atol=1e-6)
    assert absolute[-1] == pytest.approx(.07)


def test_fast_steps_use_current_observation_and_keep_fixed_chunk_target():
    rt = runtime()
    first, slow = rt.predict(observation(1., np.pi/2))
    assert slow
    np.testing.assert_allclose(first[0, :3], [.1, 0, 0], atol=1e-6)
    moved = observation(1.05, np.pi/3)
    next_action, slow = rt.predict(moved)
    assert not slow
    current = np.eye(4)
    current[:3, 3] = [1.05, 0, 0]
    current[:3, :3] = Rotation.from_euler('z', np.pi/3).as_matrix()
    expected = np.eye(4)
    expected[:3, 3] = [1., .1, 0]
    expected[:3, :3] = Rotation.from_euler('z', np.pi/2).as_matrix()
    np.testing.assert_allclose(current @ pose9_to_matrix(next_action[0, :9]), expected, atol=1e-6)
    rt.predict(observation(1.06))
    rt.predict(observation(1.07))
    replanned, slow = rt.predict(observation(2., -.4))
    assert slow
    np.testing.assert_allclose(replanned[0, :9], [.1, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-6)
    assert len(rt.policy.observations) == 2


def test_dual_wire_deltas_reconstruct_both_absolute_targets():
    rt = runtime('both')
    first = observation(1., .3)
    first[STATE_KEY][:3] = [0, 2, 1]
    first[STATE_KEY][3:6] = [.2, .3, .4]
    rt.predict(first)
    moved = observation(1.02, -.2)
    moved[STATE_KEY][:3] = [.1, 2.1, 1.1]
    moved[STATE_KEY][3:6] = [-.4, .1, .2]
    relative, _ = rt.predict(moved)
    decoded = np.tile([.1, 0, 0, 1, 0, 0, 0, 1, 0, .04], 2)
    expected = rt.absolute_action_target(decoded)
    for index, start in enumerate((0, 7)):
        current = np.eye(4)
        current[:3, 3] = moved[STATE_KEY][start:start+3]
        current[:3, :3] = Rotation.from_rotvec(moved[STATE_KEY][start+3:start+6]).as_matrix()
        offset = index * 10
        np.testing.assert_allclose(current @ pose9_to_matrix(relative[0, offset:offset+9]), pose9_to_matrix(expected[offset:offset+9]), atol=1e-6)


def test_timestamp_gap_resets_chunk_base_and_touch_context():
    rt = runtime()
    first = observation(1.)
    first['observation.timestamp'] = 10.
    rt.predict(first)
    next_obs = observation(3., .2)
    next_obs['observation.timestamp'] = 11.
    action, slow = rt.predict(next_obs)
    assert slow
    assert rt.step == 1
    assert len(rt.observation_history) == 1
    assert len(rt.tactile_history) == 4
    np.testing.assert_allclose(action[0, :9], [.1, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(rt.chunk_bases['right'][:3, 3], [3., 0, 0])
