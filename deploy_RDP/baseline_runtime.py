"""0906 baseline observations and fixed-chunk SE(3) execution adapter.

This module deliberately uses the deployment RDP package, without putting the
training project's duplicate reactive_diffusion_policy package on sys.path.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from deploy_pick_tube_rdp import PickTubeRDPRuntime, SINGLE_RIGHT_ARM_7X10, DUAL_ARM_20X20, STATE_KEY


def matrix_to_pose9(matrix):
    matrix = np.asarray(matrix)
    return np.concatenate((matrix[..., :3, 3], matrix[..., :3, :2].swapaxes(-1, -2).reshape(*matrix.shape[:-2], 6)), axis=-1).astype(np.float32)


def pose9_to_matrix(pose):
    pose = np.asarray(pose, dtype=np.float64)
    x, y = pose[..., 3:6], pose[..., 6:9]
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(pose).all() or np.any(norm < 1e-8):
        raise ValueError('Invalid baseline pose rotation')
    x = x / norm
    z = np.cross(x, y)
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('Degenerate baseline pose rotation')
    z = z / norm
    matrix = np.broadcast_to(np.eye(4), (*pose.shape[:-1], 4, 4)).copy()
    matrix[..., :3, :3] = np.stack((x, np.cross(z, x), z), axis=-1)
    matrix[..., :3, 3] = pose[..., :3]
    return matrix


def state_to_matrix(state):
    matrix = np.eye(4)
    matrix[:3, 3] = state[:3]
    matrix[:3, :3] = Rotation.from_rotvec(state[3:6]).as_matrix()
    return matrix


class BaselineRDPRuntime(PickTubeRDPRuntime):
    """Reuse native timing/touch history, with baseline observation/action frames."""

    def __init__(self, *args, arms=None, **kwargs):
        if arms is not None:
            if arms not in ('right', 'both'):
                raise ValueError('arms must be right or both')
            kwargs['profile'] = SINGLE_RIGHT_ARM_7X10 if arms == 'right' else DUAL_ARM_20X20
        super().__init__(*args, **kwargs)
        self.arms = ('right',) if self.profile == SINGLE_RIGHT_ARM_7X10 else ('left', 'right')
        if self.tactile_pca.output_dim != 30:
            raise ValueError('0906 baseline requires a two-arm PCA with 15 components per arm')

    def reset(self):
        super().reset()
        self.chunk_bases = {}

    def _prepare_observation(self, observation):
        prepared, tactile = super()._prepare_observation(observation)
        if self.arms == ('right',):
            tactile = tactile[..., 15:30]
        result = {'camera2': prepared['camera2'], 'tactile_embedding': tactile}
        if 'left' in self.arms:
            result['camera1'] = prepared['camera1']
        state = np.asarray(observation[STATE_KEY], dtype=np.float64)
        for arm in self.arms:
            start = 0 if arm == 'left' else 7
            pose = matrix_to_pose9(state_to_matrix(state[start:start+7]))
            result[arm+'_robot_tcp_pose'] = torch.from_numpy(pose).to(self.device).reshape(1, 1, 9)
            result[arm+'_robot_gripper_width'] = torch.tensor(state[start+6], dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        return result, tactile

    def _slow_policy_observation(self):
        result = super()._slow_policy_observation()
        for arm in self.arms:
            key = arm+'_robot_tcp_pose'
            latest = self.observation_history[-1][key][0, 0].detach().cpu().numpy()
            base = pose9_to_matrix(latest)
            self.chunk_bases[arm] = base
            poses = pose9_to_matrix(result[key].detach().cpu().numpy())
            relative = matrix_to_pose9(np.linalg.inv(base) @ poses)
            result[key] = torch.from_numpy(relative).to(self.device)
        return result

    def absolute_action_target(self, action):
        """Recover absolute TCP targets from the unchanged slow-inference base."""
        action = np.asarray(action)
        if action.shape != (10 * len(self.arms),) or not np.isfinite(action).all():
            raise ValueError('Invalid baseline decoded action shape or values')
        targets = []
        for index, arm in enumerate(self.arms):
            if arm not in self.chunk_bases:
                raise RuntimeError('Baseline chunk base is not initialized')
            decoded = action[index*10:(index+1)*10]
            target = self.chunk_bases[arm] @ pose9_to_matrix(decoded[:9])
            targets.append(np.r_[matrix_to_pose9(target), decoded[9]])
        return np.concatenate(targets).astype(np.float32)

    def _action_for_execution(self, action, observation):
        """Encode against the captured observation used by rdp_observation_step_v1."""
        targets = self.absolute_action_target(action)
        state = np.asarray(observation[STATE_KEY], dtype=np.float64)
        if state.shape != (20,) or not np.isfinite(state).all():
            raise ValueError('Expected finite 20D bridge observation state')
        deltas = []
        for index, arm in enumerate(self.arms):
            start = 0 if arm == 'left' else 7
            reference = state_to_matrix(state[start:start+7])
            target = targets[index*10:(index+1)*10]
            delta = np.linalg.inv(reference) @ pose9_to_matrix(target[:9])
            deltas.append(np.r_[matrix_to_pose9(delta), target[9]])
        return np.concatenate(deltas).astype(np.float32)
