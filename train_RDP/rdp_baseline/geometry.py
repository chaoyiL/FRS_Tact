"""Insert01 rotvec states and column-6D poses, matching original RDP geometry."""
import numpy as np
from scipy.spatial.transform import Rotation


def state7_to_matrix(state):
    state = np.asarray(state)
    matrix = np.broadcast_to(np.eye(4), (*state.shape[:-1], 4, 4)).copy()
    matrix[..., :3, 3] = state[..., :3]
    matrix[..., :3, :3] = Rotation.from_rotvec(state[..., 3:6].reshape(-1, 3)).as_matrix().reshape(*state.shape[:-1], 3, 3)
    return matrix


def pose10_to_matrix(pose):
    pose = np.asarray(pose)
    x = pose[..., 3:6]
    y = pose[..., 6:9]
    xnorm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(xnorm < 1e-8) or not np.isfinite(pose).all():
        raise ValueError('Invalid nonterminal action_raw rotation or nonfinite value')
    x = x / xnorm
    z = np.cross(x, y)
    znorm = np.linalg.norm(z, axis=-1, keepdims=True)
    if np.any(znorm < 1e-8):
        raise ValueError('Degenerate nonterminal action_raw rotation')
    z = z / znorm
    y = np.cross(z, x)
    matrix = np.broadcast_to(np.eye(4), (*pose.shape[:-1], 4, 4)).copy()
    matrix[..., :3, :3] = np.stack((x, y, z), axis=-1)
    matrix[..., :3, 3] = pose[..., :3]
    return matrix


def matrix_to_pose9(matrix):
    return np.concatenate((matrix[..., :3, 3], matrix[..., :3, :2].swapaxes(-1, -2).reshape(*matrix.shape[:-2], 6)), axis=-1).astype(np.float32)


def relative_to_base(matrix, base):
    """Broadcast a single fixed base per chunk; never integrate decoded steps."""
    inverse = np.broadcast_to(np.eye(4), base.shape).copy()
    inverse[..., :3, :3] = base[..., :3, :3].swapaxes(-1, -2)
    inverse[..., :3, 3] = -np.einsum('...ij,...j->...i', inverse[..., :3, :3], base[..., :3, 3])
    return inverse[..., None, :, :] @ matrix
