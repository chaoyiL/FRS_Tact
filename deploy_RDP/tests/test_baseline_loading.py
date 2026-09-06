import copy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from reactive_diffusion_policy.common.pick_tube_action_contract import SINGLE_RIGHT_ARM_7X10


def configs():
    cfg = OmegaConf.create({
        'action_contract': 'single_right_chunk_relative10d_v1',
        'task': {'arms': 'right'}, 'horizon': 32,
        'n_obs_steps': 2, 'dataset_obs_temporal_downsample_ratio': 2,
        'n_action_steps': 29, 'tactile_pca_path': '/server/press/tactile_pca.npz',
        'shape_meta': {'action': {'shape': [10]},
            'obs': {'right_robot_tcp_pose': {'shape': [9]},
                    'right_robot_gripper_width': {'shape': [1]},
                    'tactile_embedding': {'shape': [15]}},
            'extended_obs': {'tactile_embedding': {'shape': [15]}}}})
    at_cfg = copy.deepcopy(cfg)
    cfg.shape_meta.obs.camera2 = {'shape': [3, 224, 224]}
    return cfg, at_cfg


def test_baseline_accepts_bimanual_pca_for_right_only_model():
    from baseline_loading import validate_baseline_metadata
    cfg, at_cfg = configs()
    validate_baseline_metadata(cfg, at_cfg, 30, SINGLE_RIGHT_ARM_7X10, 16)


@pytest.mark.parametrize('field,value', [
    ('action_contract', 'dual_arm_chunk_relative20d_v1'),
    ('shape_meta.obs.tactile_embedding.shape', [30]),
    ('horizon', 16),
    ('tactile_pca_path', '/server/insert/tactile_pca.npz'),
])
def test_baseline_rejects_incompatible_pair(field, value):
    from baseline_loading import validate_baseline_metadata
    cfg, at_cfg = configs()
    OmegaConf.update(at_cfg, field, value)
    with pytest.raises(ValueError):
        validate_baseline_metadata(cfg, at_cfg, 30, SINGLE_RIGHT_ARM_7X10, 16)


def test_embedded_at_must_match_selected_external_at_weights():
    from baseline_loading import validate_embedded_at
    weights = {key: {'weight': torch.ones(2)} for key in ('encoder', 'decoder', 'quant', 'post_quant')}
    weights['normalizer'] = {'not_compared': torch.ones(1)}
    selected = {'_extra_state': {'at': copy.deepcopy(weights)}}
    external = {'state_dicts': {'model': weights}}
    validate_embedded_at(selected, external)
    selected['_extra_state']['at']['decoder']['weight'][0] = 9
    with pytest.raises(ValueError, match='AT'):
        validate_embedded_at(selected, external)


def test_baseline_requests_observation_reference_protocol_and_checks_ack():
    import deploy_pick_tube_rdp as deploy
    from reactive_diffusion_policy.deploy.bridge_client import RobotBridgeClient
    cfg = deploy.load_config(Path(deploy.__file__).parent / 'configs/deploy_pick_tube_rdp_right.yaml')
    assert deploy.build_server_config(cfg, baseline=True)['execution_protocol'] == 'rdp_observation_step_v1'
    client = object.__new__(RobotBridgeClient)
    client._send = lambda message: None
    client.send_config(deploy.build_server_config(cfg, baseline=True))
    ack = dict(type='action_ack', obs_seq=1, status='scheduled', scheduled_count=1,
               target_timestamp=100., reference_timestamp=99., reference_source='observation')
    client._receive = lambda timeout=None: ack
    assert client.receive_action_ack(1, 1)['reference_source'] == 'observation'
    ack['reference_source'] = 'accepted_target'
    with pytest.raises(RuntimeError, match='reference'):
        client.receive_action_ack(1, 1)
