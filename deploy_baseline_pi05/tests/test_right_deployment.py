from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from deploy_baseline_pi05.deployment import load_deployment_config, make_server_config
from deploy_baseline_pi05.runtime import DirectDecoderRuntime

RIGHT_KEYS = ('observation.images.tactile_left_1', 'observation.images.tactile_right_1')


def test_right_config_keeps_twenty_dimensional_wire(tmp_path):
    raw = yaml.safe_load(Path('deploy_baseline_pi05/configs/deploy_baseline_pi05.yaml').read_text())
    raw['source'].update(action_dim=10, state_dim=7, model_action_dim=10,
                         camera_map={'right_wrist_0_rgb': 'observation.images.camera1'})
    raw['direct_decoder'].update(action_dim=10, tactile_keys=list(RIGHT_KEYS))
    raw['tactile_encoder'].update(tactile_keys=list(RIGHT_KEYS), key_map=dict(zip(RIGHT_KEYS, ('observation.images.tactile_left_1', RIGHT_KEYS[1]))))
    raw['observation']['single_arm_mode'] = True
    path = tmp_path / 'right.yaml'; path.write_text(yaml.safe_dump(raw))
    config = load_deployment_config(path)
    assert config.source.state_dim == 7
    assert config.direct_decoder.action_dim == 10
    wire = make_server_config(config)
    assert wire['single_arm_mode'] is False
    assert wire['frs_tactile_keys'] == ['observation.images.tactile_left_1', RIGHT_KEYS[1]]
    raw['observation']['single_arm_mode'] = False
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='single_arm_mode'):
        load_deployment_config(path)


def test_right_runtime_projects_state_and_holds_left_from_each_request():
    seen = []
    class Policy:
        def predict_action_chunk(self, observation, task, **kwargs):
            seen.append(observation['observation.state'].copy())
            return np.zeros((1, 50, 10), np.float32)
        def unnormalize_actions(self, actions):
            return np.asarray(actions, np.float32) + 0.1
    class Encoder:
        tactile_keys = RIGHT_KEYS
        key_map = {RIGHT_KEYS[0]: 'observation.images.tactile_left_1', RIGHT_KEYS[1]: RIGHT_KEYS[1]}
        def encode(self, observation):
            assert self.key_map[RIGHT_KEYS[0]] in observation
            return np.ones((1, 2, 512), np.float32)
    class Decoder:
        def decode(self, coarse, tactile):
            assert coarse.shape == (1, 50, 10)
            assert tactile.shape == (1, 2, 512)
            return np.full_like(coarse, 0.2)
    obs = {'observation.state': np.arange(20, dtype=np.float32) / 100,
           'observation.images.tactile_left_1': np.zeros((2, 2, 3), np.uint8),
           RIGHT_KEYS[1]: np.zeros((2, 2, 3), np.uint8)}
    runtime = DirectDecoderRuntime(policy=Policy(), tactile_encoder=Encoder(), decoder=Decoder(),
                                   action_dim=10, max_normalized_action_abs=8, max_normalized_delta_rms=4)
    ready = runtime.begin_chunk(1, obs, 'insert')
    np.testing.assert_array_equal(seen[0], obs['observation.state'][7:14])
    assert ready.action_vla_normalized.shape == (1, 50, 10)
    assert ready.action_vla.shape == (1, 50, 20)
    obs['observation.state'][6] = 0.4
    result = runtime.steer_action(1, 1, obs, 0)
    np.testing.assert_array_equal(result.selected_action[:9], [0,0,0,1,0,0,0,1,0])
    assert result.selected_action[9] == np.float32(0.4)
    np.testing.assert_allclose(result.selected_action[10:], 0.3)
    assert runtime.steer_action(1, 1, obs, 0) is result
    # A changed left grip affects the outgoing command, even with identical tactile.
    obs['observation.state'][6] = 0.5
    with pytest.raises(ValueError, match='conflicting duplicate'):
        runtime.steer_action(1, 1, obs, 0)
    result2 = runtime.steer_action(1, 2, obs, 1)
    assert result2.selected_action[9] == np.float32(0.5)


def test_right_decoder_rejects_legacy_cross_arm_sensor_pair():
    from deploy_baseline_pi05.direct_decoder import DirectDecoderConfig

    DirectDecoderConfig(action_dim=10, tactile_keys=RIGHT_KEYS).validate()
    with pytest.raises(ValueError, match='canonical order'):
        DirectDecoderConfig(
            action_dim=10,
            tactile_keys=('observation.images.tactile_right_0', 'observation.images.tactile_right_1'),
        ).validate()
