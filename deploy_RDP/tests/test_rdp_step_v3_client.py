"""No sockets or hardware: exercise timestamp and acceptance contracts."""
import numpy as np
import pytest
import torch
import importlib.util
from pathlib import Path

from reactive_diffusion_policy.deploy.bridge_client import RobotBridgeClient
_spec = importlib.util.spec_from_file_location('rdp_test_fixtures', Path(__file__).with_name('test_pick_tube_rdp_deploy.py'))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
deploy, FakePolicy, FakeTactileEncoder, observation = (
    _fixtures.deploy, _fixtures.FakePolicy, _fixtures.FakeTactileEncoder, _fixtures.observation,
)


def runtime():
    policy = FakePolicy(15)
    components = np.zeros((2, 15, 1024), dtype=np.float32)
    components[:, np.arange(15), np.arange(15)] = 1
    r = deploy.PickTubeRDPRuntime(
        policy, FakeTactileEncoder(), torch.device('cpu'),
        deploy.BimanualTactilePCA(np.zeros((2, 1024)), components), 16, 2, 2,
    )
    return r, policy


def stamped(tick):
    obs = observation(tick)
    obs['observation.timestamp'] = 1000.0 + tick / 30.0
    obs['observation.camera_timestamps'] = [1000.0 + tick / 30.0] * 2
    return obs


def test_elapsed_capture_time_skips_expired_decoder_step():
    r, p = runtime()
    for tick in [0, 1, 3]:
        r.predict(stamped(tick))
    assert p.fast_history_lengths == [4, 5, 7]


def test_replan_deadline_uses_capture_time_not_call_count():
    r, p = runtime()
    flags = [r.predict(stamped(t))[1] for t in [0, 8, 16]]
    assert flags == [True, False, True]
    assert p.fast_history_lengths == [4, 12, 4]


def test_visual_history_uses_capture_spacing_after_missing_frames():
    r, p = runtime()
    for tick in [0, 10, 12, 14, 16]:
        r.predict(stamped(tick))
    assert p.slow_observation_states[-1] == [14.0, 16.0]


@pytest.mark.parametrize('tick', [0, -1, float('nan'), float('inf')])
def test_bad_timestamp_does_not_advance_runtime(tick):
    r, p = runtime()
    r.predict(stamped(0))
    with pytest.raises(ValueError, match='timestamp'):
        r.predict(stamped(tick))
    assert r.step == 1
    assert p.fast_history_lengths == [4]


def test_long_gap_restarts_with_current_observation_padding():
    r, p = runtime()
    r.predict(stamped(0))
    r.predict(stamped(100))
    assert p.slow_observation_states[-1] == [100.0, 100.0]
    assert p.fast_history_lengths == [4, 4]


def test_new_frame_in_same_decoder_tick_replans_instead_of_repeating_old_action():
    r, p = runtime()
    r.predict(stamped(0))
    assert r.predict(stamped(.2))[1] is True
    assert p.slow_calls == 2


def bridge_with(messages):
    client = object.__new__(RobotBridgeClient)
    messages = iter(messages)
    client._receive = lambda timeout=None: next(messages)
    return client


def ack(**updates):
    return dict(type='action_ack', obs_seq=5, status='scheduled', scheduled_count=1,
                target_timestamp=1000.05, reason=None,
                reference_source='latest_measured', reference_timestamp=999.999) | updates


def test_ack_returns_actual_scheduling_receipt():
    c = bridge_with([ack()])
    receipt = c.receive_action_ack(5, timeout=1)
    assert receipt['scheduled_count'] == 1
    assert receipt['status'] == 'scheduled'


@pytest.mark.parametrize('message', [
    {'type': 'action_ack', 'obs_seq': 5},
    ack(status='rejected', scheduled_count=0, reason='stale'),
    ack(scheduled_count=0), ack(scheduled_count=True), ack(target_timestamp=float('nan')),
    ack(reference_timestamp=float('nan')), ack(obs_seq=6),
])
def test_legacy_empty_rejected_or_malformed_ack_is_not_success(message):
    c = bridge_with([message])
    with pytest.raises(RuntimeError):
        c.receive_action_ack(5, timeout=1)


@pytest.mark.parametrize('which', ['observation.timestamp', 'observation.camera_timestamps'])
def test_live_bridge_requires_sampling_timestamps(which):
    obs = stamped(0)
    del obs[which]
    c = bridge_with([{'type': 'obs', 'obs_seq': 1, 'obs': obs}])
    with pytest.raises(RuntimeError, match='timestamp'):
        c.receive_observation()


def test_live_bridge_rejects_reused_camera_frame_even_if_public_time_advances():
    first, second = stamped(0), stamped(1)
    second['observation.camera_timestamps'][1] = first['observation.camera_timestamps'][1]
    c = bridge_with([{'type': 'obs', 'obs_seq': 1, 'obs': first},
                     {'type': 'obs', 'obs_seq': 2, 'obs': second}])
    c.receive_observation()
    with pytest.raises(RuntimeError, match='timestamp'):
        c.receive_observation()


def test_live_contract_requests_step_v3():
    config = deploy.load_config(deploy.Path(__file__).resolve().parents[1] / 'configs/deploy_pick_tube_rdp_right.yaml')
    assert deploy.build_server_config(config)['execution_protocol'] == 'rdp_step_v3'
