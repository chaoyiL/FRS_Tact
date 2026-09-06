"""Deterministic CPU checks for concurrent planning and tactile execution."""
import importlib
import threading
import time

import numpy as np
import pytest
import torch

from baseline_runtime import matrix_to_pose9
from deploy_pick_tube_rdp import STATE_KEY


def runtime_class():
    try:
        module = importlib.import_module('async_baseline_runtime')
    except ModuleNotFoundError:
        pytest.fail('Async baseline runtime is not implemented')
    return module.AsyncBaselineRDPRuntime


class Policy:
    original_horizon = 32
    n_action_steps = 29
    def __init__(self):
        self.calls = []
        self.decoded = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail = False
        self.block = False
        self.concurrent = 0
        self.max_concurrent = 0
    def predict_action(self, obs, **kwargs):
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            self.calls.append(float(obs['tactile_embedding'][0, -1, 0]))
            if self.block:
                self.entered.set()
                assert self.release.wait(5), 'test did not release worker'
            if self.fail:
                raise ValueError('LDP failed deliberately')
            return {'action': torch.tensor([[[self.calls[-1]]]])}
        finally:
            self.concurrent -= 1
    def predict_from_latent_action(self, latent, extended_obs, **kwargs):
        touch = extended_obs['tactile_embedding']
        self.decoded.append((float(latent.item()), touch.clone(), kwargs['extended_obs_last_step']))
        action = torch.tensor([.1, 0, 0, 1, 0, 0, 0, 1, 0, float(touch[0, -1, 0])])
        return {'action': action.reshape(1, 1, 10)}


@pytest.fixture
def make_runtime():
    instances = []
    def make(interval=4):
        cls = runtime_class()
        policy = Policy()
        rt = cls(policy=policy, tactile_encoder=None, device=torch.device('cpu'),
                 tactile_pca=type('PCA', (), {'output_dim': 30})(),
                 slow_update_interval=interval, dataset_obs_temporal_downsample_ratio=2,
                 n_obs_steps=2, arms='right', control_frequency=30, inference_fps=15)
        def prepare(obs):
            matrix = np.eye(4)
            matrix[0, 3] = obs[STATE_KEY][7]
            touch = torch.full((1, 1, 15), obs['touch'])
            return {'right_robot_tcp_pose': torch.tensor(matrix_to_pose9(matrix)).reshape(1, 1, 9),
                    'tactile_embedding': touch}, touch
        rt._prepare_observation = prepare
        instances.append(rt)
        return rt, policy
    yield make
    for rt in instances:
        rt.policy.release.set()
        rt.close()


def obs(tick, x=0.):
    state = np.zeros(20)
    state[7] = x
    return {STATE_KEY: state, 'observation.timestamp': 100 + tick / 30, 'touch': float(tick)}


def wait_for(predicate):
    end = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < end, 'background planner did not finish'
        time.sleep(.001)


def test_blocked_ldp_does_not_block_fresh_tactile_execution(make_runtime):
    rt, policy = make_runtime()
    rt.predict(obs(0))
    policy.block = True
    rt.predict(obs(2, .2))
    assert policy.entered.wait(1)
    start = time.monotonic()
    action, adopted = rt.predict(obs(3, .3))
    assert time.monotonic() - start < .5
    assert not adopted
    assert action[0, -1] == 3
    assert rt.last_decoder_tick == 3
    assert policy.decoded[-1][2] == 7
    np.testing.assert_allclose(action[0, 0], -.2, atol=1e-6)


def test_adoption_keeps_snapshot_base_and_elapsed_decoder_tick(make_runtime):
    rt, policy = make_runtime()
    rt.predict(obs(0))
    policy.block = True
    rt.predict(obs(2, 2.))
    assert policy.entered.wait(1)
    rt.predict(obs(3, 3.))
    policy.release.set()
    wait_for(lambda: rt._completed_plan is not None)
    action, adopted = rt.predict(obs(4, 4.))
    assert adopted
    assert rt.plan_timestamp == pytest.approx(100 + 2 / 30)
    assert rt.last_decoder_tick == 2
    assert policy.decoded[-1][0] == 2
    assert policy.decoded[-1][2] == 6
    np.testing.assert_allclose(action[0, 0], -1.9, atol=1e-6)


def test_slow_worker_uses_latest_pending_snapshot_without_queue(make_runtime):
    rt, policy = make_runtime(interval=16)
    rt.predict(obs(0))
    policy.block = True
    rt.predict(obs(2))
    assert policy.entered.wait(1)
    for tick in (4, 6, 8):
        rt.predict(obs(tick))
    policy.release.set()
    wait_for(lambda: len(policy.calls) == 3)
    assert policy.calls == [0, 2, 8]
    assert policy.max_concurrent == 1


def test_exhausted_plan_stops_instead_of_reusing_last_step(make_runtime):
    rt, policy = make_runtime(interval=16)
    rt.predict(obs(0))
    policy.block = True
    rt.predict(obs(2))
    assert policy.entered.wait(1)
    rt.predict(obs(28))
    count = len(policy.decoded)
    with pytest.raises(RuntimeError, match='exhausted|expired'):
        rt.predict(obs(29))
    assert len(policy.decoded) == count


def test_reset_discards_running_old_episode_result(make_runtime):
    rt, policy = make_runtime()
    rt.predict(obs(0))
    policy.block = True
    rt.predict(obs(2, 20.))
    assert policy.entered.wait(1)
    rt.reset()
    policy.release.set()
    action, adopted = rt.predict(obs(100, 100.))
    assert adopted
    assert policy.decoded[-1][0] == 100
    assert rt.plan_timestamp == pytest.approx(100 + 100 / 30)
    np.testing.assert_allclose(action[0, 0], .1, atol=1e-5)
    assert policy.max_concurrent == 1


def test_worker_exception_propagates(make_runtime):
    rt, policy = make_runtime()
    rt.predict(obs(0))
    policy.fail = True
    rt.predict(obs(2))
    wait_for(lambda: rt._worker_error is not None)
    with pytest.raises(RuntimeError, match='planner') as error:
        rt.predict(obs(3))
    assert isinstance(error.value.__cause__, ValueError)


def test_close_stops_worker_and_rejects_further_predictions(make_runtime):
    rt, _ = make_runtime()
    rt.predict(obs(0))
    rt.close()
    assert not rt._worker.is_alive()
    with pytest.raises(RuntimeError, match='closed'):
        rt.predict(obs(1))


def test_completed_plan_waits_until_adoption_interval(make_runtime):
    rt, policy = make_runtime(interval=4)
    rt.predict(obs(0))
    rt.predict(obs(2, 2.))
    wait_for(lambda: rt._completed_plan is not None)
    action, adopted = rt.predict(obs(3, 3.))
    assert not adopted
    assert policy.decoded[-1][0] == 0
    np.testing.assert_allclose(action[0, 0], -2.9, atol=1e-6)


def test_rejects_duplicate_timestamp_before_decoding(make_runtime):
    rt, policy = make_runtime()
    rt.predict(obs(0))
    with pytest.raises(ValueError, match='strictly increase'):
        rt.predict(obs(0))
    assert len(policy.decoded) == 1


def test_untimed_offline_frames_use_same_action_age(make_runtime):
    rt, _ = make_runtime()
    first = obs(0)
    first.pop('observation.timestamp')
    rt.predict(first)
    second = obs(1)
    second.pop('observation.timestamp')
    rt.predict(second)
    assert rt.last_decoder_tick == 1


def test_stale_completed_plan_cannot_replace_exhausted_active_plan(make_runtime):
    rt, policy = make_runtime(interval=16)
    rt.predict(obs(0))
    rt.predict(obs(2))
    wait_for(lambda: rt._completed_plan is not None)
    policy.block = True
    with pytest.raises(RuntimeError, match='exhausted'):
        rt.predict(obs(31))
    assert len(policy.decoded) == 2


def test_fresh_plan_can_replace_expiring_plan_before_normal_interval(make_runtime):
    rt, policy = make_runtime(interval=16)
    rt.predict(obs(0))
    rt.predict(obs(2))
    wait_for(lambda: rt._completed_plan is not None)
    policy.block = True
    _, adopted = rt.predict(obs(28))
    assert adopted
    assert rt.last_decoder_tick == 26
    assert policy.entered.wait(1)
    policy.release.set()
    wait_for(lambda: rt._completed_plan is not None)
    # Prevent a further request so this checks the completed tick-28 plan.
    rt.inference_fps = .01
    _, adopted = rt.predict(obs(31))
    assert adopted
    assert rt.last_decoder_tick == 3
    assert policy.decoded[-1][0] == 28


def test_six_hz_cadence_is_stable_with_epoch_timestamps(make_runtime):
    rt, policy = make_runtime(interval=16)
    rt.inference_fps = 6
    policy.block = False
    start = 1788609335.280661
    requests = []
    for tick in range(16):
        frame = obs(tick)
        frame['observation.timestamp'] = start + tick / 30
        rt.predict(frame)
        if rt._last_request_timestamp == frame['observation.timestamp']:
            requests.append(tick)
    assert requests == [0, 5, 10, 15]


def test_cuda_worker_initialization_failure_is_reported(monkeypatch):
    def fail_stream(*args, **kwargs):
        raise RuntimeError('CUDA stream initialization failed')
    monkeypatch.setattr(torch.cuda, 'Stream', fail_stream)
    rt = runtime_class()(policy=Policy(), tactile_encoder=None, device=torch.device('cuda:0'),
                         tactile_pca=type('PCA', (), {'output_dim': 30})(),
                         slow_update_interval=16, dataset_obs_temporal_downsample_ratio=2,
                         n_obs_steps=2, arms='right')
    try:
        wait_for(lambda: rt._worker_error is not None)
        with pytest.raises(RuntimeError, match='planner') as error:
            rt.predict(obs(0))
        assert 'initialization failed' in str(error.value.__cause__)
        rt.reset()
        assert rt._worker_error is not None
        with pytest.raises(RuntimeError, match='planner'):
            rt.predict(obs(1))
    finally:
        rt.close()
