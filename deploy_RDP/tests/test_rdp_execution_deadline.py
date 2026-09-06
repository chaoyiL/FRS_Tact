"""Execution-time indexing and unchanged history with hold-last extrapolation."""
import json

import numpy as np
import pytest

from deploy_RDP.tests.test_async_baseline_runtime import make_runtime, obs, wait_for


def at_deadline(tick, delay=.215, x=0.):
    frame = obs(tick, x)
    frame['observation.action_target_timestamp'] = frame['observation.timestamp'] + delay
    return frame


def test_215ms_execution_delay_advances_decoder_six_or_seven_ticks(make_runtime):
    rt, policy = make_runtime(interval=16)
    rt.predict(at_deadline(0))
    assert rt.last_decoder_tick == 6
    assert policy.decoded[-1][2] == 10
    assert rt.last_diagnostics['capture_tick'] == 0
    assert rt.last_diagnostics['target_tick'] == 6
    assert rt.last_diagnostics['lookahead_ticks'] == 6
    assert rt.last_diagnostics['lookahead_seconds'] == pytest.approx(.215)
    rt.predict(at_deadline(1, .225))
    assert rt.last_decoder_tick == 8
    assert rt.last_diagnostics['lookahead_ticks'] == 7
    json.dumps(rt.last_diagnostics, allow_nan=False)


@pytest.mark.parametrize('deadline', [True, float('nan'), float('inf'), 99., 100.])
def test_invalid_execution_timestamp_is_rejected_before_planning(make_runtime, deadline):
    rt, policy = make_runtime()
    frame = obs(0)
    frame['observation.action_target_timestamp'] = deadline
    with pytest.raises(ValueError, match='action_target_timestamp'):
        rt.predict(frame)
    assert not policy.calls


def test_deadline_preserves_legacy_history_prefix_and_holds_future(make_runtime):
    rt, policy = make_runtime(interval=16)
    legacy, legacy_policy = make_runtime(interval=16)
    rt.predict(at_deadline(0))
    legacy.predict(obs(0))
    rt.predict(at_deadline(1.8))
    legacy.predict(obs(1.8))
    values = policy.decoded[-1][1][0, :, 0].numpy()
    prefix = legacy_policy.decoded[-1][1][0, :, 0].numpy()
    np.testing.assert_array_equal(values[:len(prefix)], prefix)
    np.testing.assert_allclose(values[len(prefix):], 1.8)
    # Preserve nearest-history sampling: tick 1 selects known capture 1.8.
    assert values[4] == pytest.approx(1.8)
    assert values.shape == (12,)


def test_decode_does_not_read_cached_observations_after_capture(make_runtime):
    rt, policy = make_runtime(interval=16)
    rt.predict(at_deadline(0))
    rt.predict(at_deadline(1.8))
    expected = policy.decoded[-1][1].clone()
    # Replay callers may have prefetched future observations: these must never
    # condition an action belonging to the earlier capture.
    future, _ = rt._prepare_observation(obs(2.1))
    rt.observation_history.append(future)
    rt.observation_timestamps.append(obs(2.1)['observation.timestamp'])
    current_touch = rt.observation_history[-2]['tactile_embedding']
    rt._decode_at(rt.latent_action, rt.plan_timestamp,
                  at_deadline(1.8)['observation.action_target_timestamp'],
                  current_touch, causal=True,
                  capture_timestamp=obs(1.8)['observation.timestamp'])
    np.testing.assert_array_equal(policy.decoded[-1][1].numpy(), expected.numpy())


def test_switch_compares_old_and_new_at_identical_execution_time(make_runtime):
    rt, policy = make_runtime(interval=4)
    rt.predict(at_deadline(0))
    rt.predict(at_deadline(2, x=2.))
    wait_for(lambda: rt._completed_plan is not None)
    rt.inference_fps = .01
    rt.predict(at_deadline(4, x=4.))
    diagnostic = rt.last_diagnostics
    switch = diagnostic['plan_switch']
    assert diagnostic['adopted']
    assert switch['execution_timestamp'] == at_deadline(4)['observation.action_target_timestamp']
    assert switch['old']['target_tick'] == 10
    assert switch['new']['target_tick'] == 8
    assert switch['old']['absolute_action'][0] == pytest.approx(.1)
    assert switch['new']['absolute_action'][0] == pytest.approx(2.1)
    assert switch['new']['latent'] == [[2.]]
    json.dumps(diagnostic, allow_nan=False)


def test_capture_time_controls_adoption_cadence_not_deadline_tick(make_runtime):
    rt, policy = make_runtime(interval=16)
    rt.predict(at_deadline(0))
    rt.predict(at_deadline(2))
    wait_for(lambda: rt._completed_plan is not None)
    rt.inference_fps = .01
    _, adopted = rt.predict(at_deadline(10))
    assert rt.last_decoder_tick == 16
    assert not adopted


def test_execution_horizon_forces_early_valid_plan_adoption(make_runtime):
    rt, policy = make_runtime(interval=16)
    rt.predict(at_deadline(0))
    rt.predict(at_deadline(2))
    wait_for(lambda: rt._completed_plan is not None)
    rt.inference_fps = .01
    # Capture cadence only 10 steps, but execution is at active-plan tick 28.
    _, adopted = rt.predict(at_deadline(10, delay=.6))
    assert adopted
    assert rt.last_decoder_tick == 26


def test_candidate_expired_at_execution_time_is_not_adopted(make_runtime):
    rt, policy = make_runtime(interval=16)
    rt.predict(at_deadline(0))
    rt.predict(at_deadline(2))
    wait_for(lambda: rt._completed_plan is not None)
    rt.inference_fps = .01
    with pytest.raises(RuntimeError, match='exhausted'):
        rt.predict(at_deadline(25))


def test_bootstrap_deadline_beyond_horizon_reports_exhaustion(make_runtime):
    rt, policy = make_runtime()
    with pytest.raises(RuntimeError, match='exhausted'):
        rt.predict(at_deadline(0, delay=1.))
    assert not policy.decoded
