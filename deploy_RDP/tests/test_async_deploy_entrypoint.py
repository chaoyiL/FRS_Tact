"""Exercise entrypoint ownership and shutdown without model or robot access."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

import deploy_pick_tube_rdp as deploy


@pytest.fixture
def entrypoint(monkeypatch, tmp_path):
    events = []
    artifacts = {}
    for name in ('ldp_checkpoint', 'at_checkpoint', 'tactile_pca_path'):
        path = tmp_path / name
        path.touch()
        artifacts[name] = str(path)
    encoder = tmp_path / 'encoder'
    encoder.mkdir()
    config = {
        'model': {**artifacts, 'tactile_encoder_dir': str(encoder), 'device': 'cpu'},
        'connection': {'address': 'not-a-real-robot', 'require_token': False},
        'control': {'slow_update_interval': 16, 'control_frequency': 30,
                    'ldp_inference_frequency': 7.5},
        'runtime': {'auto_start': True, 'warmup_runs': 0, 'max_iterations': 1,
                    'trace_dir': str(tmp_path / 'traces')},
    }
    state = SimpleNamespace(config=config, events=events, runtime=None,
                            create_error=None, config_error=None, stop_error=None,
                            bridge_close_error=None, runtime_close_error=None)

    class Runtime:
        kind = 'synchronous'

        def __init__(self, *args, **kwargs):
            self.options = kwargs
            self.last_decoder_tick = 0
            state.runtime = self
            events.append('runtime_created')

        def reset(self):
            events.append('runtime_reset')

        def predict(self, observation):
            events.append('predict')
            return np.zeros((1, 20), dtype=np.float32), False

        def close(self):
            # Stop must already have been delivered before a potentially
            # blocking worker join begins.
            events.append('runtime_close')
            if state.runtime_close_error:
                raise state.runtime_close_error

    class AsyncRuntime(Runtime):
        kind = 'asynchronous'

    class Bridge:
        def __init__(self, **kwargs):
            events.append('bridge_created')
            if state.create_error:
                raise state.create_error

        def send_config(self, payload):
            events.append('send_config')
            state.payload = payload
            if state.config_error:
                raise state.config_error

        def receive_observation(self):
            return 1, {'observation.timestamp': 1.0}

        def send_state(self, command):
            events.append(command)
            if command == 'stop' and state.stop_error:
                raise state.stop_error

        def send_action(self, action, obs_seq):
            events.append('send_action')

        def receive_action_ack(self, obs_seq, **kwargs):
            return {'status': 'scheduled', 'reference_source': 'observation',
                    'target_timestamp': 1.1}

        def close(self):
            events.append('bridge_close')
            if state.bridge_close_error:
                raise state.bridge_close_error

    monkeypatch.setattr(deploy, 'load_config', lambda _: config)
    monkeypatch.setattr(deploy, 'BimanualTactilePCA', SimpleNamespace(
        from_npz=lambda *args, **kwargs: SimpleNamespace(output_dim=30)))
    monkeypatch.setattr(deploy, 'load_policy', lambda *args, **kwargs: (
        object(), SimpleNamespace(dataset_obs_temporal_downsample_ratio=2, n_obs_steps=2)))
    monkeypatch.setattr(deploy, 'load_tactile_resnet18', lambda *args, **kwargs: object())
    monkeypatch.setattr(deploy, 'RobotBridgeClient', Bridge)
    monkeypatch.setitem(sys.modules, 'baseline_loading', SimpleNamespace(is_baseline_config=lambda _: True))
    monkeypatch.setitem(sys.modules, 'baseline_runtime', SimpleNamespace(BaselineRDPRuntime=Runtime))
    monkeypatch.setitem(sys.modules, 'async_baseline_runtime', SimpleNamespace(AsyncBaselineRDPRuntime=AsyncRuntime))
    state.run = lambda: deploy.run(tmp_path / 'unused.yaml')
    return state


def test_baseline_default_uses_async_frequency_and_stops_before_worker_join(entrypoint):
    entrypoint.run()
    assert entrypoint.runtime.kind == 'asynchronous'
    assert entrypoint.runtime.options['inference_fps'] == 7.5
    assert entrypoint.payload['execution_protocol'] == 'rdp_observation_deadline_v1'
    assert entrypoint.events.index('stop') < entrypoint.events.index('runtime_close')
    assert entrypoint.events.index('bridge_close') < entrypoint.events.index('runtime_close')
    assert entrypoint.events.count('runtime_close') == 1


def test_synchronous_baseline_does_not_receive_async_only_arguments(entrypoint):
    entrypoint.config['control']['planning_mode'] = 'synchronous'
    entrypoint.run()
    assert entrypoint.runtime.kind == 'synchronous'
    assert 'inference_fps' not in entrypoint.runtime.options


@pytest.mark.parametrize('failure', ['create_error', 'config_error'])
def test_bridge_setup_failure_closes_runtime_and_preserves_original_exception(entrypoint, failure):
    original = RuntimeError('initial bridge failure')
    setattr(entrypoint, failure, original)
    with pytest.raises(RuntimeError) as caught:
        entrypoint.run()
    assert caught.value is original
    assert entrypoint.events.count('runtime_close') == 1


@pytest.mark.parametrize('cleanup_failure', ['stop_error', 'bridge_close_error', 'runtime_close_error'])
def test_cleanup_failure_does_not_hide_send_config_failure(entrypoint, cleanup_failure):
    original = RuntimeError('configuration rejected')
    entrypoint.config_error = original
    setattr(entrypoint, cleanup_failure, ValueError('secondary cleanup failure'))
    with pytest.raises(RuntimeError) as caught:
        entrypoint.run()
    assert caught.value is original
    assert entrypoint.events.count('runtime_close') == 1


def test_worker_close_failure_does_not_hide_bridge_constructor_failure(entrypoint):
    original = RuntimeError('connection could not be constructed')
    entrypoint.create_error = original
    entrypoint.runtime_close_error = ValueError('worker cleanup failed')
    with pytest.raises(RuntimeError) as caught:
        entrypoint.run()
    assert caught.value is original
    assert entrypoint.events.count('runtime_close') == 1


def test_cleanup_failure_is_reported_when_main_loop_succeeded(entrypoint):
    original = ValueError('worker cleanup failed')
    entrypoint.runtime_close_error = original
    with pytest.raises(ValueError) as caught:
        entrypoint.run()
    assert caught.value is original
    assert entrypoint.events.index('stop') < entrypoint.events.index('runtime_close')


def test_rejected_action_is_traced_before_error_is_propagated(entrypoint, monkeypatch):
    import json
    receipt={'type':'action_ack','obs_seq':1,'status':'rejected','scheduled_count':0,
             'target_timestamp':None,'reason':'tracking error'}
    original=RuntimeError('tracking error')
    bridge_cls=deploy.RobotBridgeClient
    def reject(self, *args, **kwargs):
        self.last_action_receipt=receipt
        raise original
    monkeypatch.setattr(bridge_cls,'receive_action_ack',reject)
    with pytest.raises(RuntimeError) as caught:
        entrypoint.run()
    assert caught.value is original
    paths=list(__import__('pathlib').Path(entrypoint.config['runtime']['trace_dir']).glob('*.jsonl'))
    rows=[json.loads(x) for x in paths[0].read_text().splitlines()]
    action=next(r for r in rows if r['type']=='action')
    assert action['obs_seq']==1 and action['receipt']==receipt
    assert action['wire_action']
