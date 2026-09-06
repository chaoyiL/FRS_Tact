import numpy as np
import pytest
from reactive_diffusion_policy.deploy.bridge_client import RobotBridgeClient
import deploy_pick_tube_rdp as deploy

PROTOCOL='rdp_observation_deadline_v1'


def client(messages, started=True):
    c=object.__new__(RobotBridgeClient)
    c._send=lambda message: None
    c.send_config({'execution_protocol':PROTOCOL})
    if started: c.send_state('start')
    source=iter(messages)
    c._receive=lambda timeout=None: next(source)
    return c


def observation(deadline=100.215):
    obs={'observation.timestamp':100., 'observation.camera_timestamps':[100.,100.]}
    if deadline is not None: obs['observation.action_target_timestamp']=deadline
    return {'type':'obs','obs_seq':5,'obs':obs}


def ack(**updates):
    return dict(type='action_ack',obs_seq=5,status='scheduled',scheduled_count=1,
                target_timestamp=100.215,reference_source='observation',reference_timestamp=100.,reason=None)|updates


def test_deadline_protocol_selected_only_for_async_baseline():
    config={'control':{}}
    assert deploy.build_server_config(config,baseline=True,deadline=True)['execution_protocol']==PROTOCOL
    assert deploy.build_server_config(config,baseline=True)['execution_protocol']=='rdp_observation_step_v1'


def test_ack_confirms_exact_predicted_execution_time():
    c=client([observation(),ack()])
    c.receive_observation()
    assert c.receive_action_ack(5,1)['target_timestamp']==100.215


def test_silent_server_rescheduling_is_rejected():
    c=client([observation(),ack(target_timestamp=100.25)])
    c.receive_observation()
    with pytest.raises(RuntimeError,match='execution time'):
        c.receive_action_ack(5,1)


@pytest.mark.parametrize('deadline',[None,True,100.,99.,float('nan')])
def test_execution_observation_requires_a_future_finite_deadline(deadline):
    with pytest.raises(RuntimeError,match='action_target_timestamp'):
        client([observation(deadline)]).receive_observation()


def test_warmup_can_omit_execution_deadline():
    assert client([observation(None)],started=False).receive_observation()[0]==5


def test_missed_deadline_reports_no_execution_and_allows_next_observation():
    c=client([observation(),ack(status='rejected',scheduled_count=0,target_timestamp=None,
        reference_source=None,reference_timestamp=None,reason='rdp_execution_deadline_missed')])
    c.receive_observation()
    assert c.receive_action_ack(5,1)['status']=='rejected'


@pytest.mark.parametrize('fields',[
    {'scheduled_count':1}, {'target_timestamp':100.215}, {'reason':'tracking error'},
])
def test_only_well_formed_deadline_miss_is_recoverable(fields):
    c=client([observation(),ack(status='rejected',scheduled_count=0,target_timestamp=None,
        reference_source=None,reference_timestamp=None,reason='rdp_execution_deadline_missed',**{})|fields])
    c.receive_observation()
    with pytest.raises(RuntimeError): c.receive_action_ack(5,1)


def test_rejection_preserves_receipt_without_claiming_protocol_mismatch():
    message=ack(status='rejected',scheduled_count=0,target_timestamp=None,
        reference_source=None,reference_timestamp=None,
        reason='action translation delta exceeds the configured limit')
    c=client([message])
    with pytest.raises(RuntimeError) as error:
        c.receive_action_ack(5,1)
    assert c.last_action_receipt==message
    assert 'Both endpoints' not in str(error.value)
