import pytest
import deploy_pick_tube_rdp as deploy


def test_baseline_defaults_to_separate_six_hz_planner():
    assert deploy.resolve_planning_config({'control_frequency': 30}, baseline=True) == ('asynchronous', 6.0)


def test_legacy_defaults_to_existing_synchronous_path():
    assert deploy.resolve_planning_config({}, baseline=False) == ('synchronous', 6.0)


def test_can_select_synchronous_baseline_for_comparison():
    assert deploy.resolve_planning_config({'planning_mode': 'synchronous'}, baseline=True)[0] == 'synchronous'


@pytest.mark.parametrize('control,baseline', [
    ({'planning_mode': 'unknown'}, True),
    ({'planning_mode': 'asynchronous'}, False),
    ({'ldp_inference_frequency': 0}, True),
    ({'ldp_inference_frequency': float('nan')}, True),
    ({'ldp_inference_frequency': 40, 'control_frequency': 30}, True),
    ({'ldp_inference_frequency': True}, True),
])
def test_invalid_planning_config_rejected(control, baseline):
    with pytest.raises(ValueError):
        deploy.resolve_planning_config(control, baseline=baseline)
