import numpy as np

from deploy_deco.bread_phase_client import prepare_phase_action
from deploy_deco.bread_phase_controller import BreadPhaseController


def _observation(right_gripper: float) -> dict:
    state = np.zeros(20, dtype=np.float32)
    state[6] = 0.31
    state[13] = right_gripper
    return {"observation.state": state}


def test_prepare_phase_action_updates_phase_once_and_masks_the_whole_chunk():
    controller = BreadPhaseController()
    action = np.ones((32, 20), dtype=np.float32)

    phase, filtered = prepare_phase_action(
        controller, _observation(0.08), action, now_s=0.0
    )
    assert phase == 0
    assert np.all(filtered[:, 0:3] == 0)

    prepare_phase_action(controller, _observation(0.10), action, now_s=0.1)
    assert controller.phase == 0
    phase, filtered = prepare_phase_action(
        controller, _observation(0.11), action, now_s=0.2
    )
    assert phase == 1
    assert np.all(filtered[:, 10:13] == 0)
    assert np.all(filtered[:, 19] == np.float32(0.12))
