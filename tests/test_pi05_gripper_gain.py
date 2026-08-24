import unittest

import numpy as np

from deploy_pi05.frs_runtime import _apply_gripper_gain


class Pi05GripperGainTest(unittest.TestCase):
    def test_multiplies_only_widths_below_threshold(self):
        action = np.zeros(20, dtype=np.float32)
        action[[9, 19]] = (0.09, 0.11)
        original = action.copy()

        adjusted = _apply_gripper_gain(action, threshold=0.1, gain=0.7)

        np.testing.assert_allclose(adjusted[[9, 19]], (0.063, 0.11), atol=1e-7)
        np.testing.assert_array_equal(action, original)


if __name__ == "__main__":
    unittest.main()
