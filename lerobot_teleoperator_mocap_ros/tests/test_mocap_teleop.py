import time
import unittest

import numpy as np

from lerobot_teleoperator_mocap_ros import MocapRosTeleop, MocapRosTeleopConfig


class MocapTeleopRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.teleop = MocapRosTeleop(MocapRosTeleopConfig(stale_timeout_s=0.25))
        # Transport callbacks are tested independently of a live ROS master.
        self.teleop._connected = True

    def test_neutral_motion_and_stale_safety(self) -> None:
        neutral_fingers = np.zeros(57)
        self.teleop._update_wrist([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
        self.teleop._update_fingers(neutral_fingers)
        neutral = self.teleop.get_action()
        self.assertTrue(all(abs(float(value)) < 1e-12 for value in neutral.values()))

        moved_fingers = neutral_fingers.copy()
        moved_fingers[14] = -0.3
        self.teleop._update_wrist([2.0, 4.0, 6.0], [0.0, 0.0, 0.0, 1.0])
        self.teleop._update_fingers(moved_fingers)
        moved = self.teleop.get_action()
        self.assertAlmostEqual(moved["delta_x"], 0.02)  # clipped from mapped +0.03
        self.assertAlmostEqual(moved["delta_y"], 0.01)
        self.assertAlmostEqual(moved["delta_z"], 0.02)
        self.assertAlmostEqual(moved["r_f_joint2_2.delta"], 0.05)  # per-frame safety clip

        for _ in range(5):
            self.assertAlmostEqual(self.teleop.get_action()["r_f_joint2_2.delta"], 0.05)
        settled = self.teleop.get_action()
        self.assertAlmostEqual(settled["r_f_joint2_2.delta"], 0.0)

        self.teleop._last_wrist_time = time.monotonic() - 1.0
        self.teleop._last_finger_time = time.monotonic() - 1.0
        stale = self.teleop.get_action()
        self.assertTrue(all(abs(float(value)) < 1e-12 for value in stale.values()))


if __name__ == "__main__":
    unittest.main()
