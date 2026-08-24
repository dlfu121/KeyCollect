import time
import unittest

import numpy as np

from lerobot_teleoperator_mocap_ros import MocapRosTeleop, MocapRosTeleopConfig


class MocapTeleopRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.teleop = MocapRosTeleop(
            MocapRosTeleopConfig(
                stale_timeout_s=0.25,
                finger_filter_alpha=1.0,
                finger_deadband_rad=0.0,
                finger_outlier_threshold=1.0,
            )
        )
        # Transport callbacks are tested independently of a live ROS master.
        self.teleop._connected = True

    def test_neutral_motion_and_stale_safety(self) -> None:
        neutral_fingers = np.zeros(57)
        self.teleop._update_wrist([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
        self.teleop._update_fingers(neutral_fingers)
        neutral = self.teleop.get_action()
        self.assertTrue(all(abs(float(value)) < 1e-12 for value in neutral.values()))

        moved_fingers = neutral_fingers.copy()
        moved_fingers[14] = 0.3
        self.teleop._update_wrist([2.0, 4.0, 6.0], [0.0, 0.0, 0.0, 1.0])
        self.teleop._update_fingers(moved_fingers)
        moved = self.teleop.get_action()
        self.assertAlmostEqual(
            moved["delta_x"], self.teleop.config.max_translation_delta_m
        )  # clipped from mapped +0.03
        self.assertAlmostEqual(moved["delta_y"], self.teleop.config.max_translation_delta_m)
        self.assertAlmostEqual(moved["delta_z"], self.teleop.config.max_translation_delta_m)
        finger_step = self.teleop.config.max_finger_delta_rad
        self.assertAlmostEqual(moved["r_f_joint2_2.delta"], finger_step)

        total_finger_steps = int(np.ceil(0.3 / finger_step))
        for _ in range(total_finger_steps - 1):
            self.assertAlmostEqual(self.teleop.get_action()["r_f_joint2_2.delta"], finger_step)
        settled = self.teleop.get_action()
        self.assertAlmostEqual(settled["r_f_joint2_2.delta"], 0.0)

        self.teleop._last_wrist_time = time.monotonic() - 1.0
        self.teleop._last_finger_time = time.monotonic() - 1.0
        stale = self.teleop.get_action()
        self.assertTrue(all(abs(float(value)) < 1e-12 for value in stale.values()))


if __name__ == "__main__":
    unittest.main()
