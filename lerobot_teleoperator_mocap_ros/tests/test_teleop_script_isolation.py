"""Regression tests keeping standalone teleop independent from recording."""

import sys
import unittest

from scripts import teleop


class TeleopScriptIsolationTest(unittest.TestCase):
    def test_standalone_control_calibration_is_explicit(self) -> None:
        previous_argv = sys.argv
        sys.argv = ["teleop.py", "--transport", "auto"]
        try:
            args = teleop.parse_args()
        finally:
            sys.argv = previous_argv

        config = teleop.build_teleop_config(args)
        self.assertEqual(config.transport, "auto")
        self.assertEqual(
            config.position_axis_map,
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        self.assertEqual(
            config.orientation_axis_map,
            [0.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
        )
        self.assertEqual(config.position_scale, 0.01)
        self.assertEqual(config.orientation_scale, 0.4)
        self.assertEqual(config.finger_scale, 1.0)
        self.assertEqual(config.finger_filter_alpha, 0.45)
        self.assertEqual(config.finger_deadband_rad, 0.003)
        self.assertEqual(config.finger_outlier_threshold, 0.40)
        self.assertEqual(config.max_translation_delta_m, 0.005)
        self.assertEqual(config.max_rotation_delta_rad, 0.04)
        self.assertEqual(config.max_finger_delta_rad, 0.05)
        self.assertEqual(config.expected_joint_values, 57)


if __name__ == "__main__":
    unittest.main()
