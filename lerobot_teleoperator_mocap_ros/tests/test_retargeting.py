import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lerobot_teleoperator_mocap_ros"))

from retargeting import (  # noqa: E402
    map_glove_to_hand_offsets,
    map_wrist_orientation,
    quaternion_multiply,
    map_wrist_position,
    quaternion_to_rotvec,
    rotvec_to_quaternion,
)


class RetargetingTest(unittest.TestCase):
    def test_legacy_position_axis_mapping(self) -> None:
        mapped = map_wrist_position(
            [2.0, 3.0, 4.0],
            [1.0, 1.0, 1.0],
            [0, 0, 1, 1, 0, 0, 0, 1, 0],
            0.01,
        )
        np.testing.assert_allclose(mapped, [0.03, 0.01, 0.02])

    def test_orientation_neutral_frame_maps_to_identity(self) -> None:
        quat = rotvec_to_quaternion([0.2, -0.1, 0.3])
        mapped = map_wrist_orientation(quat, quat, [-1, 0, 0, 0, 0, 1, 0, 1, 0])
        np.testing.assert_allclose(quaternion_to_rotvec(mapped), np.zeros(3), atol=1e-12)

    def test_rm65_orientation_maps_flip_swing_and_wave_independently(self) -> None:
        axis_map = [0, -1, 0, 0, 0, 1, -1, 0, 0]
        for mocap_axis, robot_axis in (
            ([0.2, 0.0, 0.0], [0.0, 0.0, -0.2]),
            ([0.0, 0.2, 0.0], [-0.2, 0.0, 0.0]),
            ([0.0, 0.0, 0.2], [0.0, 0.2, 0.0]),
        ):
            mapped = map_wrist_orientation(
                rotvec_to_quaternion(mocap_axis),
                [0.0, 0.0, 0.0, 1.0],
                axis_map,
            )
            np.testing.assert_allclose(quaternion_to_rotvec(mapped), robot_axis, atol=1e-12)

    def test_orientation_delta_stays_on_local_axis_after_nonidentity_calibration(self) -> None:
        axis_map = [0, -1, 0, 0, 0, 1, -1, 0, 0]
        initial = rotvec_to_quaternion([0.0, 0.0, np.pi / 2.0])
        motion = rotvec_to_quaternion([0.2, 0.0, 0.0])
        # The mocap wrist rotates around its local X after calibration; the
        # robot should receive the same mapped axis regardless of initial yaw.
        current = quaternion_multiply(initial, motion)
        mapped = map_wrist_orientation(current, initial, axis_map)
        np.testing.assert_allclose(
            quaternion_to_rotvec(mapped),
            [0.0, 0.0, -0.2],
            atol=1e-12,
        )

    def test_orientation_mapping_rejects_reflections(self) -> None:
        with self.assertRaisesRegex(ValueError, "orthogonal"):
            map_wrist_orientation(
                rotvec_to_quaternion([0.1, 0.0, 0.0]),
                [0.0, 0.0, 0.0, 1.0],
                [0, -1, 0, 0, 0, 1, 1, 0, 0.2],
            )

    def test_legacy_finger_indices_and_coupling(self) -> None:
        initial = np.zeros(57)
        current = np.zeros(57)
        for index in (1, 5, 8, 14, 17, 20, 26, 29, 32, 38, 41, 44, 50, 53, 56):
            current[index] = -index / 100.0
        mapped = map_glove_to_hand_offsets(current, initial, pip_dip_coupling=0.5)
        self.assertEqual(mapped.shape, (20,))
        np.testing.assert_allclose(
            mapped,
            [
                0.01, 0.05, 0.08, 0.04,
                0.0, 0.14, 0.185, 0.0925,
                0.0, 0.26, 0.305, 0.1525,
                0.0, 0.38, 0.425, 0.2125,
                0.0, 0.50, 0.545, 0.2725,
            ],
        )

    def test_legacy_mapping_requires_complete_57_value_frame(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 57 values"):
            map_glove_to_hand_offsets(np.zeros(56), np.zeros(57))


if __name__ == "__main__":
    unittest.main()
