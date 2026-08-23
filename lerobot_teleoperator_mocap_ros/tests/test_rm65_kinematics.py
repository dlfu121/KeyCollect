from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lerobot_robot_mujoco" / "lerobot_robot_mujoco"))

from rm65_kinematics import (  # noqa: E402
    RM65_HOME_Q,
    RM65_Q_MAX,
    RM65_Q_MIN,
    CartesianPoseTarget,
    damped_least_squares,
    forward_kinematics,
    jacobian_condition_number,
    quaternion_xyzw_to_matrix,
    rotation_matrix_to_quaternion_xyzw,
)


class RM65KinematicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        scene = ROOT / "assets" / "scenes" / "rm65_dexhand_scene.xml"
        cls.model = mujoco.MjModel.from_xml_path(str(scene))
        cls.body_id = mujoco.mj_name2id(cls.model, mujoco.mjtObj.mjOBJ_BODY, "link_6")
        cls.columns = np.array(
            [
                cls.model.jnt_dofadr[
                    mujoco.mj_name2id(cls.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{index}")
                ]
                for index in range(1, 7)
            ]
        )
        cls.base_position_world = np.array([-0.7, 0.0, 0.6])

    def flange_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = mujoco.MjData(self.model)
        data.qpos[:6] = q
        mujoco.mj_forward(self.model, data)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, data, jacp, jacr, self.body_id)
        return (
            data.xpos[self.body_id].copy(),
            data.xmat[self.body_id].reshape(3, 3).copy(),
            np.vstack([jacp, jacr])[:, self.columns],
        )

    def test_official_mdh_zero_pose(self) -> None:
        transform = forward_kinematics(np.zeros(6))
        np.testing.assert_allclose(transform[:3, 3], [0.0, 0.0, 0.879], atol=1e-12)
        np.testing.assert_allclose(transform[:3, :3], np.diag([-1.0, -1.0, 1.0]), atol=1e-12)

    def test_official_joint_limits_match_mujoco(self) -> None:
        model_limits = []
        for index in range(1, 7):
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{index}"
            )
            model_limits.append(self.model.jnt_range[joint_id])
        np.testing.assert_allclose(
            np.asarray(model_limits), np.column_stack([RM65_Q_MIN, RM65_Q_MAX]), atol=1e-9
        )

    def test_quaternion_interface_uses_xyzw_order(self) -> None:
        rotation = forward_kinematics(RM65_HOME_Q)[:3, :3]
        quaternion = rotation_matrix_to_quaternion_xyzw(rotation)
        np.testing.assert_allclose(quaternion_xyzw_to_matrix(quaternion), rotation, atol=1e-12)

    def test_mujoco_matches_official_mdh_for_random_joint_angles(self) -> None:
        rng = np.random.default_rng(65)
        for _ in range(100):
            q = rng.uniform(RM65_Q_MIN * 0.8, RM65_Q_MAX * 0.8)
            position_world, rotation_world, _ = self.flange_pose(q)
            official = forward_kinematics(q)
            np.testing.assert_allclose(
                position_world - self.base_position_world, official[:3, 3], atol=7e-6
            )
            np.testing.assert_allclose(rotation_world, official[:3, :3], atol=2e-5)

    def test_tcp_site_is_offset_from_rm65_flange(self) -> None:
        q = RM65_HOME_Q.copy()
        data = mujoco.MjData(self.model)
        data.qpos[:6] = q
        mujoco.mj_forward(self.model, data)
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.assertGreaterEqual(site_id, 0)
        flange_position = data.xpos[self.body_id]
        flange_rotation = data.xmat[self.body_id].reshape(3, 3)
        expected = flange_position + flange_rotation @ np.array([0.0, 0.0, -0.08])
        np.testing.assert_allclose(data.site_xpos[site_id], expected, atol=2e-6)
        np.testing.assert_allclose(data.site_xmat[site_id].reshape(3, 3), flange_rotation, atol=2e-6)

    def test_home_is_palm_down_and_clear_of_documented_singularities(self) -> None:
        position_world, rotation_world, jacobian = self.flange_pose(RM65_HOME_Q)
        np.testing.assert_allclose(rotation_world[:, 0], [0.0, 0.0, -1.0], atol=2e-5)
        self.assertGreater(position_world[2], 0.90)
        self.assertGreater(abs(RM65_HOME_Q[2]), np.deg2rad(30.0))
        self.assertGreater(abs(RM65_HOME_Q[4]), np.deg2rad(30.0))
        self.assertLess(jacobian_condition_number(jacobian), 30.0)

    def test_each_local_wrist_axis_converges_without_pose_crosstalk(self) -> None:
        for axis in np.eye(3):
            q = RM65_HOME_Q.copy()
            position_world, rotation_world, _ = self.flange_pose(q)
            target = CartesianPoseTarget.from_pose(position_world, rotation_world)
            target.integrate(np.zeros(3), axis * 0.10)

            for _ in range(80):
                current_position, current_rotation, jacobian = self.flange_pose(q)
                error = target.error(current_position, current_rotation)
                if np.linalg.norm(error) < 1e-7:
                    break
                q += np.clip(damped_least_squares(jacobian, error, 0.02), -0.08, 0.08)

            final_position, final_rotation, _ = self.flange_pose(q)
            final_error = target.error(final_position, final_rotation)
            self.assertLess(np.linalg.norm(final_error[:3]), 1e-6)
            self.assertLess(np.linalg.norm(final_error[3:]), 1e-6)


if __name__ == "__main__":
    unittest.main()
