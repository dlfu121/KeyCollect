from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lerobot_robot_mujoco"))

try:
    from lerobot_robot_mujoco import MuJoCoRobot, MuJoCoRobotConfig  # noqa: E402
    from lerobot_teleoperator_mocap_ros.config_mocap_ros import RIGHT_HAND_JOINTS  # noqa: E402

    LEROBOT_AVAILABLE = True
except ModuleNotFoundError as error:
    if error.name != "lerobot":
        raise
    MuJoCoRobot = None
    MuJoCoRobotConfig = None
    RIGHT_HAND_JOINTS = [f"r_f_joint{finger}_{joint}" for finger in range(1, 6) for joint in range(1, 5)]
    LEROBOT_AVAILABLE = False


class KeyCollectSceneIntegrationTest(unittest.TestCase):
    @staticmethod
    def load_home() -> tuple[mujoco.MjModel, mujoco.MjData]:
        scene = ROOT / "assets" / "scenes" / "rm65_dexhand_scene.xml"
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        mujoco.mj_forward(model, data)
        return model, data

    def test_home_pose_has_palm_facing_down(self) -> None:
        model, data = self.load_home()

        hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_6")
        hand_rotation = data.xmat[hand_body_id].reshape(3, 3)
        palm_normal = hand_rotation[:, 0]  # DexHand fingers flex toward local +X.
        np.testing.assert_allclose(palm_normal, [0.0, 0.0, -1.0], atol=1e-5)
        self.assertGreater(data.xpos[hand_body_id, 2], 0.80)
        np.testing.assert_allclose(
            data.qpos[:6], np.deg2rad([0.0, -90.0, 45.0, 0.0, -45.0, 180.0]), atol=1e-6
        )

    def test_front_camera_is_fixed_and_wrist_camera_follows_link_6(self) -> None:
        model, _ = self.load_home()
        table_camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "table_camera"
        )
        wrist_camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_overhead_camera"
        )
        link_6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_6")

        self.assertEqual(model.cam_bodyid[table_camera_id], 0)
        self.assertEqual(model.cam_bodyid[wrist_camera_id], link_6_id)

    def test_metric_depth_render_has_expected_shape_and_range(self) -> None:
        if not LEROBOT_AVAILABLE:
            self.skipTest("lerobot is not installed in this Python environment")

        from lerobot_robot_mujoco.simulation import MuJoCoSimulation

        scene = ROOT / "assets" / "scenes" / "rm65_dexhand_scene.xml"
        sim = MuJoCoSimulation(str(scene))
        sim.load()
        sim.reset()
        try:
            depth = sim.render_depth_camera("table_camera", width=160, height=120)
            self.assertEqual(depth.shape, (120, 160, 1))
            self.assertEqual(depth.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(depth)))
            self.assertGreater(float(depth.min()), 0.0)
            self.assertGreater(float(depth.max()), float(depth.min()))
        finally:
            sim.close()

    def test_robot_uses_position_servos_and_holds_home(self) -> None:
        model, data = self.load_home()
        self.assertEqual(model.nu, 26)
        np.testing.assert_allclose(data.ctrl, data.qpos[: model.nu], atol=1e-12)
        robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_1")
        robot_body_ids = np.flatnonzero(model.body_rootid == robot_body_id)
        np.testing.assert_allclose(model.body_gravcomp[robot_body_ids], 1.0)
        self.assertEqual(model.body_gravcomp[0], 0.0)
        np.testing.assert_allclose(model.opt.gravity, [0.0, 0.0, -9.81])
        for actuator_id in range(model.nu):
            kp = model.actuator_gainprm[actuator_id, 0]
            bias = model.actuator_biasprm[actuator_id]
            self.assertGreater(kp, 0.0)
            self.assertAlmostEqual(bias[1], -kp)
            self.assertLessEqual(bias[2], 0.0)

        home_qpos = data.qpos.copy()
        hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_6")
        home_hand_position = data.xpos[hand_body_id].copy()
        for _ in range(round(10.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)

        np.testing.assert_allclose(data.qpos[:6], home_qpos[:6], atol=1e-5)
        # Fingers initialized exactly at their lower limits can settle slightly
        # under joint/contact constraints without moving the arm or palm.
        np.testing.assert_allclose(data.qpos[6:26], home_qpos[6:26], atol=0.005)
        np.testing.assert_allclose(data.xpos[hand_body_id], home_hand_position, atol=1e-5)
        self.assertGreater(data.xpos[hand_body_id, 2], 0.80)

    def test_all_mapped_hand_joints_exist_and_accept_deltas(self) -> None:
        scene = ROOT / "assets" / "scenes" / "rm65_dexhand_scene.xml"
        model = mujoco.MjModel.from_xml_path(str(scene))
        missing = [
            name
            for name in RIGHT_HAND_JOINTS
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0
        ]
        self.assertEqual(missing, [])

        if not LEROBOT_AVAILABLE:
            self.skipTest("lerobot is not installed in this Python environment")

        robot = MuJoCoRobot(
            MuJoCoRobotConfig(
                id="mocap_test",
                scene_path=str(scene),
                arm_joint_names=[f"joint_{index}" for index in range(1, 7)],
                gripper_joint_names=list(RIGHT_HAND_JOINTS),
                ee_site_name="link_6",
                cameras={},
            )
        )
        robot.connect()
        try:
            action = {name: 0.0 for name in robot.action_features}
            for name in RIGHT_HAND_JOINTS:
                action[f"{name}.delta"] = 0.05
            executed = robot.send_action(action)
            self.assertAlmostEqual(executed["r_f_joint2_2.pos"], 0.05, places=6)
            self.assertTrue(np.all(np.isfinite(robot._sim.data.qpos)))
            self.assertTrue(np.all(np.isfinite(robot._sim.data.qvel)))
            self.assertGreater(robot._sim.data.time, 0.0)
            actuator_id = robot._sim.get_joint_actuator_id("r_f_joint2_2")
            self.assertAlmostEqual(robot._sim.data.ctrl[actuator_id], 0.05, places=6)
        finally:
            robot.disconnect()


if __name__ == "__main__":
    unittest.main()
