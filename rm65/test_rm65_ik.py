"""RM65 封闭解析运动学的自动测试。"""

import unittest

import numpy as np

from rm65_ik import (
    IKUnreachableError,
    RM65IKContinuitySelector,
    RM65Kinematics,
    pose_matrix,
)


class TestRM65IK(unittest.TestCase):
    def setUp(self):
        self.robot = RM65Kinematics("RM65-B")

    def assert_pose_equal(self, actual, expected):
        """检查两个齐次位姿在浮点误差范围内相等。"""
        self.assertLess(np.linalg.norm(actual[:3, 3] - expected[:3, 3]), 1e-7)
        relative_rotation = expected[:3, :3] @ actual[:3, :3].T
        angle = np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1))
        self.assertLess(angle, 1e-7)

    def test_典型位姿的全部解析解均能回代(self):
        test_joints = (
            [20, -35, 55, 30, -40, 70],
            [-80, 45, -60, 100, 55, -120],
            [100, -60, 80, -90, 70, 250],
        )
        for joints in test_joints:
            with self.subTest(joints=joints):
                target = self.robot.forward(joints, degrees=True)
                solutions = self.robot.inverse_all(target, seed=joints, seed_degrees=True)
                self.assertGreater(len(solutions), 0)
                for solution in solutions:
                    self.assert_pose_equal(self.robot.forward(solution.joints), target)

    def test_seed选择最近解析支路(self):
        joints = np.array([20, -35, 55, 30, -40, 70.0])
        target = self.robot.forward(joints, degrees=True)
        solution = self.robot.inverse(target, seed=joints, seed_degrees=True)
        np.testing.assert_allclose(solution.joints_deg, joints, atol=1e-6)
        self.assertEqual(solution.iterations, 0)

    def test_腕部奇异使用seed确定代表解(self):
        joints = np.array([15, -20, 40, 50, 0, -30.0])
        target = self.robot.forward(joints, degrees=True)
        solution = self.robot.inverse(target, seed=joints, seed_degrees=True)
        self.assert_pose_equal(self.robot.forward(solution.joints), target)
        self.assertIn("腕部奇异", solution.wrist)
        self.assertAlmostEqual(solution.joints_deg[3], joints[3], places=6)

    def test_不可达目标抛出异常(self):
        target = pose_matrix([2.0, 0, 0], np.eye(3))
        with self.assertRaises(IKUnreachableError):
            self.robot.inverse(target)

    def test_连续选择器保持解析分支并跟踪平滑轨迹(self):
        selector = RM65IKContinuitySelector(self.robot, sample_period=1 / 30)
        previous = None
        branches = []
        for phase in np.linspace(0, 1, 61):
            joints = np.array([
                20 + 5 * np.sin(2 * np.pi * phase),
                -35 + 4 * np.sin(2 * np.pi * phase),
                55 + 3 * np.sin(2 * np.pi * phase),
                30 + 6 * np.sin(2 * np.pi * phase),
                -40 + 3 * np.sin(2 * np.pi * phase),
                70 + 8 * np.sin(2 * np.pi * phase),
            ])
            target = self.robot.forward(joints, degrees=True)
            if previous is None:
                solution = selector.solve(target, initial_seed=joints, seed_degrees=True)
            else:
                solution = selector.solve(target)
                self.assertLess(np.max(np.abs(solution.joints - previous.joints)), np.deg2rad(2))
            branches.append(solution.branch)
            np.testing.assert_allclose(solution.joints_deg, joints, atol=1e-6)
            previous = solution
        self.assertEqual(len(set(branches)), 1)

    def test_连续选择器记录各项选择代价(self):
        selector = RM65IKContinuitySelector(self.robot, sample_period=1 / 30)
        first = self.robot.forward([20, -35, 55, 30, -40, 70], degrees=True)
        second = self.robot.forward([20.5, -34.5, 55.5, 30.5, -39.5, 70.5], degrees=True)
        selector.solve(first, initial_seed=[20, -35, 55, 30, -40, 70], seed_degrees=True)
        selector.solve(second)
        self.assertGreaterEqual(selector.last_cost, 0)
        self.assertEqual(
            set(selector.last_cost_terms),
            {"关节位移", "速度变化", "速度超限", "分支切换", "关节限位", "腕部奇异"},
        )


if __name__ == "__main__":
    unittest.main()
