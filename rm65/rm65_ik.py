"""睿尔曼 RM65 系列六自由度机械臂的封闭解析运动学。

采用厂家给出的改进 D-H 约定：
    T[i-1,i] = Rx(alpha[i-1]) Tx(a[i-1]) Rz(q[i] + offset[i]) Tz(d[i])

内部长度单位为米，角度单位为弧度。逆解仅使用反三角函数和矩阵运算，
不包含雅可比矩阵、随机初值或数值迭代。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# 不同腕部版本的法兰长度。也可以在构造对象时通过 d6 显式覆盖。
MODEL_D6_M = {
    "RM65-B": 0.144,
    "RM65-B-V": 0.144,
    "RM65-6FB": 0.161,
    "RM65-6FB-V": 0.161,
    "RM65-6F": 0.1725,
}

# 厂家公布的机械关节角范围，单位为度。
JOINT_LIMITS_DEG = np.array(
    [[-178, 178], [-130, 130], [-135, 135], [-178, 178], [-128, 128], [-360, 360]],
    dtype=float,
)

# 厂家公布的六个关节最大角速度，单位为度/秒。
MAX_JOINT_SPEED_DEG = np.array([180, 180, 225, 225, 225, 225], dtype=float)


class IKUnreachableError(ValueError):
    """目标位姿不可达，或所有几何逆解均超出关节限位。"""


def _rx(angle: float) -> np.ndarray:
    """绕 X 轴旋转的齐次变换矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def _rz(angle: float) -> np.ndarray:
    """绕 Z 轴旋转的齐次变换矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _tx(distance: float) -> np.ndarray:
    """沿 X 轴平移的齐次变换矩阵。"""
    transform = np.eye(4)
    transform[0, 3] = distance
    return transform


def _tz(distance: float) -> np.ndarray:
    """沿 Z 轴平移的齐次变换矩阵。"""
    transform = np.eye(4)
    transform[2, 3] = distance
    return transform


def rotation_from_rpy(rpy: Sequence[float], degrees: bool = False) -> np.ndarray:
    """由固定轴 XYZ 欧拉角生成旋转矩阵，等价于 Rz(yaw)Ry(pitch)Rx(roll)。"""
    roll, pitch, yaw = np.deg2rad(rpy) if degrees else np.asarray(rpy, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def pose_matrix(position: Sequence[float], rotation: np.ndarray) -> np.ndarray:
    """由位置向量和旋转矩阵组成 4×4 齐次位姿矩阵。"""
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform


def _rotation_angle(rotation: np.ndarray) -> float:
    """返回旋转矩阵对应的最小旋转角，用于回代校验。"""
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _equivalent_angles(angle: float, lower: float, upper: float, tolerance: float) -> list[float]:
    """列出关节限位内与 angle 相差整周的全部等价角。"""
    period = 2.0 * np.pi
    first = int(np.ceil((lower - angle - tolerance) / period))
    last = int(np.floor((upper - angle + tolerance) / period))
    return [float(np.clip(angle + period * k, lower, upper)) for k in range(first, last + 1)]


@dataclass(frozen=True)
class IKSolution:
    """一组经过关节限位与正运动学回代验证的解析逆解。"""

    joints: np.ndarray
    shoulder: str
    elbow: str
    wrist: str
    position_error: float
    orientation_error: float

    @property
    def joints_deg(self) -> np.ndarray:
        """以度为单位返回六个机械关节角。"""
        return np.rad2deg(self.joints)

    @property
    def converged(self) -> bool:
        """为兼容旧接口保留；解析解经过回代验证后恒为真。"""
        return True

    @property
    def iterations(self) -> int:
        """解析解没有迭代过程，因此恒为零。"""
        return 0

    @property
    def branch(self) -> tuple[str, str, str]:
        """返回肩部、肘部、腕部三个分支标签。"""
        return self.shoulder, self.elbow, self.wrist


@dataclass(frozen=True)
class BranchSelectionWeights:
    """连续轨迹解析分支选择器的各项无量纲权重。"""

    joint_motion: float = 1.0
    velocity_change: float = 0.35
    speed_excess: float = 1000.0
    shoulder_switch: float = 60.0
    elbow_switch: float = 45.0
    wrist_switch: float = 30.0
    joint_limit: float = 4.0
    wrist_singularity: float = 2.0


class RM65Kinematics:
    """RM65 系列正运动学及肩部×肘部×腕部封闭解析逆解。"""

    def __init__(self, model: str = "RM65-B", d6: float | None = None):
        if d6 is None and model not in MODEL_D6_M:
            raise ValueError(f"未知型号 {model!r}；请显式传入 d6（米）")
        self.model = model
        self.a = np.array([0, 0, 0.256, 0, 0, 0.0])
        self.alpha = np.deg2rad([0, 90, 0, 90, -90, 90])
        selected_d6 = MODEL_D6_M[model] if d6 is None else float(d6)
        self.d = np.array([0.2405, 0, 0, 0.210, 0, selected_d6])
        self.offset = np.deg2rad([0, 90, 90, 0, 0, 0])
        self.limits = np.deg2rad(JOINT_LIMITS_DEG)

    def _transform_to_joint(self, joints: np.ndarray, count: int) -> np.ndarray:
        """计算基坐标系到指定关节坐标系的变换。"""
        transform = np.eye(4)
        for index in range(count):
            transform = (
                transform @ _rx(self.alpha[index]) @ _tx(self.a[index])
                @ _rz(joints[index] + self.offset[index]) @ _tz(self.d[index])
            )
        return transform

    def forward(self, joints: Sequence[float], degrees: bool = False) -> np.ndarray:
        """计算法兰相对于基坐标系的齐次变换矩阵。"""
        values = np.deg2rad(joints) if degrees else np.asarray(joints, dtype=float)
        if values.shape != (6,):
            raise ValueError("joints 必须包含 6 个角度")
        return self._transform_to_joint(values, 6)

    @staticmethod
    def _validate_target(target: np.ndarray, tolerance: float) -> None:
        """检查输入是否为合法的刚体齐次变换矩阵。"""
        if target.shape != (4, 4):
            raise ValueError("target 必须是 4×4 齐次变换矩阵")
        rotation = target[:3, :3]
        if not np.allclose(target[3], [0, 0, 0, 1], atol=tolerance):
            raise ValueError("target 最后一行必须为 [0, 0, 0, 1]")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=tolerance):
            raise ValueError("target 的旋转部分不是正交矩阵")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=tolerance):
            raise ValueError("target 的旋转部分行列式必须为 +1")

    def _append_if_valid(self, solutions: list[IKSolution], joints: np.ndarray,
                         target: np.ndarray, branch: tuple[str, str, str],
                         tolerance: float) -> None:
        """对解析候选解进行限位检查、去重及正运动学回代。"""
        if np.any(joints < self.limits[:, 0] - tolerance) or np.any(joints > self.limits[:, 1] + tolerance):
            return
        if any(np.max(np.abs(item.joints - joints)) < tolerance for item in solutions):
            return
        actual = self.forward(joints)
        position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        orientation_error = _rotation_angle(target[:3, :3] @ actual[:3, :3].T)
        if position_error <= tolerance and orientation_error <= tolerance:
            solutions.append(IKSolution(joints.copy(), *branch, position_error, orientation_error))

    def inverse_all(self, target: np.ndarray, *, seed: Sequence[float] | None = None,
                    seed_degrees: bool = False, tolerance: float = 1e-8) -> list[IKSolution]:
        """计算关节限位内的全部封闭解析逆解。

        常规位姿按两个肩部分支、两个肘部分支和两个腕部分支展开。
        腕部奇异时 q4 与 q6 不再分别唯一，此处固定 seed 给出的 q4，
        再由姿态方程直接计算 q6；未提供 seed 时取 q4=0。
        """
        target = np.asarray(target, dtype=float)
        self._validate_target(target, max(tolerance, 1e-7))
        reference = np.zeros(6) if seed is None else np.asarray(seed, dtype=float)
        if reference.shape != (6,):
            raise ValueError("seed 必须包含 6 个角度")
        if seed_degrees:
            reference = np.deg2rad(reference)

        # 末端沿 z6 回退 d6，得到三个腕轴的交点（腕心）。
        wrist_center = target[:3, 3] - self.d[5] * target[:3, 2]
        x, y, z = wrist_center
        radial_distance = float(np.hypot(x, y))
        vertical_distance = float(z - self.d[0])
        upper_arm, forearm = self.a[2], self.d[3]

        # 肩部两支对应平面二连杆径向坐标 A 的正、负号。
        shoulder_candidates: list[tuple[float, float, str]] = []
        if radial_distance <= tolerance:
            # 肩部奇异：q1 任意。选取最接近当前状态的一支作为代表。
            shoulder_candidates.append((float(reference[0]), 0.0, "肩部奇异"))
        else:
            base = float(np.arctan2(-y, -x))
            shoulder_candidates.extend([
                (base, radial_distance, "肩部正支"),
                (base + np.pi, -radial_distance, "肩部反支"),
            ])

        solutions: list[IKSolution] = []
        for q1_raw, planar_radius, shoulder_name in shoulder_candidates:
            cosine_q3 = (planar_radius**2 + vertical_distance**2 - upper_arm**2 - forearm**2) / (
                2.0 * upper_arm * forearm
            )
            if cosine_q3 < -1.0 - tolerance or cosine_q3 > 1.0 + tolerance:
                continue
            cosine_q3 = float(np.clip(cosine_q3, -1.0, 1.0))
            elbow_angle = float(np.arccos(cosine_q3))

            for q3, elbow_name in ((elbow_angle, "肘下"), (-elbow_angle, "肘上")):
                q2 = float(np.arctan2(planar_radius, vertical_distance) - np.arctan2(
                    forearm * np.sin(q3), upper_arm + forearm * np.cos(q3)
                ))
                q1_values = _equivalent_angles(q1_raw, *self.limits[0], tolerance)
                q2_values = _equivalent_angles(q2, *self.limits[1], tolerance)
                q3_values = _equivalent_angles(q3, *self.limits[2], tolerance)

                for q1_value in q1_values:
                    for q2_value in q2_values:
                        for q3_value in q3_values:
                            first_three = np.array([q1_value, q2_value, q3_value, 0.0, 0.0, 0.0])
                            rotation_03 = self._transform_to_joint(first_three, 3)[:3, :3]
                            rotation_36 = rotation_03.T @ target[:3, :3]

                            # 由 R36 的元素直接分离 q4、q5、q6。
                            cosine_q5 = float(np.clip(-rotation_36[1, 2], -1.0, 1.0))
                            sine_q5_abs = float(np.hypot(rotation_36[1, 0], rotation_36[1, 1]))
                            if sine_q5_abs <= tolerance:
                                # q5 限位排除了 ±π，因此这里只需处理 q5=0。
                                q5 = 0.0
                                q4 = float(reference[3])
                                combined = float(np.arctan2(rotation_36[2, 0], rotation_36[0, 0]))
                                wrist_candidates = [(q4, q5, combined - q4, "腕部奇异代表解")]
                            else:
                                q5_positive = float(np.arctan2(sine_q5_abs, cosine_q5))
                                wrist_candidates = []
                                for sign, wrist_name in ((1.0, "腕部不翻转"), (-1.0, "腕部翻转")):
                                    q5 = sign * q5_positive
                                    q4 = float(np.arctan2(sign * rotation_36[2, 2], sign * rotation_36[0, 2]))
                                    q6 = float(np.arctan2(-sign * rotation_36[1, 1], sign * rotation_36[1, 0]))
                                    wrist_candidates.append((q4, q5, q6, wrist_name))

                            for q4, q5, q6, wrist_name in wrist_candidates:
                                q4_values = _equivalent_angles(q4, *self.limits[3], tolerance)
                                q5_values = _equivalent_angles(q5, *self.limits[4], tolerance)
                                q6_values = _equivalent_angles(q6, *self.limits[5], tolerance)
                                for q4_value in q4_values:
                                    for q5_value in q5_values:
                                        for q6_value in q6_values:
                                            joints = np.array([
                                                q1_value, q2_value, q3_value, q4_value, q5_value, q6_value
                                            ])
                                            self._append_if_valid(
                                                solutions, joints, target,
                                                (shoulder_name, elbow_name, wrist_name),
                                                max(tolerance, 1e-7),
                                            )

        solutions.sort(key=lambda item: float(np.linalg.norm(item.joints - reference)))
        return solutions

    def inverse(self, target: np.ndarray, seed: Sequence[float] | None = None, *,
                seed_degrees: bool = False, tolerance: float = 1e-8) -> IKSolution:
        """返回关节限位内最接近 seed 的解析逆解。"""
        solutions = self.inverse_all(target, seed=seed, seed_degrees=seed_degrees, tolerance=tolerance)
        if not solutions:
            raise IKUnreachableError("目标位姿不可达，或全部解析分支均超出关节限位")
        return solutions[0]


class RM65IKContinuitySelector:
    """带分支滞回、速度约束和奇异惩罚的连续解析逆解选择器。

    该类不改变封闭解析公式，只负责从 ``inverse_all`` 返回的有效解中
    选择最适合连续运动的一支。每条新轨迹应创建一个对象或调用 ``reset``。
    """

    def __init__(
        self,
        robot: RM65Kinematics,
        sample_period: float = 1.0 / 30.0,
        *,
        weights: BranchSelectionWeights | None = None,
        maximum_speed_deg: Sequence[float] = MAX_JOINT_SPEED_DEG,
        joint_limit_soft_zone: float = 0.10,
        singularity_zone_deg: float = 5.0,
    ):
        if sample_period <= 0:
            raise ValueError("sample_period 必须大于零")
        self.robot = robot
        self.sample_period = float(sample_period)
        self.weights = BranchSelectionWeights() if weights is None else weights
        self.maximum_speed = np.deg2rad(np.asarray(maximum_speed_deg, dtype=float))
        if self.maximum_speed.shape != (6,) or np.any(self.maximum_speed <= 0):
            raise ValueError("maximum_speed_deg 必须包含 6 个正数")
        if not 0 < joint_limit_soft_zone < 0.5:
            raise ValueError("joint_limit_soft_zone 必须位于 (0, 0.5) 内")
        self.joint_limit_soft_zone = float(joint_limit_soft_zone)
        self.singularity_zone = np.deg2rad(float(singularity_zone_deg))
        self.previous_solution: IKSolution | None = None
        self.previous_velocity = np.zeros(6)
        self.last_cost = 0.0
        self.last_cost_terms: dict[str, float] = {}

    def reset(
        self,
        joints: Sequence[float] | None = None,
        *,
        degrees: bool = False,
    ) -> None:
        """清除轨迹历史；可选 joints 仅作为下一目标点的初始参考角。"""
        self.previous_solution = None
        self.previous_velocity = np.zeros(6)
        self.last_cost = 0.0
        self.last_cost_terms = {}
        if joints is not None:
            values = np.deg2rad(joints) if degrees else np.asarray(joints, dtype=float)
            if values.shape != (6,):
                raise ValueError("joints 必须包含 6 个关节角")
            # 初始角没有对应目标位姿，暂存到下一次 solve 使用。
            self._initial_seed = values.copy()
        else:
            self._initial_seed = None

    @staticmethod
    def _branch_changed(previous: str, current: str) -> bool:
        """奇异代表标签不视为普通分支切换，避免穿越奇异点时误罚。"""
        return previous != current and "奇异" not in previous and "奇异" not in current

    def _selection_cost(self, candidate: IKSolution) -> tuple[float, dict[str, float]]:
        """计算候选解析解相对于上一帧状态的连续性代价。"""
        assert self.previous_solution is not None
        weights = self.weights
        previous = self.previous_solution
        delta = candidate.joints - previous.joints
        velocity = delta / self.sample_period
        normalized_step = delta / (self.maximum_speed * self.sample_period)

        motion_cost = weights.joint_motion * float(np.dot(normalized_step, normalized_step))
        normalized_velocity_change = (velocity - self.previous_velocity) / self.maximum_speed
        velocity_change_cost = weights.velocity_change * float(
            np.dot(normalized_velocity_change, normalized_velocity_change)
        )

        # 超过厂家最大速度的部分施加强惩罚，但不直接删除候选，以免无解时失去诊断信息。
        speed_ratio = np.abs(velocity) / self.maximum_speed
        speed_excess = np.maximum(speed_ratio - 1.0, 0.0)
        speed_cost = weights.speed_excess * float(np.dot(speed_excess, speed_excess))

        branch_cost = 0.0
        if self._branch_changed(previous.shoulder, candidate.shoulder):
            branch_cost += weights.shoulder_switch
        # 肘部接近伸直时两支合并，此区域不施加肘部分支切换惩罚。
        elbow_near_singular = min(abs(previous.joints[2]), abs(candidate.joints[2])) < self.singularity_zone
        if not elbow_near_singular and self._branch_changed(previous.elbow, candidate.elbow):
            branch_cost += weights.elbow_switch
        # 腕部接近 q5=0 时两支合并，交由关节运动项维持 J4/J6 连续。
        wrist_near_singular = min(abs(previous.joints[4]), abs(candidate.joints[4])) < self.singularity_zone
        if not wrist_near_singular and self._branch_changed(previous.wrist, candidate.wrist):
            branch_cost += weights.wrist_switch

        # 在关节范围两端各留一段软区，越靠近硬限位惩罚越大。
        ranges = self.robot.limits[:, 1] - self.robot.limits[:, 0]
        lower_fraction = (candidate.joints - self.robot.limits[:, 0]) / ranges
        upper_fraction = (self.robot.limits[:, 1] - candidate.joints) / ranges
        margin_fraction = np.minimum(lower_fraction, upper_fraction)
        limit_intrusion = np.maximum(
            (self.joint_limit_soft_zone - margin_fraction) / self.joint_limit_soft_zone,
            0.0,
        )
        limit_cost = weights.joint_limit * float(np.dot(limit_intrusion, limit_intrusion))

        # 腕部靠近奇异点时额外抑制 J4、J6 的大幅互相补偿运动。
        sine_q5 = abs(np.sin(candidate.joints[4]))
        sine_zone = max(np.sin(self.singularity_zone), 1e-9)
        singularity_ratio = max(sine_zone / max(sine_q5, 1e-9) - 1.0, 0.0)
        singularity_ratio = min(singularity_ratio, 100.0)
        wrist_step = normalized_step[[3, 5]]
        singularity_cost = weights.wrist_singularity * singularity_ratio**2 * float(
            np.dot(wrist_step, wrist_step)
        )

        terms = {
            "关节位移": motion_cost,
            "速度变化": velocity_change_cost,
            "速度超限": speed_cost,
            "分支切换": branch_cost,
            "关节限位": limit_cost,
            "腕部奇异": singularity_cost,
        }
        return float(sum(terms.values())), terms

    def solve(
        self,
        target: np.ndarray,
        *,
        initial_seed: Sequence[float] | None = None,
        seed_degrees: bool = False,
        tolerance: float = 1e-8,
    ) -> IKSolution:
        """求解一个路径点并更新内部连续性状态。

        第一帧通过 initial_seed 选择起始支路；后续帧自动使用上一帧结果，
        调用者不需要再手工更新 seed。
        """
        if self.previous_solution is None:
            stored_seed = getattr(self, "_initial_seed", None)
            if initial_seed is None and stored_seed is not None:
                seed = stored_seed
                seed_is_degrees = False
            else:
                seed = np.zeros(6) if initial_seed is None else initial_seed
                seed_is_degrees = seed_degrees
            solutions = self.robot.inverse_all(
                target, seed=seed, seed_degrees=seed_is_degrees, tolerance=tolerance
            )
            if not solutions:
                raise IKUnreachableError("目标位姿不可达，或全部解析分支均超出关节限位")
            selected = solutions[0]
            self.last_cost = 0.0
            self.last_cost_terms = {}
        else:
            solutions = self.robot.inverse_all(
                target, seed=self.previous_solution.joints, tolerance=tolerance
            )
            if not solutions:
                raise IKUnreachableError("目标位姿不可达，或全部解析分支均超出关节限位")
            scored = [(self._selection_cost(solution), solution) for solution in solutions]
            (self.last_cost, self.last_cost_terms), selected = min(scored, key=lambda item: item[0][0])

        if self.previous_solution is None:
            self.previous_velocity = np.zeros(6)
        else:
            self.previous_velocity = (
                selected.joints - self.previous_solution.joints
            ) / self.sample_period
        self.previous_solution = selected
        self._initial_seed = None
        return selected


if __name__ == "__main__":
    robot = RM65Kinematics("RM65-B")
    known_joints = np.array([20, -35, 55, 30, -40, 70.0])
    target_pose = robot.forward(known_joints, degrees=True)
    all_solutions = robot.inverse_all(target_pose, seed=known_joints, seed_degrees=True)
    print(f"有效解析解数量：{len(all_solutions)}")
    for number, solution in enumerate(all_solutions, 1):
        branch = f"{solution.shoulder} / {solution.elbow} / {solution.wrist}"
        print(f"{number}: {np.round(solution.joints_deg, 5)}  {branch}")
