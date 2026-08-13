"""End-effector control processor.

Converts teleoperator EE deltas into joint targets via IK,
then applies safety clipping before sending to the robot.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .ik import IKSolver

logger = logging.getLogger(__name__)


class EEControlProcessor:
    """Pipeline: EE delta → IK → joint target → safety clip → robot action.

    This processor sits between the teleoperator and the robot plugin,
    converting end-effector space commands into joint space commands.
    """

    def __init__(
        self,
        ik_solver: IKSolver,
        arm_joint_names: list[str],
        gripper_joint_names: list[str] | None = None,
        gripper_range: tuple[float, float] = (0.0, 0.04),
    ):
        self.ik = ik_solver
        self.arm_joint_names = arm_joint_names
        self.gripper_joint_names = gripper_joint_names or []
        self.gripper_range = gripper_range

        # State
        self._current_gripper_pos: float = 0.0

    def process(self, teleop_action: dict[str, Any]) -> dict[str, Any]:
        """Convert teleop EE delta to robot joint action.

        Args:
            teleop_action: Dict with keys:
                delta_x, delta_y, delta_z (translation)
                delta_roll, delta_pitch, delta_yaw (rotation)
                gripper_delta (optional)

        Returns:
            Robot action dict with joint_name.pos keys.
        """
        # Extract deltas
        dx = teleop_action.get("delta_x", 0.0)
        dy = teleop_action.get("delta_y", 0.0)
        dz = teleop_action.get("delta_z", 0.0)
        droll = teleop_action.get("delta_roll", 0.0)
        dpitch = teleop_action.get("delta_pitch", 0.0)
        dyaw = teleop_action.get("delta_yaw", 0.0)

        delta_pos = np.array([dx, dy, dz])
        delta_euler = np.array([droll, dpitch, dyaw])

        # Solve IK
        joint_targets = self.ik.solve_from_delta(delta_pos, delta_euler)

        if joint_targets is None:
            logger.warning("IK failed, returning zero action.")
            return {f"{name}.pos": 0.0 for name in self.arm_joint_names}

        # Build action dict
        action = {}
        for name, target in zip(self.arm_joint_names, joint_targets):
            action[f"{name}.pos"] = float(target)

        # Gripper
        if self.gripper_joint_names:
            gripper_delta = teleop_action.get("gripper_delta", 0.0)
            self._current_gripper_pos += gripper_delta
            self._current_gripper_pos = np.clip(
                self._current_gripper_pos,
                self.gripper_range[0],
                self.gripper_range[1],
            )
            for name in self.gripper_joint_names:
                action[f"{name}.pos"] = float(self._current_gripper_pos)

        return action

    def reset(self) -> None:
        """Reset internal state (e.g., at episode boundary)."""
        self._current_gripper_pos = 0.0
