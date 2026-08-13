"""Inverse Kinematics solver using MuJoCo Jacobian + Damped Least Squares."""

from __future__ import annotations

import logging

import numpy as np

from lerobot_robot_mujoco.simulation import MuJoCoSimulation

logger = logging.getLogger(__name__)


class IKSolver:
    """Damped Least Squares IK solver using MuJoCo's Jacobian.

    Computes joint targets from EE delta commands using:
    - MuJoCo forward kinematics for current EE pose
    - MuJoCo Jacobian computation
    - Damped Least Squares (DLS) for pseudo-inverse
    """

    def __init__(
        self,
        sim: MuJoCoSimulation,
        joint_names: list[str],
        ee_site: str = "ee_site",
        damping: float = 0.05,
        max_iterations: int = 50,
        pos_tol: float = 0.001,
        ori_tol: float = 0.01,
        max_joint_step: float = 0.1,
    ):
        self.sim = sim
        self.joint_names = joint_names
        self.ee_site = ee_site
        self.damping = damping
        self.max_iterations = max_iterations
        self.pos_tol = pos_tol
        self.ori_tol = ori_tol
        self.max_joint_step = max_joint_step

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Solve IK for target EE pose.

        Args:
            target_pos: Target position [x, y, z].
            target_quat: Target orientation [qx, qy, qz, qw]. If None, orientation is unconstrained.

        Returns:
            Joint positions array, or None if no convergence.
        """
        import mujoco

        for iteration in range(self.max_iterations):
            # Current EE pose
            ee_pose = self.sim.get_ee_pose(self.ee_site)
            current_pos = ee_pose[:3]
            current_quat = ee_pose[3:]

            # Position error
            pos_err = target_pos - current_pos

            # Orientation error (if specified)
            ori_err = np.zeros(3)
            if target_quat is not None:
                # Compute orientation error via quaternion difference
                err_quat = np.zeros(4)
                mujoco.mju_negQuat(err_quat, current_quat)
                mujoco.mju_mulQuat(err_quat, target_quat, err_quat)
                # Convert to rotation vector (axis-angle)
                angle = 2.0 * np.arccos(np.clip(err_quat[3], -1.0, 1.0))
                if abs(angle) > 1e-6:
                    axis = err_quat[:3] / np.sin(angle / 2.0)
                    ori_err = axis * angle

            # Check convergence
            pos_converged = np.linalg.norm(pos_err) < self.pos_tol
            ori_converged = target_quat is None or np.linalg.norm(ori_err) < self.ori_tol

            if pos_converged and ori_converged:
                logger.debug("IK converged in %d iterations", iteration)
                return self.sim.get_joint_positions(self.joint_names)

            # Build task-space error
            if target_quat is not None:
                error = np.concatenate([pos_err, ori_err])  # (6,)
            else:
                error = pos_err  # (3,)

            # Get Jacobian
            jac = self.sim.get_site_jacobian(self.ee_site, self.joint_names)
            if target_quat is None:
                jac = jac[:3, :]  # Translational only

            # Damped Least Squares
            n_dof = jac.shape[1]
            jtj = jac.T @ jac
            damped = jtj + (self.damping ** 2) * np.eye(n_dof)
            delta_q = np.linalg.solve(damped, jac.T @ error)

            # Clip to max step
            delta_q = np.clip(delta_q, -self.max_joint_step, self.max_joint_step)

            # Apply
            current_q = self.sim.get_joint_positions(self.joint_names)
            target_q = current_q + delta_q

            # Clip to joint limits
            target_q = self.sim.clip_to_joint_limits(self.joint_names, target_q, margin=0.01)

            # Set and update (directly write qpos for IK convergence)
            self.sim.set_joint_qpos(self.joint_names, target_q)
            self.sim.forward()

        logger.warning("IK did not converge after %d iterations", self.max_iterations)
        return None

    def solve_from_delta(
        self,
        delta_pos: np.ndarray,
        delta_euler: np.ndarray = np.zeros(3),
    ) -> np.ndarray | None:
        """Solve IK from EE delta command.

        Args:
            delta_pos: [dx, dy, dz] translation delta.
            delta_euler: [droll, dpitch, dyaw] rotation delta (Euler angles).

        Returns:
            Joint positions array, or None if no convergence.
        """
        import mujoco

        # Current EE pose
        ee_pose = self.sim.get_ee_pose(self.ee_site)
        current_pos = ee_pose[:3]
        current_quat = ee_pose[3:]

        # Target position
        target_pos = current_pos + delta_pos

        # Target orientation: apply delta as incremental rotation
        target_quat = None
        if np.linalg.norm(delta_euler) > 1e-6:
            # Convert Euler delta to quaternion
            # Using axis-angle representation
            angle = np.linalg.norm(delta_euler)
            axis = delta_euler / angle
            delta_quat = np.zeros(4)
            delta_quat[3] = np.cos(angle / 2.0)
            delta_quat[:3] = axis * np.sin(angle / 2.0)

            target_quat = np.zeros(4)
            mujoco.mju_mulQuat(target_quat, delta_quat, current_quat)

        return self.solve(target_pos, target_quat)
