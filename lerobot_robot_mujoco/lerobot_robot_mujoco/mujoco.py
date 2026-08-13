"""MuJoCo Robot plugin for LeRobot 0.6.1.

Implements the LeRobot Robot interface backed by MuJoCo simulation.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from lerobot.robots.robot import Robot
from lerobot.robots.config import RobotConfig

from .config_mujoco import MuJoCoRobotConfig
from .simulation import MuJoCoSimulation
from .safety import clip_joint_step, check_nan_inf

logger = logging.getLogger(__name__)


class MuJoCoRobot(Robot):
    """MuJoCo-simulated robot implementing the LeRobot Robot interface.

    This plugin connects MuJoCo physics simulation to LeRobot's
    recording pipeline. It provides:
    - Joint position/velocity observations
    - Gripper state observations
    - End-effector pose observations
    - Named camera image observations
    - Joint position control via send_action
    """

    config_class = MuJoCoRobotConfig
    name = "mujoco"

    def __init__(self, config: MuJoCoRobotConfig):
        super().__init__(config)
        self.config = config
        self._sim = MuJoCoSimulation(
            scene_path=config.scene_path,
            physics_dt=config.physics_dt,
        )
        self._connected = False
        self._camera_configs: dict[str, tuple[int, int]] = {}

        # Build joint name lists
        self._arm_joints = list(config.arm_joint_names)
        self._gripper_joints = list(config.gripper_joint_names)
        self._all_joints = self._arm_joints + self._gripper_joints
        self._ee_site = config.ee_site_name

    @property
    def observation_features(self) -> dict:
        features = {}

        # Joint positions
        for name in self._arm_joints:
            features[f"{name}.pos"] = float
        # Joint velocities
        for name in self._arm_joints:
            features[f"{name}.vel"] = float
        # Gripper
        for name in self._gripper_joints:
            features[f"{name}.pos"] = float
        # EE pose: [x, y, z, qx, qy, qz, qw]
        features["ee_pose"] = (7,)

        # Camera images
        for cam_name, (w, h) in self._camera_configs.items():
            features[f"images.{cam_name}"] = (h, w, 3)

        return features

    @property
    def action_features(self) -> dict:
        features = {}
        for name in self._arm_joints:
            features[f"{name}.pos"] = float
        for name in self._gripper_joints:
            features[f"{name}.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True  # Simulation doesn't need calibration

    def connect(self, calibrate: bool = True) -> None:
        """Load scene, initialize simulation."""
        logger.info("Connecting MuJoCo robot...")
        self._sim.load()

        # Validate joints exist
        for name in self._all_joints:
            try:
                self._sim.get_joint_id(name)
            except ValueError as e:
                raise RuntimeError(f"Joint validation failed: {e}") from e

        # Validate EE site
        try:
            self._sim.get_site_id(self._ee_site)
        except ValueError as e:
            raise RuntimeError(f"EE site validation failed: {e}") from e

        # Build camera configs from LeRobot CameraConfig
        for cam_name, cam_config in self.config.cameras.items():
            # Validate camera exists in scene
            self._sim.validate_camera(cam_name)
            self._camera_configs[cam_name] = (cam_config.width, cam_config.height)

        self._sim.reset()
        self._connected = True
        logger.info("MuJoCo robot connected. Joints: %s, Cameras: %s",
                     self._all_joints, list(self._camera_configs.keys()))

    def calibrate(self) -> None:
        pass  # No calibration needed for simulation

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        """Get current robot state + camera images.

        Returns a flat dict matching observation_features.
        """
        if not self._connected:
            raise RuntimeError("Robot not connected.")

        obs = {}

        # Joint positions and velocities
        for name in self._arm_joints:
            obs[f"{name}.pos"] = self._sim.get_joint_position(name)
            obs[f"{name}.vel"] = self._sim.get_joint_velocity(name)

        # Gripper
        for name in self._gripper_joints:
            obs[f"{name}.pos"] = self._sim.get_joint_position(name)

        # EE pose
        ee_pose = self._sim.get_ee_pose(self._ee_site)
        obs["ee_pose"] = ee_pose

        # Camera images
        for cam_name, (w, h) in self._camera_configs.items():
            img = self._sim.render_camera(cam_name, width=w, height=h)
            obs[f"images.{cam_name}"] = img

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send joint position targets to the robot.

        Args:
            action: Dict with keys like "joint1.pos", "joint2.pos", etc.

        Returns:
            The actually executed action (after safety clipping).
        """
        if not self._connected:
            raise RuntimeError("Robot not connected.")

        # Extract targets
        targets = []
        joint_names = []
        for name in self._arm_joints:
            key = f"{name}.pos"
            if key in action:
                targets.append(action[key])
                joint_names.append(name)

        for name in self._gripper_joints:
            key = f"{name}.pos"
            if key in action:
                targets.append(action[key])
                joint_names.append(name)

        if not joint_names:
            return action

        targets = np.array(targets)

        # Get current positions
        current = self._sim.get_joint_positions(joint_names)

        # Safety: clip step size
        targets = clip_joint_step(current, targets, self.config.max_joint_step)

        # Safety: clip to joint limits
        targets = self._sim.clip_to_joint_limits(joint_names, targets, self.config.joint_limit_margin)

        # Validate
        if not check_nan_inf(targets, "joint_targets"):
            logger.warning("NaN/Inf in targets, keeping current positions.")
            targets = current

        # Apply
        self._sim.set_joint_positions(joint_names, targets)

        # Step simulation to reach target
        steps_per_control = int(1.0 / (self.config.control_fps * self.config.physics_dt))
        self._sim.step(max(1, steps_per_control))

        # Build executed action dict
        executed = {}
        for name, target in zip(joint_names, targets):
            executed[f"{name}.pos"] = float(target)

        return executed

    def disconnect(self) -> None:
        """Close simulation."""
        self._sim.close()
        self._connected = False
        logger.info("MuJoCo robot disconnected.")

    # ── Simulation-specific methods ─────────────────────────────

    def reset_simulation(self) -> None:
        """Reset simulation state (e.g., at episode boundary)."""
        self._sim.reset()

    def get_simulation_time(self) -> float:
        """Get current MuJoCo simulation time."""
        return self._sim.time

    @property
    def simulation(self) -> MuJoCoSimulation:
        """Access the underlying simulation."""
        return self._sim
