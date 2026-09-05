"""MuJoCo Robot plugin for LeRobot 0.6.1.

Implements the LeRobot Robot interface backed by MuJoCo simulation.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

# Must be set before importing mujoco; callers may still override it explicitly.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from lerobot.robots.robot import Robot
from lerobot.robots.config import RobotConfig

from .config_mujoco import MuJoCoRobotConfig
from .rm65_kinematics import CartesianPoseTarget
from .simulation import MuJoCoSimulation
from .safety import clip_joint_step, check_nan_inf
from .teleop_runtime import (
    CameraPanel,
    RM65IKContinuitySelector,
    analytic_action_to_joint_targets,
    update_mapping_markers,
)

from rm65.rm65_ik import RM65Kinematics

logger = logging.getLogger(__name__)

RM65_BASE_HEIGHT_M = 0.2405
RM65_THEORETICAL_FLANGE_RADIUS_M = 0.256 + 0.210 + 0.1725
# The screwdriver handle mesh extends 82.656 mm from its center.  Keep a
# rounded 83 mm envelope plus 5 mm numerical/modeling margin inside the flange
# reach so the complete handle, rather than just its center, stays reachable.
RM65_HANDLE_ENVELOPE_RADIUS_M = 0.083
RM65_HANDLE_MARGIN_M = 0.005
RM65_TASK_RADIUS_M = (
    RM65_THEORETICAL_FLANGE_RADIUS_M
    - RM65_HANDLE_ENVELOPE_RADIUS_M
    - RM65_HANDLE_MARGIN_M
)

def _has_delta_motion(action: dict[str, Any]) -> bool:
    motion_keys = (
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_roll",
        "delta_pitch",
        "delta_yaw",
    )
    if any(abs(float(action.get(key, 0.0))) > 1e-12 for key in motion_keys):
        return True
    if any(
        key.endswith(".delta") and abs(float(value)) > 1e-12
        for key, value in action.items()
    ):
        return True
    return False


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
        self.cameras = dict(config.cameras)
        for camera_name in config.depth_camera_names:
            if camera_name in config.cameras:
                self.cameras[f"{camera_name}_depth"] = config.cameras[camera_name]
        self._camera_configs: dict[str, tuple[int, int]] = {}

        # Build joint name lists
        self._arm_joints = list(config.arm_joint_names)
        self._gripper_joints = list(config.gripper_joint_names)
        self._all_joints = self._arm_joints + self._gripper_joints
        self._ee_site = config.ee_site_name
        self._ee_frame_type = "site"
        self._rng = np.random.default_rng()
        self._screwdriver_body = "screwdriver_red"
        self._cartesian_target: CartesianPoseTarget | None = None
        self._analytic_selector = RM65IKContinuitySelector(
            RM65Kinematics("RM65-6F"),
            sample_period=1.0 / max(1, config.control_fps),
        )
        self._hand_target = np.array([], dtype=np.float64)
        self._camera_panel: CameraPanel | None = None

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
        # EE pose as scalar state features. In LeRobot 0.6.1 tuple-shaped
        # hardware features are interpreted as cameras.
        for pose_name in ("x", "y", "z", "qx", "qy", "qz", "qw"):
            features[f"ee_pose.{pose_name}"] = float

        # Camera images. Use config directly because LeRobot queries features
        # before connect() has populated _camera_configs.
        for cam_name, cam_config in self.config.cameras.items():
            features[cam_name] = (cam_config.height, cam_config.width, 3)
        for cam_name in self.config.depth_camera_names:
            if cam_name not in self.config.cameras:
                raise ValueError(
                    f"Depth camera '{cam_name}' must also be present in cameras."
                )
            cam_config = self.config.cameras[cam_name]
            features[f"{cam_name}_depth"] = (cam_config.height, cam_config.width, 1)

        return features

    @property
    def action_features(self) -> dict:
        features = {
            "delta_x": float,
            "delta_y": float,
            "delta_z": float,
            "delta_roll": float,
            "delta_pitch": float,
            "delta_yaw": float,
        }
        # Per-joint hand deltas make continuous glove motion explicit in the
        # LeRobot dataset instead of hiding it in a non-serializable dict.
        features.update({f"{name}.delta": float for name in self._gripper_joints})
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

        # Validate EE frame. Prefer a MuJoCo site, but allow using a body name
        # for scenes converted from URDF that do not define sites.
        try:
            self._sim.get_site_id(self._ee_site)
        except ValueError as e:
            try:
                self._sim.get_body_id(self._ee_site)
            except ValueError as body_error:
                raise RuntimeError(
                    f"End-effector frame '{self._ee_site}' was not found as a site or body."
                ) from body_error
            self._ee_frame_type = "body"

        # Build camera configs from LeRobot CameraConfig
        for cam_name, cam_config in self.config.cameras.items():
            # Validate camera exists in scene
            self._sim.validate_camera(cam_name)
            self._camera_configs[cam_name] = (cam_config.width, cam_config.height)
        for cam_name in self.config.depth_camera_names:
            if cam_name not in self._camera_configs:
                raise ValueError(
                    f"Depth camera '{cam_name}' must also be present in cameras."
                )

        self._sim.reset()
        self._randomize_screwdrivers_if_enabled()
        if self._all_joints:
            current_positions = self._sim.get_joint_positions(self._all_joints)
            self._sim.set_joint_positions(self._all_joints, current_positions)
            self._sim.forward()
        self._hand_target = (
            self._sim.get_joint_positions(self._gripper_joints)
            if self._gripper_joints
            else np.array([], dtype=np.float64)
        )
        if self.config.show_viewer:
            self._sim.launch_viewer()
        if self.config.show_camera_panel:
            self._camera_panel = CameraPanel(
                self._sim,
                list(self._camera_configs),
                width=self.config.camera_panel_width,
                height=self.config.camera_panel_height,
            )
            if self._camera_panel.enabled:
                logger.info(
                    "Camera panel running: %s",
                    ", ".join(name for name, _ in self._camera_panel.cameras),
                )
                self._camera_panel.update()
            else:
                logger.warning("Camera panel was requested but could not be opened.")
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
        if self._ee_frame_type == "body":
            ee_pose = self._sim.get_body_pose(self._ee_site)
        else:
            ee_pose = self._sim.get_ee_pose(self._ee_site)
        for pose_name, value in zip(("x", "y", "z", "qx", "qy", "qz", "qw"), ee_pose):
            obs[f"ee_pose.{pose_name}"] = float(value)

        # Camera images
        for cam_name, (w, h) in self._camera_configs.items():
            img = self._sim.render_camera(cam_name, width=w, height=h)
            obs[cam_name] = img
        for cam_name in self.config.depth_camera_names:
            w, h = self._camera_configs[cam_name]
            obs[f"{cam_name}_depth"] = self._sim.render_depth_camera(
                cam_name, width=w, height=h
            )

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

        is_delta_action = (
            any(key.startswith("delta_") for key in action)
            or any(key.endswith(".delta") for key in action)
        )
        if is_delta_action:
            action = self._delta_action_to_joint_action(action)
        else:
            # A direct joint command invalidates the previously integrated
            # Cartesian target; initialize it again on the next delta command.
            self._cartesian_target = None

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
        # The mocap delta path has already been rate-limited by MocapRosTeleop
        # and solved with teleop.py's continuity-aware analytical IK. Applying
        # another joint-space clip here would make recording control differ
        # from scripts/teleop.py.
        if not is_delta_action:
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
        if (
            self.config.show_mapping_markers
            and self._cartesian_target is not None
            and self.config.show_viewer
        ):
            update_mapping_markers(
                self._sim,
                self._cartesian_target.position_world,
                self._ee_site,
            )
        if self._camera_panel is not None:
            self._camera_panel.update()

        # Build executed action dict
        executed = {}
        for name, target in zip(joint_names, targets):
            executed[f"{name}.pos"] = float(target)

        return executed

    def _delta_action_to_joint_action(self, action: dict[str, Any]) -> dict[str, float]:
        current_arm = self._sim.get_joint_positions(self._arm_joints)
        if self._ee_frame_type == "body":
            frame_id = self._sim.get_body_id(self._ee_site)
            ee_position_world = self._sim.data.xpos[frame_id].copy()
            ee_rotation_world = self._sim.data.xmat[frame_id].reshape(3, 3).copy()
        else:
            frame_id = self._sim.get_site_id(self._ee_site)
            ee_position_world = self._sim.data.site_xpos[frame_id].copy()
            ee_rotation_world = self._sim.data.site_xmat[frame_id].reshape(3, 3).copy()

        if self._cartesian_target is None:
            self._cartesian_target = CartesianPoseTarget.from_pose(ee_position_world, ee_rotation_world)
        arm_targets = analytic_action_to_joint_targets(
            self._sim,
            action,
            self._cartesian_target,
            self._arm_joints,
            self._analytic_selector,
        )
        if arm_targets is not None:
            current_arm = arm_targets

        self._hand_target = self._apply_hand_action(action, self._hand_target)

        joint_action = {}
        for name, value in zip(self._arm_joints, current_arm):
            joint_action[f"{name}.pos"] = float(value)
        for name, value in zip(self._gripper_joints, self._hand_target):
            joint_action[f"{name}.pos"] = float(value)
        return joint_action

    def _apply_hand_action(self, action: dict[str, Any], current_hand: np.ndarray) -> np.ndarray:
        if not self._gripper_joints:
            return current_hand
        desired = current_hand.copy()
        for index, joint_name in enumerate(self._gripper_joints):
            key = f"{joint_name}.delta"
            if key in action:
                desired[index] += float(action[key])

        desired = self._sim.clip_to_joint_limits(self._gripper_joints, desired, margin=0.0)
        limited_step = np.clip(
            desired - current_hand,
            -self.config.max_finger_step,
            self.config.max_finger_step,
        )
        return self._sim.clip_to_joint_limits(self._gripper_joints, current_hand + limited_step, margin=0.0)

    def disconnect(self) -> None:
        """Close simulation."""
        if self._camera_panel is not None:
            self._camera_panel.close()
            self._camera_panel = None
        self._sim.close()
        self._connected = False
        logger.info("MuJoCo robot disconnected.")

    # ── Simulation-specific methods ─────────────────────────────

    def reset_simulation(self) -> None:
        """Reset simulation state (e.g., at episode boundary)."""
        self._sim.reset()
        self._cartesian_target = None
        self._analytic_selector = RM65IKContinuitySelector(
            RM65Kinematics("RM65-6F"),
            sample_period=1.0 / max(1, self.config.control_fps),
        )
        self._hand_target = (
            self._sim.get_joint_positions(self._gripper_joints)
            if self._gripper_joints
            else np.array([], dtype=np.float64)
        )
        self._randomize_screwdrivers_if_enabled()

    def _randomize_screwdrivers_if_enabled(self) -> None:
        if not getattr(self.config, "randomize_screwdrivers", False):
            return

        x_bounds = tuple(getattr(self.config, "screwdriver_workspace_x", (-0.18, -0.03)))
        y_bounds = tuple(getattr(self.config, "screwdriver_workspace_y", (-0.15, 0.15)))
        front_offsets = tuple(
            getattr(self.config, "screwdriver_palm_front_offsets", (0.04, 0.22))
        )
        right_offsets = tuple(
            getattr(self.config, "screwdriver_palm_right_offsets", (-0.20, 0.20))
        )
        angle_bounds_deg = tuple(
            getattr(self.config, "screwdriver_y_axis_angle_deg", (-35.0, 35.0))
        )
        arc_center = np.asarray(
            getattr(self.config, "screwdriver_arc_center_xy", (-0.27, -0.05)),
            dtype=np.float64,
        )
        arc_radius = tuple(getattr(self.config, "screwdriver_arc_radius", (0.06, 0.16)))
        arc_angle_deg = tuple(
            getattr(self.config, "screwdriver_arc_angle_deg", (-65.0, 65.0))
        )
        use_arc = bool(getattr(self.config, "screwdriver_use_arc", False))
        palm_site = mujoco.mj_name2id(
            self._sim.model, mujoco.mjtObj.mjOBJ_SITE, "dexhand_palm_center"
        )
        if palm_site < 0:
            logger.warning("Could not find DexHand palm site; screwdriver position was not randomized.")
            return
        palm_xy = self._sim.data.site_xpos[palm_site, :2].copy()
        palm_rotation = self._sim.data.site_xmat[palm_site].reshape(3, 3)
        front_xy = palm_rotation[:2, 2].copy()
        right_xy = -palm_rotation[:2, 1].copy()
        front_norm = np.linalg.norm(front_xy)
        right_norm = np.linalg.norm(right_xy)
        if front_norm < 1e-6 or right_norm < 1e-6:
            logger.warning("Could not project the DexHand palm's front/right axes onto the table.")
            return
        front_xy /= front_norm
        right_xy /= right_norm

        link1_id = mujoco.mj_name2id(
            self._sim.model, mujoco.mjtObj.mjOBJ_BODY, "link_1"
        )
        if link1_id < 0:
            logger.warning("Could not find RM65 link_1; screwdriver position was not randomized.")
            return
        base_position = self._sim.data.xpos[link1_id].copy()
        base_position[2] -= RM65_BASE_HEIGHT_M
        try:
            body_id = self._sim.get_body_id(self._screwdriver_body)
        except ValueError:
            logger.warning("Could not find screwdriver body: %s", self._screwdriver_body)
            return
        handle_height = self._sim.data.xpos[body_id, 2]

        # Reject samples outside the table camera's view.  The camera is
        # oblique, so a simple world-X bound is insufficient: world Y also
        # changes the projected horizontal position.  Keep a small margin
        # from each image edge to account for the screwdriver's extent.
        camera_id = mujoco.mj_name2id(
            self._sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "table_camera"
        )
        if camera_id >= 0:
            camera_pos = self._sim.model.cam_pos0[camera_id].copy()
            camera_rotation = self._sim.model.cam_mat0[camera_id].reshape(3, 3)
            vertical_half_fov = np.deg2rad(float(self._sim.model.cam_fovy[camera_id])) / 2.0
            # The configured table camera is rendered at 640x480 (4:3).
            horizontal_half_fov = np.arctan(np.tan(vertical_half_fov) * (4.0 / 3.0))
            view_margin = 0.90

            def inside_table_camera_view(point_xy: np.ndarray) -> bool:
                camera_point = camera_rotation.T @ (
                    np.array([point_xy[0], point_xy[1], handle_height]) - camera_pos
                )
                depth = -camera_point[2]
                if depth <= 0.0:
                    return False
                return (
                    abs(camera_point[0] / depth)
                    <= np.tan(horizontal_half_fov) * view_margin
                    and abs(camera_point[1] / depth)
                    <= np.tan(vertical_half_fov) * view_margin
                )
        else:
            inside_table_camera_view = lambda _point_xy: True

        handle_center: np.ndarray | None = None
        for _ in range(4096):
            if use_arc:
                arc_angle = np.deg2rad(self._rng.uniform(*arc_angle_deg))
                arc_radius_value = self._rng.uniform(*arc_radius)
                candidate = arc_center + arc_radius_value * np.array(
                    [np.sin(arc_angle), np.cos(arc_angle)], dtype=np.float64
                )
            else:
                # Uniformly cover the full configured workspace. The camera
                # frustum check below removes only points outside the view.
                candidate = np.array(
                    [self._rng.uniform(*x_bounds), self._rng.uniform(*y_bounds)],
                    dtype=np.float64,
                )
            inside_workspace = (
                x_bounds[0] <= candidate[0] <= x_bounds[1]
                and y_bounds[0] <= candidate[1] <= y_bounds[1]
            )
            inside_camera_view = inside_table_camera_view(candidate)
            inside_flange_radius = (
                np.linalg.norm(
                    np.array([candidate[0], candidate[1], handle_height]) - base_position
                )
                <= RM65_TASK_RADIUS_M
            )
            if inside_workspace and inside_camera_view and inside_flange_radius:
                handle_center = candidate
                break
        if handle_center is None:
            logger.warning(
                "Could not place the screwdriver within the RM65 task radius %.4f m.",
                RM65_TASK_RADIUS_M,
            )
            return

        body_name = self._screwdriver_body
        qpos_addr = self._sim.get_body_qpos_addr(body_name)
        angle_from_y = np.deg2rad(self._rng.uniform(*angle_bounds_deg))
        yaw = 0.5 * np.pi + angle_from_y
        self._sim.data.qpos[qpos_addr + 3 : qpos_addr + 7] = (
            np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)
        )
        self._sim.forward()
        rotation = self._sim.data.xmat[body_id].reshape(3, 3)
        handle_offset = rotation @ np.array([-0.04, 0.0, 0.0])
        base = self._sim.get_body_pose(body_name)[:3].copy()
        base[:2] = handle_center - handle_offset[:2]
        self._sim.set_body_position(body_name, base)
        self._sim.forward()

    def get_simulation_time(self) -> float:
        """Get current MuJoCo simulation time."""
        return self._sim.time

    @property
    def simulation(self) -> MuJoCoSimulation:
        """Access the underlying simulation."""
        return self._sim
