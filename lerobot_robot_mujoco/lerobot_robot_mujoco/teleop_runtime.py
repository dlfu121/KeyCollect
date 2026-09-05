"""Shared RM65 mocap control and visualization used by teleop and recording."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import glfw
import mujoco
import numpy as np

# The editable robot plugin lives one directory below the project package that
# owns the analytical RM65 implementation. Match scripts/teleop.py's import
# setup so both entry points execute the same solver source file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rm65.rm65_ik import RM65IKContinuitySelector, pose_matrix

from .rm65_kinematics import CartesianPoseTarget
from .simulation import MuJoCoSimulation

logger = logging.getLogger(__name__)

RM65_SAFE_REACH_RADIUS_M = 0.5785


def analytic_action_to_joint_targets(
    sim: MuJoCoSimulation,
    action: dict[str, Any],
    pose_target: CartesianPoseTarget,
    arm_joints: list[str],
    selector: RM65IKContinuitySelector,
) -> np.ndarray | None:
    """Solve an integrated Cartesian mocap target with RM65 closed-form IK."""
    previous_position = pose_target.position_world.copy()
    previous_rotation = pose_target.rotation_world.copy()
    translation_world = np.asarray(
        [action.get("delta_x", 0.0), action.get("delta_y", 0.0), action.get("delta_z", 0.0)],
        dtype=np.float64,
    )
    rotation_local = np.asarray(
        [action.get("delta_roll", 0.0), action.get("delta_pitch", 0.0), action.get("delta_yaw", 0.0)],
        dtype=np.float64,
    )
    pose_target.integrate(translation_world, rotation_local)

    link1_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "link_1")
    if link1_id < 0:
        return None
    base_position = sim.data.xpos[link1_id] - np.array([0.0, 0.0, 0.2405])
    world_from_base = np.eye(4)
    world_from_base[:3, 3] = base_position
    target_offset = pose_target.position_world - base_position
    target_distance = float(np.linalg.norm(target_offset))
    if target_distance > RM65_SAFE_REACH_RADIUS_M:
        pose_target.position_world[:] = base_position + target_offset * (
            RM65_SAFE_REACH_RADIUS_M / target_distance
        )
        logger.debug(
            "Clamped Cartesian target to RM65 reach radius %.3f m (requested %.3f m).",
            RM65_SAFE_REACH_RADIUS_M,
            target_distance,
        )
    target_world = pose_matrix(pose_target.position_world, pose_target.rotation_world)
    target_base = np.linalg.inv(world_from_base) @ target_world
    current_q = sim.get_joint_positions(arm_joints)
    try:
        solution = selector.solve(target_base, initial_seed=current_q)
    except Exception as exc:
        wrist_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "link_6")
        if wrist_id >= 0:
            current_rotation = sim.data.xmat[wrist_id].reshape(3, 3).copy()
            relaxed_world = pose_matrix(pose_target.position_world, current_rotation)
            try:
                solution = selector.solve(
                    np.linalg.inv(world_from_base) @ relaxed_world,
                    initial_seed=current_q,
                )
                pose_target.rotation_world[:] = current_rotation
                logger.debug("Relaxed unreachable wrist orientation to preserve Cartesian translation: %s", exc)
            except Exception:
                pose_target.position_world[:] = previous_position
                pose_target.rotation_world[:] = previous_rotation
                logger.warning("Analytic RM65 IK failed; holding the current arm target: %s", exc)
                return None
        else:
            pose_target.position_world[:] = previous_position
            pose_target.rotation_world[:] = previous_rotation
            logger.warning("Analytic RM65 IK failed; holding the current arm target: %s", exc)
            return None
    return sim.clip_to_joint_limits(arm_joints, solution.joints)


def update_mapping_markers(
    sim: MuJoCoSimulation,
    glove_target: np.ndarray,
    ee_body: str,
) -> None:
    """Draw the same glove-target and actual-EE markers as scripts/teleop.py."""
    viewer = getattr(sim, "_viewer", None)
    if viewer is None or not viewer.is_running():
        return
    body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
    if body_id < 0:
        return
    ee_position = sim.data.xpos[body_id].copy()
    user_scene = viewer.user_scn
    user_scene.ngeom = 8
    marker_size = np.array([0.035, 0.035, 0.035], dtype=np.float64)
    marker_positions = []
    for index, (position, color, label) in enumerate(
        (
            (np.asarray(glove_target, dtype=np.float64), [0.1, 0.3, 1.0, 1.0], "glove target"),
            (ee_position, [1.0, 0.1, 0.1, 1.0], f"{ee_body} actual"),
        )
    ):
        marker_positions.append(position)
        geom = user_scene.geoms[index]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            marker_size,
            position,
            np.eye(3).reshape(-1),
            np.asarray(color, dtype=np.float32),
        )
        geom.label = f"{label} xyz=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})"

    axis_length = 0.14
    axis_colors = (
        np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.3, 1.0, 1.0], dtype=np.float32),
    )
    for marker_index, origin in enumerate(marker_positions):
        for axis_index, axis in enumerate(np.eye(3)):
            z_axis = axis.astype(np.float64)
            helper = (
                np.array([0.0, 0.0, 1.0])
                if abs(z_axis[2]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            x_axis = np.cross(helper, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            rotation = np.column_stack((x_axis, y_axis, z_axis)).reshape(-1)
            geom = user_scene.geoms[2 + marker_index * 3 + axis_index]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.array([0.008, 0.012, axis_length], dtype=np.float64),
                origin,
                rotation,
                axis_colors[axis_index],
            )
    viewer.sync()


class CameraPanel:
    """Side-by-side fixed-camera window shared by teleop and recording."""

    def __init__(
        self,
        sim: MuJoCoSimulation,
        camera_names: list[str],
        width: int = 960,
        height: int = 480,
    ):
        self.sim = sim
        self.camera_names = camera_names
        self.width = width
        self.height = height
        self.window = None
        self.scene = mujoco.MjvScene(sim.model, maxgeom=10000)
        self.context = None
        self.option = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        self.cameras: list[tuple[str, mujoco.MjvCamera]] = []

        if not camera_names:
            return
        if not glfw.init():
            logger.warning("Camera panel disabled: GLFW initialization failed.")
            return
        try:
            self.window = glfw.create_window(width, height, "camera_panel", None, None)
            if self.window is None:
                logger.warning("Camera panel disabled: failed to create GLFW window.")
                return
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)
            self.context = mujoco.MjrContext(sim.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            for name in camera_names:
                cam_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                if cam_id < 0:
                    logger.warning("Camera panel skipped unknown MuJoCo camera: %s", name)
                    continue
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                camera.fixedcamid = cam_id
                self.cameras.append((name, camera))
        except Exception as exc:
            logger.warning("Camera panel disabled: %s", exc)
            self.close()

    @property
    def enabled(self) -> bool:
        return self.window is not None and self.context is not None and bool(self.cameras)

    def is_running(self) -> bool:
        return self.enabled and not glfw.window_should_close(self.window)

    def update(self) -> None:
        if not self.is_running():
            return
        glfw.make_context_current(self.window)
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        mujoco.mjr_rectangle(
            mujoco.MjrRect(0, 0, fb_width, fb_height),
            0.05,
            0.05,
            0.05,
            1.0,
        )
        panel_width = max(1, fb_width // len(self.cameras))
        for index, (_, camera) in enumerate(self.cameras):
            viewport = mujoco.MjrRect(index * panel_width, 0, panel_width, fb_height)
            mujoco.mjv_updateScene(
                self.sim.model,
                self.sim.data,
                self.option,
                self.perturb,
                camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene,
            )
            mujoco.mjr_render(viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self) -> None:
        if self.context is not None:
            self.context.free()
            self.context = None
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
