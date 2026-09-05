#!/usr/bin/env python
"""Motion-capture glove teleoperation for the RM65 + DexHand scene."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBOT_PKG = ROOT / "lerobot_robot_mujoco"
TELEOP_PKG = ROOT / "lerobot_teleoperator_mocap_ros"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROBOT_PKG) not in sys.path:
    sys.path.insert(0, str(ROBOT_PKG))
if str(TELEOP_PKG) not in sys.path:
    sys.path.insert(0, str(TELEOP_PKG))

from rm65.rm65_ik import RM65IKContinuitySelector, RM65Kinematics, pose_matrix

from lerobot_robot_mujoco.simulation import MuJoCoSimulation
from lerobot_robot_mujoco.rm65_kinematics import CartesianPoseTarget
from lerobot_teleoperator_mocap_ros import MocapRosTeleop, MocapRosTeleopConfig
from lerobot_teleoperator_mocap_ros.config_mocap_ros import RIGHT_HAND_JOINTS

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = ROOT / "assets" / "scenes" / "current_state.npz"
# RM65-6F theoretical flange reach: a2 + d4 + d6 = 0.6385 m.
RM65_BASE_HEIGHT_M = 0.2405
RM65_FLANGE_LINK_LENGTHS_M = (0.256, 0.210, 0.1725)
RM65_THEORETICAL_FLANGE_RADIUS_M = sum(RM65_FLANGE_LINK_LENGTHS_M)
# The screwdriver handle mesh extends 82.656 mm from its center.  Keep a
# rounded 83 mm envelope plus 5 mm numerical/modeling margin inside the flange
# reach so the complete handle stays inside the theoretical radius.
RM65_HANDLE_ENVELOPE_RADIUS_M = 0.083
RM65_HANDLE_MARGIN_M = 0.005
RM65_TASK_RADIUS_M = (
    RM65_THEORETICAL_FLANGE_RADIUS_M
    - RM65_HANDLE_ENVELOPE_RADIUS_M
    - RM65_HANDLE_MARGIN_M
)
# Keep an additional small margin for Cartesian target commands and clamp
# targets instead of freezing the arm when a lateral mocap delta crosses the
# boundary.
RM65_SAFE_REACH_RADIUS_M = RM65_TASK_RADIUS_M - 0.01
SCREWDRIVER_PALM_FRONT_OFFSETS = (0.04, 0.22)
SCREWDRIVER_PALM_RIGHT_OFFSETS = (-0.20, 0.20)
SCREWDRIVER_Y_AXIS_ANGLE_DEG = (-35.0, 35.0)

# Standalone teleop control calibration. Keep these values local to this entry
# point so changes made for lerobot-record cannot alter teleop.py implicitly
# through MocapRosTeleopConfig defaults or recording YAML files.
TELEOP_POSITION_AXIS_MAP = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
TELEOP_ORIENTATION_AXIS_MAP = [0.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0]
TELEOP_FINGER_FILTER_ALPHA = 0.45
TELEOP_FINGER_DEADBAND_RAD = 0.003
TELEOP_FINGER_OUTLIER_THRESHOLD = 0.40
TELEOP_PIP_DIP_COUPLING = 1.0
TELEOP_FINGER_SPREAD_COUPLING = [1.0, 1.0, 1.0]
TELEOP_MAX_TRANSLATION_DELTA_M = 0.005
TELEOP_MAX_ROTATION_DELTA_RAD = 0.04
TELEOP_MAX_FINGER_DELTA_RAD = 0.05
TELEOP_EXPECTED_JOINT_VALUES = 57


def has_motion(action: dict[str, object]) -> bool:
    keys = (
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_roll",
        "delta_pitch",
        "delta_yaw",
    )
    if any(abs(float(action.get(key, 0.0))) > 1e-9 for key in keys):
        return True
    return any(
        key.endswith(".delta") and abs(float(value)) > 1e-9
        for key, value in action.items()
    )


def has_all_joints(sim: MuJoCoSimulation, joint_names: list[str]) -> bool:
    return all(
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
        for name in joint_names
    )


def save_state(sim: MuJoCoSimulation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        qpos=sim.data.qpos.copy(),
        qvel=sim.data.qvel.copy(),
        time=np.array([sim.data.time], dtype=np.float64),
        scene=str(sim.scene_path),
    )
    print(f"Saved current state: {path}")


def randomize_screwdrivers(
    sim: MuJoCoSimulation,
    rng: np.random.Generator,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Randomize the screwdriver across the palm's left/right front region."""
    palm_site = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "dexhand_palm_center")
    if palm_site < 0:
        raise ValueError("Scene does not contain the 'dexhand_palm_center' site.")
    palm_xy = sim.data.site_xpos[palm_site, :2].copy()
    palm_rotation = sim.data.site_xmat[palm_site].reshape(3, 3)
    front_xy = palm_rotation[:2, 2].copy()
    right_xy = -palm_rotation[:2, 1].copy()
    front_xy /= np.linalg.norm(front_xy)
    right_xy /= np.linalg.norm(right_xy)

    link1_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "link_1")
    if link1_id < 0:
        raise ValueError("Scene does not contain the RM65 'link_1' body.")
    # link_1 is located RM65 d1 above the robot base; subtracting d1 yields
    # the base-frame origin used by the analytical kinematics.
    base_position = sim.data.xpos[link1_id].copy()
    base_position[2] -= RM65_BASE_HEIGHT_M
    body_name = "screwdriver_red"
    handle_height = sim.data.xpos[sim.get_body_id(body_name), 2]

    handle_center = None
    for _ in range(4096):
        candidate = (
            palm_xy
            + rng.uniform(*SCREWDRIVER_PALM_FRONT_OFFSETS) * front_xy
            + rng.uniform(*SCREWDRIVER_PALM_RIGHT_OFFSETS) * right_xy
        )
        inside_workspace = (
            x_bounds[0] <= candidate[0] <= x_bounds[1]
            and y_bounds[0] <= candidate[1] <= y_bounds[1]
        )
        inside_flange_radius = (
            np.linalg.norm(
                np.array([candidate[0], candidate[1], handle_height]) - base_position
            )
            <= RM65_TASK_RADIUS_M
        )
        if inside_workspace and inside_flange_radius:
            handle_center = candidate
            break
    if handle_center is None:
        raise RuntimeError(
            "Could not place the screwdriver across the palm's front region within "
            f"the RM65 task radius ({RM65_TASK_RADIUS_M:.4f} m)."
        )

    centers: dict[str, np.ndarray] = {}
    qpos_addr = sim.get_body_qpos_addr(body_name)
    angle_from_y = np.deg2rad(rng.uniform(*SCREWDRIVER_Y_AXIS_ANGLE_DEG))
    yaw = 0.5 * np.pi + angle_from_y
    sim.data.qpos[qpos_addr + 3 : qpos_addr + 7] = (
        np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)
    )
    sim.forward()
    body_id = sim.get_body_id(body_name)
    rotation = sim.data.xmat[body_id].reshape(3, 3)
    handle_offset = rotation @ np.array([-0.04, 0.0, 0.0])
    root_position = sim.data.xpos[body_id].copy()
    root_position[:2] = handle_center - handle_offset[:2]
    sim.set_body_position(body_name, root_position)
    joint_id = int(sim.model.body_jntadr[body_id])
    dof_addr = int(sim.model.jnt_dofadr[joint_id])
    sim.data.qvel[dof_addr : dof_addr + 6] = 0.0
    centers[body_name] = root_position + handle_offset
    sim.forward()
    return centers


def analytic_action_to_joint_targets(
    sim: MuJoCoSimulation,
    action: dict[str, object],
    pose_target: CartesianPoseTarget,
    arm_joints: list[str],
    selector: RM65IKContinuitySelector,
) -> np.ndarray | None:
    """Solve the absolute Cartesian target with the RM65 closed-form IK."""
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
    # link_1's body origin is the first joint frame translated by RM65 d1.
    # Do not use the base STL geom: its visual mesh has a non-kinematic offset.
    base_rotation = np.eye(3)
    base_position = sim.data.xpos[link1_id] - np.array([0.0, 0.0, 0.2405])
    world_from_base = np.eye(4)
    world_from_base[:3, :3] = base_rotation
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
        # A noisy/mismatched wrist orientation can make an otherwise reachable
        # lateral position fail the full 6D IK. Retry at the current physical
        # wrist orientation so translation remains controllable.
        wrist_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "link_6")
        if wrist_id >= 0:
            current_rotation = sim.data.xmat[wrist_id].reshape(3, 3).copy()
            relaxed_world = pose_matrix(pose_target.position_world, current_rotation)
            try:
                solution = selector.solve(np.linalg.inv(world_from_base) @ relaxed_world, initial_seed=current_q)
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


def update_mapping_markers(sim: MuJoCoSimulation, glove_target: np.ndarray, ee_body: str) -> None:
    """Draw position markers and XYZ axes in the passive viewer."""
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
            (ee_position, [1.0, 0.1, 0.1, 1.0], "link_6 actual"),
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

    # Draw local world-frame axes at each point: X=red, Y=green, Z=blue.
    axis_length = 0.14
    axis_colors = (
        np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.3, 1.0, 1.0], dtype=np.float32),
    )
    for marker_index, origin in enumerate(marker_positions):
        for axis_index, axis in enumerate(np.eye(3)):
            # Build a frame whose local z axis points along the requested axis.
            z_axis = axis.astype(np.float64)
            helper = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
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


def apply_hand_action(
    sim: MuJoCoSimulation,
    action: dict[str, object],
    gripper_joints: list[str],
    current_hand: np.ndarray,
    max_hand_step: float = 0.12,
) -> np.ndarray:
    if not gripper_joints:
        return current_hand

    desired = current_hand + np.asarray(
        [float(action.get(f"{joint}.delta", 0.0)) for joint in gripper_joints],
        dtype=np.float64,
    )

    desired = sim.clip_to_joint_limits(gripper_joints, desired, margin=0.0)
    limited_step = np.clip(desired - current_hand, -max_hand_step, max_hand_step)
    return sim.clip_to_joint_limits(gripper_joints, current_hand + limited_step, margin=0.0)


class CameraPanel:
    def __init__(self, sim: MuJoCoSimulation, camera_names: list[str], width: int = 960, height: int = 480):
        self.sim = sim
        self.camera_names = camera_names
        self.width = width
        self.height = height
        self.window = None
        self.scene = mujoco.MjvScene(sim.model, maxgeom=10000)
        self.context = None
        self.option = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        self.cameras = []

        if not camera_names:
            return

        if not glfw.init():
            print("Camera panel disabled: GLFW initialization failed.")
            return

        try:
            self.window = glfw.create_window(width, height, "camera_panel", None, None)
            if self.window is None:
                print("Camera panel disabled: failed to create GLFW window.")
                return
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)
            self.context = mujoco.MjrContext(sim.model, mujoco.mjtFontScale.mjFONTSCALE_150)

            fallback_poses = {
                "camera_front": ((1.0, 0.0, 0.8), (0.0, 0.0, 0.3)),
                "camera_side": ((0.0, 1.0, 0.8), (0.0, 0.0, 0.3)),
                "camera_top": ((0.4, 0.0, 1.5), (0.4, 0.0, 0.2)),
            }
            for name in camera_names:
                cam_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                camera = mujoco.MjvCamera()
                if cam_id >= 0:
                    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    camera.fixedcamid = cam_id
                else:
                    look_from, look_at = fallback_poses.get(name, fallback_poses["camera_front"])
                    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                    camera.lookat[:] = look_at
                    camera.distance = float(np.linalg.norm(np.array(look_from) - np.array(look_at)))
                    camera.azimuth = 180.0 if name == "camera_front" else -90.0
                    camera.elevation = -25.0 if name != "camera_top" else -90.0
                self.cameras.append((name, camera))
        except Exception as exc:
            print(f"Camera panel disabled: {exc}")
            self.close()

    @property
    def enabled(self) -> bool:
        return self.window is not None and self.context is not None and len(self.cameras) > 0

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
        for idx, (name, camera) in enumerate(self.cameras):
            viewport = mujoco.MjrRect(idx * panel_width, 0, panel_width, fb_height)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene",
        nargs="?",
        default=str(ROOT / "assets" / "scenes" / "rm65_dexhand_scene.xml"),
        help="Path to a MuJoCo XML or URDF scene.",
    )
    parser.add_argument(
        "--control-fps",
        type=int,
        default=24,
        help="Teleop control frequency.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to run before exiting. Use 0 to run until the viewer closes.",
    )
    parser.add_argument("--cameras", nargs="*", default=["table_camera", "wrist_overhead_camera"], help="Camera names to show.")
    parser.add_argument("--ee-body", default="link_6", help="Body used as the end-effector control frame.")
    parser.add_argument("--transport", choices=("auto", "rospy", "rosbridge"), default="auto", help="ROS1 transport.")
    parser.add_argument("--wrist-topic", default="/right_wrist_pose", help="ROS1 wrist PoseStamped topic.")
    parser.add_argument("--joint-topic", default="/right_joint_poses", help="ROS1 glove Float32MultiArray topic.")
    parser.add_argument("--position-scale", type=float, default=0.01, help="Scale from mocap position units to meters.")
    parser.add_argument("--orientation-scale", type=float, default=0.4, help="Wrist orientation gain.")
    parser.add_argument("--finger-scale", type=float, default=1.0, help="Finger retargeting gain.")
    parser.add_argument("--stale-timeout", type=float, default=0.25, help="Seconds before stale mocap input is held.")
    parser.add_argument(
        "--fixed-screwdrivers",
        "--fixed-red-screwdriver",
        dest="fixed_screwdrivers",
        action="store_true",
        help="Keep the authored screwdriver pose instead of randomizing it.",
    )
    parser.add_argument("--screwdriver-x-range", nargs=2, type=float, default=(-0.18, -0.03), metavar=("MIN", "MAX"))
    parser.add_argument("--screwdriver-y-range", nargs=2, type=float, default=(-0.15, 0.15), metavar=("MIN", "MAX"))
    parser.add_argument("--random-seed", type=int, default=None, help="Optional reproducible object-randomization seed.")
    parser.add_argument("--state-out", type=Path, default=DEFAULT_STATE_PATH, help="Path to save the latest qpos/qvel state for tune_camera.py.")
    return parser.parse_args()


def build_teleop_config(args: argparse.Namespace) -> MocapRosTeleopConfig:
    """Build standalone control settings without recording-config defaults."""
    return MocapRosTeleopConfig(
        wrist_topic=args.wrist_topic,
        joint_topic=args.joint_topic,
        transport=args.transport,
        position_axis_map=list(TELEOP_POSITION_AXIS_MAP),
        orientation_axis_map=list(TELEOP_ORIENTATION_AXIS_MAP),
        position_scale=args.position_scale,
        orientation_scale=args.orientation_scale,
        finger_scale=args.finger_scale,
        finger_filter_alpha=TELEOP_FINGER_FILTER_ALPHA,
        finger_deadband_rad=TELEOP_FINGER_DEADBAND_RAD,
        finger_outlier_threshold=TELEOP_FINGER_OUTLIER_THRESHOLD,
        pip_dip_coupling=TELEOP_PIP_DIP_COUPLING,
        finger_spread_coupling=list(TELEOP_FINGER_SPREAD_COUPLING),
        max_translation_delta_m=TELEOP_MAX_TRANSLATION_DELTA_M,
        max_rotation_delta_rad=TELEOP_MAX_ROTATION_DELTA_RAD,
        max_finger_delta_rad=TELEOP_MAX_FINGER_DELTA_RAD,
        stale_timeout_s=args.stale_timeout,
        expected_joint_values=TELEOP_EXPECTED_JOINT_VALUES,
        hand_joint_names=list(RIGHT_HAND_JOINTS),
    )


def main() -> int:
    args = parse_args()
    sim = MuJoCoSimulation(args.scene)
    sim.load()
    sim.reset()
    if not args.fixed_screwdrivers:
        screwdriver_centers = randomize_screwdrivers(
            sim,
            np.random.default_rng(args.random_seed),
            tuple(args.screwdriver_x_range),
            tuple(args.screwdriver_y_range),
        )
        for body_name, handle_center in screwdriver_centers.items():
            print(
                f"Randomized {body_name} handle center: "
                f"xyz=({handle_center[0]:.3f}, {handle_center[1]:.3f}, {handle_center[2]:.3f})"
            )

    arm_joints = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    gripper_joints = list(RIGHT_HAND_JOINTS)

    required_joints = arm_joints + gripper_joints
    if not has_all_joints(sim, required_joints):
        print(
            "Loaded a scene-only model without the teleop robot joints. "
            "Viewer mode only; pass a robot+scene MJCF to enable arm/hand control."
        )
        sim.launch_viewer()
        start = time.monotonic()
        try:
            while sim.sync_viewer():
                if args.duration > 0 and time.monotonic() - start >= args.duration:
                    break
                time.sleep(1.0 / max(1, args.control_fps))
        finally:
            sim.close()
        return 0

    teleop_config = build_teleop_config(args)
    teleop = MocapRosTeleop(teleop_config)
    teleop.connect()
    sim.launch_viewer()

    period = 1.0 / max(1, args.control_fps)
    physics_steps = max(1, round(period / sim.model.opt.timestep))
    start = time.monotonic()
    camera_panel = CameraPanel(sim, list(args.cameras))
    current_hand = sim.get_joint_positions(gripper_joints)
    ee_body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, args.ee_body)
    pose_target = CartesianPoseTarget.from_pose(
        sim.data.xpos[ee_body_id], sim.data.xmat[ee_body_id].reshape(3, 3)
    )
    analytic_selector = RM65IKContinuitySelector(RM65Kinematics("RM65-6F"), sample_period=period)
    if camera_panel.enabled:
        print(f"Camera panel running: {', '.join(name for name, _ in camera_panel.cameras)}")
    elif args.cameras:
        print("Camera panel disabled.")
    try:
        print("Motion-capture glove teleop running; keyboard and mouse control are disabled.")
        print(f"  ROS transport: {teleop.transport}")
        print(f"  Wrist topic:   {args.wrist_topic}")
        print(f"  Finger topic:  {args.joint_topic}")
        print(
            f"  Control:       {args.control_fps} Hz, position scale {args.position_scale:.3f}, "
            f"orientation scale {args.orientation_scale:.3f}, finger scale {args.finger_scale:.3f}"
        )
        while sim.sync_viewer() and teleop.is_connected and (not camera_panel.enabled or camera_panel.is_running()):
            camera_panel.update()
            teleop_action = teleop.get_action()
            arm_targets = analytic_action_to_joint_targets(
                sim, teleop_action, pose_target, arm_joints, analytic_selector
            )
            if arm_targets is None:
                arm_targets = sim.get_joint_positions(arm_joints)
            current_hand = apply_hand_action(
                sim,
                teleop_action,
                gripper_joints,
                current_hand,
                max_hand_step=teleop_config.max_finger_delta_rad,
            )
            targets = np.concatenate([arm_targets, current_hand])
            sim.set_joint_positions(arm_joints + gripper_joints, targets)
            sim.step(physics_steps)
            # Blue marker: mapped glove target; red marker: actual link_6 pose.
            update_mapping_markers(sim, pose_target.position_world, args.ee_body)
            camera_panel.update()

            if args.duration > 0 and time.monotonic() - start >= args.duration:
                break

            time.sleep(period)
    finally:
        save_state(sim, args.state_out)
        camera_panel.close()
        teleop.disconnect()
        sim.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
