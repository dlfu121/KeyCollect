#!/usr/bin/env python
"""Keyboard + mouse teleoperation for the demo MuJoCo scene."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBOT_PKG = ROOT / "lerobot_robot_mujoco"
TELEOP_PKG = ROOT / "lerobot_teleoperator_keyboard_mouse"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROBOT_PKG) not in sys.path:
    sys.path.insert(0, str(ROBOT_PKG))
if str(TELEOP_PKG) not in sys.path:
    sys.path.insert(0, str(TELEOP_PKG))

from lerobot_robot_mujoco.simulation import MuJoCoSimulation
from lerobot_teleoperator_keyboard_mouse import KeyboardMouseTeleop, KeyboardMouseTeleopConfig


def has_motion(action: dict[str, float]) -> bool:
    keys = (
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_roll",
        "delta_pitch",
        "delta_yaw",
        "gripper_delta",
    )
    return any(abs(float(action.get(key, 0.0))) > 1e-9 for key in keys)


def action_to_joint_targets(
    sim: MuJoCoSimulation,
    action: dict[str, float],
    arm_joints: list[str],
    gripper_joints: list[str],
    current_gripper: float,
    ee_body: str,
    damping: float = 0.08,
    max_joint_step: float = 0.04,
) -> tuple[np.ndarray, float]:
    delta = np.array(
        [
            action.get("delta_x", 0.0),
            action.get("delta_y", 0.0),
            action.get("delta_z", 0.0),
            0.0,
            action.get("delta_pitch", 0.0),
            action.get("delta_yaw", 0.0),
        ],
        dtype=np.float64,
    )

    body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
    if body_id < 0:
        raise ValueError(f"EE body '{ee_body}' not found.")

    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))
    mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, body_id)
    full_jac = np.vstack([jacp, jacr])
    cols = []
    for name in arm_joints:
        joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        cols.append(int(sim.model.jnt_dofadr[joint_id]))
    jac = full_jac[:, cols]
    active_rows = np.abs(delta) > 1e-12
    current_q = sim.get_joint_positions(arm_joints)

    if np.any(active_rows):
        active_jac = jac[active_rows, :]
        active_delta = delta[active_rows]
        n_dof = active_jac.shape[1]
        lhs = active_jac.T @ active_jac + (damping ** 2) * np.eye(n_dof)
        dq = np.linalg.solve(lhs, active_jac.T @ active_delta)
        dq = np.clip(dq, -max_joint_step, max_joint_step)
        current_q = sim.clip_to_joint_limits(arm_joints, current_q + dq)

    wrist_roll = action.get("delta_roll", 0.0)
    if abs(wrist_roll) > 1e-12 and arm_joints:
        current_q[-1] += wrist_roll
        current_q = sim.clip_to_joint_limits(arm_joints, current_q)

    current_gripper = float(
        np.clip(current_gripper + action.get("gripper_delta", 0.0), 0.0, 0.04)
    )
    return current_q, current_gripper


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
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                name,
                "",
                self.context,
            )

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
    parser.add_argument("scene", nargs="?", default=str(ROOT / "assets" / "scene" / "rm65_dexhand_scene.urdf"), help="Path to a MuJoCo XML or URDF scene.")
    parser.add_argument(
        "--control-fps",
        type=int,
        default=30,
        help="Teleop control frequency.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to run before exiting. Use 0 to run until the viewer closes.",
    )
    parser.add_argument("--cameras", nargs="*", default=["camera_front", "camera_side"], help="Camera names to show.")
    parser.add_argument("--ee-body", default="link_6", help="Body used as the end-effector control frame.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sim = MuJoCoSimulation(args.scene)
    sim.load()
    sim.reset()
    sim.launch_viewer()

    arm_joints = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    gripper_joints = [
        "r_f_joint1_1",
        "r_f_joint1_2",
        "r_f_joint2_1",
        "r_f_joint2_2",
        "r_f_joint3_1",
        "r_f_joint3_2",
        "r_f_joint4_1",
        "r_f_joint4_2",
        "r_f_joint5_1",
        "r_f_joint5_2",
    ]

    teleop = KeyboardMouseTeleop(KeyboardMouseTeleopConfig())
    teleop.connect()

    period = 1.0 / max(1, args.control_fps)
    start = time.monotonic()
    camera_panel = CameraPanel(sim, list(args.cameras))
    current_gripper = 0.0
    if camera_panel.enabled:
        print(f"Camera panel running: {', '.join(name for name, _ in camera_panel.cameras)}")
    elif args.cameras:
        print("Camera panel disabled.")

    try:
        print("Teleop running. Hold Space while pressing W/S/A/D/Q/E/Z/X/R/F.")
        while sim.sync_viewer() and teleop.is_connected and (not camera_panel.enabled or camera_panel.is_running()):
            camera_panel.update()
            teleop_action = teleop.get_action()
            if not has_motion(teleop_action):
                sim.forward()
                time.sleep(period)
                continue

            arm_targets, current_gripper = action_to_joint_targets(
                sim,
                teleop_action,
                arm_joints,
                gripper_joints,
                current_gripper,
                args.ee_body,
            )
            targets = np.concatenate(
                [arm_targets, np.full(len(gripper_joints), current_gripper)]
            )
            sim.set_joint_qpos(arm_joints + gripper_joints, targets)
            sim.forward()
            sim.sync_viewer()
            camera_panel.update()

            if args.duration > 0 and time.monotonic() - start >= args.duration:
                break

            time.sleep(period)
    finally:
        camera_panel.close()
        teleop.disconnect()
        sim.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
