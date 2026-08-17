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
from lerobot_teleoperator_keyboard_mouse.keyboard_mouse import PYNPUT_AVAILABLE

DEFAULT_STATE_PATH = ROOT / "assets" / "scenes" / "current_state.npz"


FINGER_GROUPS = {
    "thumb": ("r_f_joint1_1", "r_f_joint1_2"),
    "index": ("r_f_joint2_1", "r_f_joint2_2"),
    "middle": ("r_f_joint3_1", "r_f_joint3_2"),
    "ring": ("r_f_joint4_1", "r_f_joint4_2"),
    "pinky": ("r_f_joint5_1", "r_f_joint5_2"),
    "ring_pinky": ("r_f_joint4_1", "r_f_joint4_2", "r_f_joint5_1", "r_f_joint5_2"),
}

HAND_PRESETS = {
    "open": {
        "thumb": (0.0, 0.0),
        "index": (0.0, 0.0),
        "middle": (0.0, 0.0),
        "ring": (0.0, 0.0),
        "pinky": (0.0, 0.0),
    },
    "pinch": {
        "thumb": (1.45, 0.65),
        "index": (0.05, 0.9),
        "middle": (0.0, 0.05),
        "ring": (0.0, 0.0),
        "pinky": (0.0, 0.0),
    },
    "tripod": {
        "thumb": (1.45, 0.7),
        "index": (0.05, 0.85),
        "middle": (0.0, 0.85),
        "ring": (0.0, 0.1),
        "pinky": (0.0, 0.05),
    },
    "power": {
        "thumb": (1.25, 0.9),
        "index": (0.2, 1.05),
        "middle": (0.0, 1.1),
        "ring": (0.2, 1.05),
        "pinky": (0.3, 1.0),
    },
    "sphere": {
        "thumb": (1.2, 0.65),
        "index": (0.25, 0.75),
        "middle": (0.0, 0.8),
        "ring": (0.25, 0.75),
        "pinky": (0.35, 0.7),
    },
    "key": {
        "thumb": (1.35, 0.45),
        "index": (0.25, 0.15),
        "middle": (0.0, 0.45),
        "ring": (0.15, 0.5),
        "pinky": (0.2, 0.5),
    },
}


def has_motion(action: dict[str, object]) -> bool:
    keys = (
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_roll",
        "delta_pitch",
        "delta_yaw",
        "gripper_delta",
        "hand_delta",
    )
    if any(abs(float(action.get(key, 0.0))) > 1e-9 for key in keys):
        return True
    if action.get("hand_preset") is not None:
        return True
    finger_deltas = action.get("finger_deltas", {})
    return isinstance(finger_deltas, dict) and any(
        abs(float(delta)) > 1e-9 for delta in finger_deltas.values()
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


class ViewerKeyTeleop:
    """Fallback teleop driven by MuJoCo viewer key presses."""

    def __init__(self, config: KeyboardMouseTeleopConfig):
        self.config = config
        self.is_connected = True
        self._pending: dict[str, object] = {}

    def key_callback(self, key: int) -> None:
        step = self.config.translation_step_m
        rot_step = self.config.rotation_step_rad
        key_actions = {
            glfw.KEY_W: {"delta_x": step},
            glfw.KEY_S: {"delta_x": -step},
            glfw.KEY_A: {"delta_y": step},
            glfw.KEY_D: {"delta_y": -step},
            glfw.KEY_Q: {"delta_z": step},
            glfw.KEY_E: {"delta_z": -step},
            glfw.KEY_Z: {"delta_roll": rot_step},
            glfw.KEY_X: {"delta_roll": -rot_step},
            glfw.KEY_R: {"hand_delta": self.config.gripper_step, "gripper_delta": self.config.gripper_step},
            glfw.KEY_F: {"hand_delta": -self.config.gripper_step, "gripper_delta": -self.config.gripper_step},
            glfw.KEY_C: {"hand_delta": self.config.gripper_step, "gripper_delta": self.config.gripper_step},
            glfw.KEY_V: {"hand_delta": -self.config.gripper_step, "gripper_delta": -self.config.gripper_step},
            glfw.KEY_1: {"hand_preset": "open"},
            glfw.KEY_2: {"hand_preset": "pinch"},
            glfw.KEY_3: {"hand_preset": "tripod"},
            glfw.KEY_4: {"hand_preset": "power"},
            glfw.KEY_5: {"hand_preset": "sphere"},
            glfw.KEY_6: {"hand_preset": "key"},
            glfw.KEY_0: {"hand_preset": "open"},
        }
        self._pending.update(key_actions.get(key, {}))

    def get_action(self) -> dict[str, object]:
        action = {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "delta_roll": 0.0,
            "delta_pitch": 0.0,
            "delta_yaw": 0.0,
            "gripper_delta": 0.0,
            "hand_delta": 0.0,
            "hand_preset": None,
            "finger_deltas": {},
        }
        action.update(self._pending)
        self._pending = {}
        return action

    def disconnect(self) -> None:
        self.is_connected = False


def action_to_joint_targets(
    sim: MuJoCoSimulation,
    action: dict[str, object],
    arm_joints: list[str],
    ee_body: str,
    damping: float = 0.04,
    max_joint_step: float = 0.10,
) -> np.ndarray:
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

    return current_q


def preset_to_hand_targets(
    preset_name: str,
    gripper_joints: list[str],
) -> np.ndarray:
    preset = HAND_PRESETS[preset_name]
    values_by_joint = {}
    for finger, values in preset.items():
        for joint_name, value in zip(FINGER_GROUPS[finger], values):
            values_by_joint[joint_name] = value
    return np.array([values_by_joint.get(joint, 0.0) for joint in gripper_joints], dtype=np.float64)


def apply_hand_action(
    sim: MuJoCoSimulation,
    action: dict[str, object],
    gripper_joints: list[str],
    current_hand: np.ndarray,
    max_hand_step: float = 0.12,
) -> np.ndarray:
    if not gripper_joints:
        return current_hand

    desired = current_hand.copy()
    preset_name = action.get("hand_preset")
    if isinstance(preset_name, str) and preset_name in HAND_PRESETS:
        desired = preset_to_hand_targets(preset_name, gripper_joints)

    open_delta = float(action.get("hand_delta", action.get("gripper_delta", 0.0)))
    if abs(open_delta) > 1e-12:
        for i, joint in enumerate(gripper_joints):
            if joint.endswith("_2"):
                desired[i] -= open_delta

    finger_deltas = action.get("finger_deltas", {})
    if isinstance(finger_deltas, dict):
        joint_to_idx = {name: idx for idx, name in enumerate(gripper_joints)}
        for finger, open_amount in finger_deltas.items():
            for joint in FINGER_GROUPS.get(str(finger), ()):
                idx = joint_to_idx.get(joint)
                if idx is not None:
                    desired[idx] -= float(open_amount)

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
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                name,
                "Move: W/S forward/back  A/D left/right  Q/E up/down\n"
                "Wrist: Z/X roll    Hand: R open  F close    Presets: 1-6",
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
    parser.add_argument(
        "scene",
        nargs="?",
        default=str(ROOT / "assets" / "scenes" / "rm65_dexhand_scene.xml"),
        help="Path to a MuJoCo XML or URDF scene.",
    )
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
    parser.add_argument("--cameras", nargs="*", default=["table_camera", "wrist_overhead_camera"], help="Camera names to show.")
    parser.add_argument("--ee-body", default="link_6", help="Body used as the end-effector control frame.")
    parser.add_argument("--translation-step", type=float, default=0.02, help="End-effector translation step per control tick, in meters.")
    parser.add_argument("--rotation-step", type=float, default=0.08, help="Wrist rotation step per control tick, in radians.")
    parser.add_argument("--hand-step", type=float, default=0.05, help="Hand open/close step per control tick, in radians.")
    parser.add_argument("--max-joint-step", type=float, default=0.10, help="Maximum arm joint change per control tick, in radians.")
    parser.add_argument("--state-out", type=Path, default=DEFAULT_STATE_PATH, help="Path to save the latest qpos/qvel state for tune_camera.py.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sim = MuJoCoSimulation(args.scene)
    sim.load()
    sim.reset()

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

    teleop_config = KeyboardMouseTeleopConfig(
        translation_step_m=args.translation_step,
        rotation_step_rad=args.rotation_step,
        gripper_step=args.hand_step,
    )
    if PYNPUT_AVAILABLE:
        teleop = KeyboardMouseTeleop(teleop_config)
        key_callback = None
    else:
        teleop = ViewerKeyTeleop(teleop_config)
        key_callback = teleop.key_callback
    sim.launch_viewer(key_callback=key_callback)
    if isinstance(teleop, KeyboardMouseTeleop):
        teleop.connect()

    period = 1.0 / max(1, args.control_fps)
    start = time.monotonic()
    camera_panel = CameraPanel(sim, list(args.cameras))
    current_hand = sim.get_joint_positions(gripper_joints)
    if camera_panel.enabled:
        print(f"Camera panel running: {', '.join(name for name, _ in camera_panel.cameras)}")
    elif args.cameras:
        print("Camera panel disabled.")
    if isinstance(teleop, ViewerKeyTeleop):
        print("pynput is not installed. Using focused MuJoCo viewer key presses for teleop.")

    try:
        deadman = "Hold Space while pressing movement keys." if isinstance(teleop, KeyboardMouseTeleop) else "Focus the MuJoCo viewer window; Space is not required."
        print("Teleop running.")
        print(f"  Mode: {deadman}")
        print("  Move EE: W/S forward/back, A/D left/right, Q/E up/down")
        print("  Wrist:   Z/X roll")
        print("  Hand:    R open, F close, 1 open, 2 pinch, 3 tripod, 4 power, 5 sphere, 6 key")
        print("  Fingers: U/J thumb, I/K index, O/L middle, P/; ring+pinky")
        print(
            f"Speed: {args.control_fps} Hz, translation {args.translation_step:.3f} m/tick, "
            f"rotation {args.rotation_step:.3f} rad/tick, hand {args.hand_step:.3f} rad/tick."
        )
        while sim.sync_viewer() and teleop.is_connected and (not camera_panel.enabled or camera_panel.is_running()):
            camera_panel.update()
            teleop_action = teleop.get_action()
            if not has_motion(teleop_action):
                sim.forward()
                time.sleep(period)
                continue

            arm_targets = action_to_joint_targets(
                sim,
                teleop_action,
                arm_joints,
                args.ee_body,
                max_joint_step=args.max_joint_step,
            )
            current_hand = apply_hand_action(
                sim,
                teleop_action,
                gripper_joints,
                current_hand,
                max_hand_step=max(0.12, args.hand_step * 2.0),
            )
            targets = np.concatenate([arm_targets, current_hand])
            sim.set_joint_qpos(arm_joints + gripper_joints, targets)
            sim.forward()
            sim.sync_viewer()
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
