"""Keyboard + Mouse Teleoperator for LeRobot 0.6.1.

Provides end-effector delta control via:
- W/S: delta_x
- A/D: delta_y
- Q/E: delta_z
- Z/X: roll
- 1/2/3/4/5/6/0: hand grasp presets
- R/F or V/C: hand open/close trim
- U/J, I/K, O/L, P/;: finger open/close trim
- Space (hold): motion deadman switch
"""

from __future__ import annotations

import logging
import time
from queue import Queue
from typing import Any

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_keyboard_mouse import KeyboardMouseTeleopConfig

logger = logging.getLogger(__name__)

# Try to import pynput for keyboard/mouse
try:
    from pynput import keyboard, mouse
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    logger.warning("pynput not available. Keyboard/mouse teleop will produce zero deltas.")


class KeyboardMouseTeleop(Teleoperator):
    """Keyboard + Mouse teleoperator for end-effector control.

    Output semantics (EE delta):
        delta_x, delta_y, delta_z (translation)
        delta_roll, delta_pitch, delta_yaw (orientation)
        gripper_delta/hand_delta/finger_deltas/hand_preset (dexterous hand)
    """

    config_class = KeyboardMouseTeleopConfig
    name = "keyboard_mouse"

    def __init__(self, config: KeyboardMouseTeleopConfig):
        super().__init__(config)
        self.config = config

        # State
        self._connected = False
        self._keys_pressed: dict[str, bool] = {}
        self._key_queue: Queue = Queue()
        self._mouse_dx: float = 0.0
        self._mouse_dy: float = 0.0
        self._mouse_scroll: float = 0.0
        self._space_held: bool = False

        # Listeners
        self._kb_listener: Any = None
        self._mouse_listener: Any = None

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
        if self.config.use_gripper:
            features["gripper_delta"] = float
            features["hand_delta"] = float
            features["hand_preset"] = str
            features["finger_deltas"] = dict
        return features

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if not PYNPUT_AVAILABLE:
            logger.warning("pynput not installed. All deltas will be zero.")
            self._connected = True
            return

        self._kb_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
            suppress=True,
        )
        self._mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll,
            suppress=True,
        )
        self._kb_listener.start()
        self._mouse_listener.start()
        self._connected = True
        logger.info("Keyboard+Mouse teleop connected.")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ── Key/Mouse callbacks ─────────────────────────────────────

    def _on_key_press(self, key: Any) -> None:
        try:
            char = key.char.lower()
        except AttributeError:
            # Special key
            if key == keyboard.Key.space:
                self._space_held = True
            elif key == keyboard.Key.esc:
                self.disconnect()
            return
        self._keys_pressed[char] = True

    def _on_key_release(self, key: Any) -> None:
        try:
            char = key.char.lower()
        except AttributeError:
            if key == keyboard.Key.space:
                self._space_held = False
            return
        self._keys_pressed.pop(char, None)

    def _on_mouse_move(self, x: int, y: int) -> None:
        # We use relative movement; accumulate for next get_action call
        pass

    def _on_mouse_click(self, x: int, y: int, button: Any, pressed: bool) -> None:
        pass

    def _on_mouse_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        self._mouse_scroll += dy * self.config.gripper_step

    # ── Action ──────────────────────────────────────────────────

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Compute EE delta from current keyboard/mouse state.

        Returns:
            Dict with delta_x..delta_yaw (+ gripper_delta).
        """
        # Deadman: only produce motion when space is held
        if not self._space_held:
            action = {
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_z": 0.0,
                "delta_roll": 0.0,
                "delta_pitch": 0.0,
                "delta_yaw": 0.0,
            }
            if self.config.use_gripper:
                action["gripper_delta"] = 0.0
                action["hand_delta"] = 0.0
                action["hand_preset"] = None
                action["finger_deltas"] = {}
            return action

        step = self.config.translation_step_m
        rot_step = self.config.rotation_step_rad

        # Translation: W/S → +X/-X, A/D → +Y/-Y, Q/E → +Z/-Z
        dx = 0.0
        dy = 0.0
        dz = 0.0
        if self._keys_pressed.get("w", False):
            dx = step
        elif self._keys_pressed.get("s", False):
            dx = -step
        if self._keys_pressed.get("a", False):
            dy = step
        elif self._keys_pressed.get("d", False):
            dy = -step
        if self._keys_pressed.get("q", False):
            dz = step
        elif self._keys_pressed.get("e", False):
            dz = -step

        # Rotation: Z/X → roll, mouse right-drag → pitch/yaw
        droll = 0.0
        dpitch = 0.0
        dyaw = 0.0
        if self._keys_pressed.get("z", False):
            droll = rot_step
        elif self._keys_pressed.get("x", False):
            droll = -rot_step

        # Mouse-based rotation (from accumulated scroll/movement)
        # For simplicity, we use keyboard-only in this version
        # Mouse right-drag can be added via pynput mouse controller

        action = {
            "delta_x": dx,
            "delta_y": dy,
            "delta_z": dz,
            "delta_roll": droll,
            "delta_pitch": dpitch,
            "delta_yaw": dyaw,
        }

        if self.config.use_gripper:
            hand_delta = 0.0
            if self._keys_pressed.get("r", False):
                hand_delta = self.config.gripper_step
            elif self._keys_pressed.get("f", False):
                hand_delta = -self.config.gripper_step
            elif self._keys_pressed.get("c", False):
                hand_delta = self.config.gripper_step
            elif self._keys_pressed.get("v", False):
                hand_delta = -self.config.gripper_step

            preset_keys = {
                "1": "open",
                "2": "pinch",
                "3": "tripod",
                "4": "power",
                "5": "sphere",
                "6": "key",
                "0": "open",
                "t": "pinch",
                "g": "power",
            }
            hand_preset = next(
                (preset for key, preset in preset_keys.items() if self._keys_pressed.get(key, False)),
                None,
            )
            finger_deltas = {
                "thumb": self._axis_delta("u", "j", self.config.gripper_step),
                "index": self._axis_delta("i", "k", self.config.gripper_step),
                "middle": self._axis_delta("o", "l", self.config.gripper_step),
                "ring_pinky": self._axis_delta("p", ";", self.config.gripper_step),
            }
            finger_deltas = {
                name: delta for name, delta in finger_deltas.items() if abs(delta) > 1e-12
            }

            action["gripper_delta"] = hand_delta
            action["hand_delta"] = hand_delta
            action["hand_preset"] = hand_preset
            action["finger_deltas"] = finger_deltas

        return action

    def _axis_delta(self, positive_key: str, negative_key: str, step: float) -> float:
        if self._keys_pressed.get(positive_key, False):
            return step
        if self._keys_pressed.get(negative_key, False):
            return -step
        return 0.0

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._kb_listener is not None:
            self._kb_listener.stop()
            self._kb_listener = None
        if self._mouse_listener is not None:
            self._mouse_listener.stop()
            self._mouse_listener = None
        self._connected = False
        logger.info("Keyboard+Mouse teleop disconnected.")
