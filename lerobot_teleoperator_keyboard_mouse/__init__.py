"""Compatibility wrapper for running the editable plugin from the repo root."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent / "lerobot_teleoperator_keyboard_mouse")]

from .config_keyboard_mouse import KeyboardMouseTeleopConfig
from .keyboard_mouse import KeyboardMouseTeleop

__all__ = ["KeyboardMouseTeleop", "KeyboardMouseTeleopConfig"]
