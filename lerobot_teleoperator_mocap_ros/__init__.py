"""Compatibility wrapper for running the editable plugin from the repo root."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent / "lerobot_teleoperator_mocap_ros")]

from .config_mocap_ros import MocapRosTeleopConfig
from .mocap_ros import MocapRosTeleop

__all__ = ["MocapRosTeleop", "MocapRosTeleopConfig"]
