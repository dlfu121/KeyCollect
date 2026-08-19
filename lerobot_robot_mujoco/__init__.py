"""Compatibility wrapper for running the editable plugin from the repo root."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent / "lerobot_robot_mujoco")]

from .config_mujoco import MuJoCoRobotConfig
from .mujoco import MuJoCoRobot

__all__ = ["MuJoCoRobot", "MuJoCoRobotConfig"]
