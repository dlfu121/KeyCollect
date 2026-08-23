"""MuJoCo Robot Plugin for LeRobot 0.6.1."""

from .config_mujoco import MuJoCoRobotConfig
from .mujoco import MuJoCoRobot

__all__ = ["MuJoCoRobot", "MuJoCoRobotConfig"]
