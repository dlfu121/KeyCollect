"""Configuration for the MuJoCo Robot plugin."""

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("mujoco")
@dataclass
class MuJoCoRobotConfig(RobotConfig):
    """Configuration for MuJoCo simulated robot.

    Attributes:
        scene_path: Path to the MuJoCo scene XML file.
        arm_joint_names: List of arm joint names (in order).
        gripper_joint_names: List of gripper joint names.
        ee_site_name: Name of the end-effector site in the scene.
        cameras: Dict of camera name -> CameraConfig.
        physics_dt: Physics simulation timestep (seconds).
        control_fps: Control loop frequency (Hz).
        max_joint_step: Maximum joint position change per control step (radians).
        max_finger_step: Maximum DexHand target change per control step (radians).
        joint_limit_margin: Safety margin from joint limits (radians).
        ik_damping: Damping factor for DLS inverse kinematics.
        ik_max_iterations: Maximum IK solver iterations.
        ik_pos_tol: Position convergence tolerance (meters).
        ik_ori_tol: Orientation convergence tolerance (radians).
    """

    scene_path: str = ""
    arm_joint_names: list[str] = field(default_factory=list)
    gripper_joint_names: list[str] = field(default_factory=list)
    ee_site_name: str = "ee_site"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Simulation
    physics_dt: float = 0.002
    control_fps: int = 30
    show_viewer: bool = False

    # Safety
    max_joint_step: float = 0.1
    max_finger_step: float = 0.05
    joint_limit_margin: float = 0.01

    # IK
    ik_damping: float = 0.05
    ik_max_iterations: int = 50
    ik_pos_tol: float = 0.005
    ik_ori_tol: float = 0.05

    # Object randomization
    randomize_screwdrivers: bool = True
    # Bounds are the central, front portion of the 1.0 x 0.8 m work table,
    # inside the RM65's comfortable downward-grasp workspace.
    screwdriver_workspace_x: tuple[float, float] = (-0.32, 0.20)
    screwdriver_workspace_y: tuple[float, float] = (-0.28, 0.28)
    screwdriver_min_separation_m: float = 0.12
