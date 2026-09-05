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
    # Named MuJoCo cameras that also expose metric depth observations. Each
    # entry adds ``<camera_name>_depth`` with shape (H, W, 1), float32 metres.
    depth_camera_names: list[str] = field(default_factory=list)

    # Simulation
    physics_dt: float = 0.001
    control_fps: int = 24
    show_viewer: bool = False
    show_camera_panel: bool = False
    show_mapping_markers: bool = True
    camera_panel_width: int = 960
    camera_panel_height: int = 480

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
    screwdriver_use_arc: bool = False
    # Sample handle centres across both sides of the palm.  At the authored
    # home pose, palm-forward is approximately world +X and palm-right is
    # world -Y; negative right offsets therefore cover the palm's left side.
    screwdriver_workspace_x: tuple[float, float] = (-0.18, -0.03)
    screwdriver_workspace_y: tuple[float, float] = (-0.15, 0.15)
    screwdriver_palm_front_offsets: tuple[float, float] = (0.04, 0.22)
    screwdriver_palm_right_offsets: tuple[float, float] = (-0.20, 0.20)
    # Signed planar angle from world +Y; zero means parallel to the y axis.
    screwdriver_y_axis_angle_deg: tuple[float, float] = (-35.0, 35.0)
    # Annular-sector placement in the table-camera's lower-right quadrant.
    # The sector angle is measured from world +Y; 180 degrees points toward
    # world -Y, with the surrounding sector covering -X/-Y positions.
    screwdriver_arc_center_xy: tuple[float, float] = (-0.27, -0.05)
    screwdriver_arc_radius: tuple[float, float] = (0.08, 0.28)
    screwdriver_arc_angle_deg: tuple[float, float] = (135.0, 225.0)
