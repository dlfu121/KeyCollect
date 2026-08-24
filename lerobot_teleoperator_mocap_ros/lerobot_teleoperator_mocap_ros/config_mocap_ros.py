"""Configuration for the ROS1 motion-capture glove teleoperator."""

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


RIGHT_HAND_JOINTS = [
    f"r_f_joint{finger}_{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]


@TeleoperatorConfig.register_subclass("mocap_ros")
@dataclass
class MocapRosTeleopConfig(TeleoperatorConfig):
    """ROS topic and retargeting parameters.

    ``transport=auto`` first tries a direct rospy subscription.  If rospy is
    unavailable it falls back to rosbridge, which is useful when LeRobot's
    Python version differs from the ROS1 system Python.
    """

    wrist_topic: str = "/right_wrist_pose"
    joint_topic: str = "/right_joint_poses"
    transport: str = "auto"  # auto, rospy, or rosbridge
    ros_node_name: str = "keycollect_mocap_teleop"
    rosbridge_host: str = "127.0.0.1"
    rosbridge_port: int = 9090

    # Calibrated translation mapping: robot xyz <- mocap zxy.
    position_axis_map: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    # Calibrated wrist basis: mocap X/Y/Z -> RM65 local -Z/-X/+Y.
    # The negative RM65 Z sign matches the observed clockwise flip direction.
    orientation_axis_map: list[float] = field(
        default_factory=lambda: [0.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0]
    )
    position_scale: float = 0.01
    orientation_scale: float = 1.0
    finger_scale: float = 1.0
    # Suppress glove packet jitter without changing the calibrated mapping.
    # alpha=0.45 adds roughly one control frame of smoothing at 30 Hz.
    finger_filter_alpha: float = 0.45
    finger_deadband_rad: float = 0.003
    finger_outlier_threshold: float = 0.40
    pip_dip_coupling: float = 1.0
    finger_spread_coupling: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # Per-control-frame safety/rate limits.
    max_translation_delta_m: float = 0.005
    max_rotation_delta_rad: float = 0.04
    max_finger_delta_rad: float = 0.05
    stale_timeout_s: float = 0.25
    expected_joint_values: int = 57
    hand_joint_names: list[str] = field(default_factory=lambda: list(RIGHT_HAND_JOINTS))
