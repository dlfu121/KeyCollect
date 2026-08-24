"""LeRobot teleoperator consuming the legacy ROS1 mocap glove topics."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_mocap_ros import MocapRosTeleopConfig
from .retargeting import (
    map_glove_to_hand_offsets,
    map_wrist_orientation,
    map_wrist_position,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_rotvec,
    rotvec_to_quaternion,
)

logger = logging.getLogger(__name__)


class MocapRosTeleop(Teleoperator):
    """Convert ROS glove frames into incremental KeyCollect robot actions."""

    config_class = MocapRosTeleopConfig
    name = "mocap_ros"

    def __init__(self, config: MocapRosTeleopConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._lock = threading.Lock()
        self._wrist_initial_pos: np.ndarray | None = None
        self._wrist_initial_quat: np.ndarray | None = None
        self._wrist_pos: np.ndarray | None = None
        self._wrist_quat: np.ndarray | None = None
        self._finger_initial: np.ndarray | None = None
        self._fingers: np.ndarray | None = None
        self._filtered_fingers: np.ndarray | None = None
        self._last_wrist_time = 0.0
        self._last_finger_time = 0.0
        self._last_finger_outlier_warning = 0.0

        self._commanded_pos = np.zeros(3, dtype=np.float64)
        self._commanded_rot = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._commanded_hand = np.zeros(len(config.hand_joint_names), dtype=np.float64)

        self._transport = ""
        self._rospy_subscribers: list[Any] = []
        self._rosbridge_client: Any = None
        self._rosbridge_topics: list[Any] = []

    @property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {
            "delta_x": float,
            "delta_y": float,
            "delta_z": float,
            "delta_roll": float,
            "delta_pitch": float,
            "delta_yaw": float,
        }
        features.update({f"{name}.delta": float for name in self.config.hand_joint_names})
        return features

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return self._wrist_initial_pos is not None and self._finger_initial is not None

    @property
    def transport(self) -> str:
        return self._transport

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        transport = self.config.transport.lower()
        if transport not in {"auto", "rospy", "rosbridge"}:
            raise ValueError("transport must be one of: auto, rospy, rosbridge")

        direct_error: Exception | None = None
        if transport in {"auto", "rospy"}:
            try:
                self._connect_rospy()
                self._transport = "rospy"
            except (ImportError, ModuleNotFoundError) as exc:
                direct_error = exc
                if transport == "rospy":
                    raise RuntimeError(
                        "rospy is unavailable in this Python environment. Source the ROS1 setup before "
                        "activating the venv, or use --teleop.transport=rosbridge."
                    ) from exc

        if not self._transport:
            try:
                self._connect_rosbridge()
                self._transport = "rosbridge"
            except (ImportError, ModuleNotFoundError) as exc:
                message = (
                    "Neither rospy nor roslibpy is available. For direct ROS1 access, source the ROS setup; "
                    "for rosbridge, install the plugin with [rosbridge] and launch rosbridge_websocket."
                )
                if direct_error is not None:
                    message += f" Direct rospy error: {direct_error}"
                raise RuntimeError(message) from exc

        self._connected = True
        logger.info(
            "Mocap ROS teleop connected via %s: wrist=%s joints=%s",
            self._transport,
            self.config.wrist_topic,
            self.config.joint_topic,
        )

    def _connect_rospy(self) -> None:
        import rospy
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Float32MultiArray

        if not rospy.core.is_initialized():
            rospy.init_node(self.config.ros_node_name, anonymous=True, disable_signals=True)
        self._rospy_subscribers = [
            rospy.Subscriber(self.config.wrist_topic, PoseStamped, self._rospy_wrist_callback, queue_size=1),
            rospy.Subscriber(self.config.joint_topic, Float32MultiArray, self._rospy_joint_callback, queue_size=1),
        ]

    def _connect_rosbridge(self) -> None:
        import roslibpy

        client = roslibpy.Ros(host=self.config.rosbridge_host, port=self.config.rosbridge_port)
        client.run(timeout=5)
        if not client.is_connected:
            raise RuntimeError(
                f"Could not connect to rosbridge at {self.config.rosbridge_host}:{self.config.rosbridge_port}."
            )
        wrist = roslibpy.Topic(client, self.config.wrist_topic, "geometry_msgs/PoseStamped", queue_length=1)
        joints = roslibpy.Topic(client, self.config.joint_topic, "std_msgs/Float32MultiArray", queue_length=1)
        wrist.subscribe(self._rosbridge_wrist_callback)
        joints.subscribe(self._rosbridge_joint_callback)
        self._rosbridge_client = client
        self._rosbridge_topics = [wrist, joints]

    def _rospy_wrist_callback(self, msg: Any) -> None:
        self._update_wrist(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w],
        )

    def _rospy_joint_callback(self, msg: Any) -> None:
        self._update_fingers(msg.data)

    def _rosbridge_wrist_callback(self, msg: dict[str, Any]) -> None:
        pose = msg["pose"]
        position, orientation = pose["position"], pose["orientation"]
        self._update_wrist(
            [position["x"], position["y"], position["z"]],
            [orientation["x"], orientation["y"], orientation["z"], orientation["w"]],
        )

    def _rosbridge_joint_callback(self, msg: dict[str, Any]) -> None:
        self._update_fingers(msg["data"])

    def _update_wrist(self, xyz: Any, xyzw: Any) -> None:
        position = np.asarray(xyz, dtype=np.float64)
        quaternion = np.asarray(xyzw, dtype=np.float64)
        if position.shape != (3,) or quaternion.shape != (4,) or not np.all(np.isfinite(position)):
            logger.warning("Ignoring malformed mocap wrist frame.")
            return
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm < 1e-8:
            logger.warning("Ignoring invalid mocap wrist quaternion.")
            return
        quaternion /= norm
        with self._lock:
            if self._wrist_initial_pos is None:
                self._wrist_initial_pos = position.copy()
                self._wrist_initial_quat = quaternion.copy()
            self._wrist_pos = position
            self._wrist_quat = quaternion
            self._last_wrist_time = time.monotonic()

    def _update_fingers(self, values: Any) -> None:
        fingers = np.asarray(values, dtype=np.float64)
        if fingers.ndim != 1 or fingers.size < self.config.expected_joint_values or not np.all(np.isfinite(fingers)):
            logger.warning(
                "Ignoring mocap joint frame: expected at least %d finite values, got %d.",
                self.config.expected_joint_values,
                fingers.size,
            )
            return
        with self._lock:
            if self._finger_initial is None:
                self._finger_initial = fingers.copy()
                self._fingers = fingers.copy()
                self._filtered_fingers = fingers.copy()
            else:
                previous = self._fingers
                accepted = fingers.copy()
                outliers = np.abs(accepted - previous) > self.config.finger_outlier_threshold
                if np.any(outliers):
                    # Reject only the channels that jumped; valid fingers in
                    # the same packet are still accepted.
                    accepted[outliers] = previous[outliers]
                    now = time.monotonic()
                    if now - self._last_finger_outlier_warning >= 1.0:
                        logger.warning(
                            "Rejected %d glove channel jump(s) above %.3f.",
                            int(np.count_nonzero(outliers)),
                            self.config.finger_outlier_threshold,
                        )
                        self._last_finger_outlier_warning = now
                self._fingers = accepted
                alpha = float(np.clip(self.config.finger_filter_alpha, 0.0, 1.0))
                assert self._filtered_fingers is not None
                self._filtered_fingers += alpha * (accepted - self._filtered_fingers)
            self._last_finger_time = time.monotonic()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        now = time.monotonic()
        with self._lock:
            wrist_ready = (
                self._wrist_pos is not None
                and self._wrist_quat is not None
                and self._wrist_initial_pos is not None
                and self._wrist_initial_quat is not None
                and now - self._last_wrist_time <= self.config.stale_timeout_s
            )
            fingers_ready = (
                self._fingers is not None
                and self._finger_initial is not None
                and now - self._last_finger_time <= self.config.stale_timeout_s
            )
            wrist_pos = None if not wrist_ready else self._wrist_pos.copy()
            wrist_quat = None if not wrist_ready else self._wrist_quat.copy()
            wrist_initial_pos = None if not wrist_ready else self._wrist_initial_pos.copy()
            wrist_initial_quat = None if not wrist_ready else self._wrist_initial_quat.copy()
            fingers = None if not fingers_ready else self._filtered_fingers.copy()
            finger_initial = None if not fingers_ready else self._finger_initial.copy()

        translation = np.zeros(3, dtype=np.float64)
        rotation = np.zeros(3, dtype=np.float64)
        if wrist_ready:
            desired_pos = map_wrist_position(
                wrist_pos, wrist_initial_pos, self.config.position_axis_map, self.config.position_scale
            )
            translation = np.clip(
                desired_pos - self._commanded_pos,
                -self.config.max_translation_delta_m,
                self.config.max_translation_delta_m,
            )
            self._commanded_pos += translation

            desired_rot = map_wrist_orientation(
                wrist_quat, wrist_initial_quat, self.config.orientation_axis_map
            )
            desired_rot = rotvec_to_quaternion(
                quaternion_to_rotvec(desired_rot) * self.config.orientation_scale
            )
            rot_error = quaternion_to_rotvec(
                quaternion_multiply(quaternion_inverse(self._commanded_rot), desired_rot)
            )
            rotation = np.clip(
                rot_error,
                -self.config.max_rotation_delta_rad,
                self.config.max_rotation_delta_rad,
            )
            self._commanded_rot = quaternion_multiply(
                self._commanded_rot, rotvec_to_quaternion(rotation)
            )
            self._commanded_rot /= np.linalg.norm(self._commanded_rot)

        hand_delta = np.zeros(len(self.config.hand_joint_names), dtype=np.float64)
        if fingers_ready:
            desired_hand = map_glove_to_hand_offsets(
                fingers,
                finger_initial,
                finger_scale=self.config.finger_scale,
                pip_dip_coupling=self.config.pip_dip_coupling,
                finger_spread_coupling=self.config.finger_spread_coupling,
            )
            if desired_hand.size != self._commanded_hand.size:
                raise ValueError(
                    f"The legacy mapping produces {desired_hand.size} hand joints, but hand_joint_names has "
                    f"{self._commanded_hand.size}."
                )
            hand_error = desired_hand - self._commanded_hand
            hand_error[np.abs(hand_error) < self.config.finger_deadband_rad] = 0.0
            hand_delta = np.clip(
                hand_error,
                -self.config.max_finger_delta_rad,
                self.config.max_finger_delta_rad,
            )
            self._commanded_hand += hand_delta

        action: RobotAction = {
            "delta_x": float(translation[0]),
            "delta_y": float(translation[1]),
            "delta_z": float(translation[2]),
            "delta_roll": float(rotation[0]),
            "delta_pitch": float(rotation[1]),
            "delta_yaw": float(rotation[2]),
        }
        action.update(
            {f"{name}.delta": float(value) for name, value in zip(self.config.hand_joint_names, hand_delta)}
        )
        return action

    def calibrate(self) -> None:
        self.recenter()

    def recenter(self) -> None:
        """Use the latest valid glove frame as the new neutral pose."""

        with self._lock:
            if self._wrist_pos is not None and self._wrist_quat is not None:
                self._wrist_initial_pos = self._wrist_pos.copy()
                self._wrist_initial_quat = self._wrist_quat.copy()
            if self._fingers is not None:
                self._finger_initial = self._fingers.copy()
                self._filtered_fingers = self._fingers.copy()
        self._commanded_pos[:] = 0.0
        self._commanded_rot = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._commanded_hand[:] = 0.0

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        for subscriber in self._rospy_subscribers:
            subscriber.unregister()
        self._rospy_subscribers = []
        for topic in self._rosbridge_topics:
            topic.unsubscribe()
        self._rosbridge_topics = []
        if self._rosbridge_client is not None:
            self._rosbridge_client.terminate()
            self._rosbridge_client = None
        self._connected = False
        logger.info("Mocap ROS teleop disconnected.")
