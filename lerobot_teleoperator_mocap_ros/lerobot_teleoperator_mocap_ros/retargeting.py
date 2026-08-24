"""Pure retargeting math shared by ROS callbacks and tests."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def matrix3(values: Iterable[float], name: str) -> np.ndarray:
    matrix = np.asarray(list(values), dtype=np.float64)
    if matrix.size != 9:
        raise ValueError(f"{name} must contain 9 values, got {matrix.size}.")
    return matrix.reshape(3, 3)


def map_wrist_position(
    current_xyz: Iterable[float],
    initial_xyz: Iterable[float],
    axis_map: Iterable[float],
    scale: float,
) -> np.ndarray:
    """Return robot-frame translation relative to the first mocap frame."""

    current = np.asarray(list(current_xyz), dtype=np.float64)
    initial = np.asarray(list(initial_xyz), dtype=np.float64)
    return matrix3(axis_map, "position_axis_map") @ (current - initial) * float(scale)


def quaternion_multiply(left_xyzw: Iterable[float], right_xyzw: Iterable[float]) -> np.ndarray:
    """Hamilton product for quaternions stored in ROS/scipy xyzw order."""

    x1, y1, z1, w1 = np.asarray(list(left_xyzw), dtype=np.float64)
    x2, y2, z2, w2 = np.asarray(list(right_xyzw), dtype=np.float64)
    return np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quaternion_inverse(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    quaternion = np.asarray(list(quaternion_xyzw), dtype=np.float64)
    norm_sq = float(np.dot(quaternion, quaternion))
    if norm_sq < 1e-16:
        raise ValueError("Cannot invert a zero quaternion.")
    return np.asarray([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]]) / norm_sq


def quaternion_to_rotvec(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    quaternion = np.asarray(list(quaternion_xyzw), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[:3]))
    if vector_norm < 1e-12:
        return 2.0 * quaternion[:3]
    angle = 2.0 * np.arctan2(vector_norm, quaternion[3])
    return quaternion[:3] * (angle / vector_norm)


def rotvec_to_quaternion(rotvec: Iterable[float]) -> np.ndarray:
    vector = np.asarray(list(rotvec), dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        quaternion = np.r_[0.5 * vector, 1.0]
    else:
        quaternion = np.r_[vector * (np.sin(0.5 * angle) / angle), np.cos(0.5 * angle)]
    return quaternion / np.linalg.norm(quaternion)


def quaternion_to_matrix(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    x, y, z, w = np.asarray(list(quaternion_xyzw), dtype=np.float64)
    norm = np.linalg.norm([x, y, z, w])
    if norm < 1e-12:
        raise ValueError("Cannot convert a zero quaternion to a rotation matrix.")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def map_wrist_orientation(
    current_xyzw: Iterable[float],
    initial_xyzw: Iterable[float],
    axis_map: Iterable[float],
) -> np.ndarray:
    """Map the mocap body-frame rotation relative to its first frame."""

    current = np.asarray(list(current_xyzw), dtype=np.float64)
    initial = np.asarray(list(initial_xyzw), dtype=np.float64)
    # The IMU reports the sensor frame in the world frame.  Wrist motion is
    # commanded around the sensor's calibrated local axes, so extract the
    # body-frame increment R_initial^T R_current: q_initial^-1 * q_current.
    # The opposite product is a world-frame increment and rotates the motion
    # axes with the initial wrist yaw, mixing flip and wave controls.
    relative = quaternion_multiply(quaternion_inverse(initial), current)
    transform = matrix3(axis_map, "orientation_axis_map")
    if not np.allclose(transform @ transform.T, np.eye(3), atol=1e-6):
        raise ValueError("orientation_axis_map must be an orthogonal axis map.")
    # Map the local rotation vector directly.  This also supports the
    # sensor's det=-1 axis permutation without treating it as a physical
    # rotation matrix, and avoids matrix-log branch ambiguity near pi.
    return rotvec_to_quaternion(transform @ quaternion_to_rotvec(relative))


def extract_glove_dofs(values: Iterable[float]) -> dict[str, float]:
    """Extract the 11 DexHand DOFs used by the legacy 60-value mapping."""

    data = np.asarray(list(values), dtype=np.float64)
    if data.size < 57:
        raise ValueError(f"Glove joint message needs at least 57 values, got {data.size}.")
    return {
        "thumb_rot": data[1],
        "thumb_mcp": data[5],
        "thumb_dip": data[8],
        "index_spread": 0.0,
        "index_mcp": data[14],
        "index_dip": 0.5 * (data[17] + data[20]),
        "middle_mcp": data[26],
        "middle_dip": 0.5 * (data[29] + data[32]),
        "ring_mcp": data[38],
        "ring_dip": 0.5 * (data[41] + data[44]),
        "pinky_mcp": data[50],
        "pinky_dip": 0.5 * (data[53] + data[56]),
    }


def map_glove_to_hand_offsets(
    current_values: Iterable[float],
    initial_values: Iterable[float],
    *,
    finger_scale: float = 1.0,
    pip_dip_coupling: float = 1.0,
    finger_spread_coupling: Iterable[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Map a 57-value glove frame to the legacy 20-joint DexHand offset vector.

    The glove flexion channels increase from their calibrated neutral values,
    matching the positive closing direction of the DexHand joints.  The
    returned order is finger 1..5, joint 1..4.
    """

    now = extract_glove_dofs(current_values)
    initial = extract_glove_dofs(initial_values)
    delta = {key: (now[key] - initial[key]) * float(finger_scale) for key in now}
    spread = list(finger_spread_coupling)
    if len(spread) != 3:
        raise ValueError("finger_spread_coupling must contain 3 values.")
    coupling = float(pip_dip_coupling)
    return np.asarray(
        [
            delta["thumb_rot"], delta["thumb_mcp"], delta["thumb_dip"], delta["thumb_dip"] * coupling,
            delta["index_spread"], delta["index_mcp"], delta["index_dip"], delta["index_dip"] * coupling,
            0.0, delta["middle_mcp"], delta["middle_dip"], delta["middle_dip"] * coupling,
            delta["index_spread"] * spread[1], delta["ring_mcp"], delta["ring_dip"], delta["ring_dip"] * coupling,
            delta["index_spread"] * spread[2], delta["pinky_mcp"], delta["pinky_dip"], delta["pinky_dip"] * coupling,
        ],
        dtype=np.float64,
    )
