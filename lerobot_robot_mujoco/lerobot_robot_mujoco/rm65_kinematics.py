"""Official RM65-6F kinematics and Cartesian-control helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RM65_DOF = 6
RM65_A = np.array([0.0, 0.0, 0.256, 0.0, 0.0, 0.0], dtype=np.float64)
RM65_ALPHA = np.array(
    [0.0, np.pi / 2.0, 0.0, np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0],
    dtype=np.float64,
)
RM65_D = np.array([0.2405, 0.0, 0.0, 0.210, 0.0, 0.1725], dtype=np.float64)
RM65_OFFSET = np.array(
    [0.0, np.pi / 2.0, np.pi / 2.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
RM65_Q_MIN = np.deg2rad([-178.0, -130.0, -135.0, -178.0, -128.0, -360.0])
RM65_Q_MAX = np.deg2rad([178.0, 130.0, 135.0, 178.0, 128.0, 360.0])
RM65_QD_MAX = np.deg2rad([180.0, 180.0, 225.0, 225.0, 225.0, 225.0])

# Palm-down working pose with q3 and q5 clear of the elbow/wrist singularities.
RM65_HOME_Q = np.deg2rad([0.0, -90.0, 45.0, 0.0, -45.0, 180.0])


def mdh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Craig modified-DH transform: Rx(alpha) Tx(a) Rz(theta) Tz(d)."""

    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Return the official base-to-flange transform for RM65-6F joint angles."""

    joints = np.asarray(q, dtype=np.float64)
    if joints.shape != (RM65_DOF,):
        raise ValueError(f"RM65-6F q must have shape ({RM65_DOF},), got {joints.shape}.")
    transform = np.eye(4, dtype=np.float64)
    for a, alpha, d, theta in zip(RM65_A, RM65_ALPHA, RM65_D, joints + RM65_OFFSET):
        transform = transform @ mdh_transform(a, alpha, d, theta)
    return transform


def rotation_vector_to_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    """SO(3) exponential map."""

    vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3) + _skew(vector)
    axis = vector / angle
    skew = _skew(axis)
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def rotation_matrix_to_vector(rotation: np.ndarray) -> np.ndarray:
    """SO(3) logarithm map with a stable small-angle branch."""

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    angle = float(np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)))
    vee = np.array(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
        dtype=np.float64,
    )
    if angle < 1e-7:
        return 0.5 * vee
    if np.pi - angle < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis /= np.linalg.norm(axis)
        return axis * angle
    return vee * (angle / (2.0 * np.sin(angle)))


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
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


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized quaternion in xyzw order."""

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    quaternion_wxyz = np.empty(4, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion_wxyz[:] = [
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ]
    else:
        index = int(np.argmax(np.diag(matrix)))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        scale = 2.0 * np.sqrt(
            max(0.0, 1.0 + matrix[index, index] - matrix[next_index, next_index] - matrix[last_index, last_index])
        )
        vector = np.zeros(3, dtype=np.float64)
        vector[index] = 0.25 * scale
        vector[next_index] = (matrix[next_index, index] + matrix[index, next_index]) / scale
        vector[last_index] = (matrix[last_index, index] + matrix[index, last_index]) / scale
        scalar = (matrix[last_index, next_index] - matrix[next_index, last_index]) / scale
        quaternion_wxyz[:] = [scalar, *vector]
    quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
    return quaternion_xyzw / np.linalg.norm(quaternion_xyzw)


def damped_least_squares(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    """Compute the RM65 joint step using the official task-space DLS form."""

    jac = np.asarray(jacobian, dtype=np.float64)
    task_error = np.asarray(error, dtype=np.float64)
    regularized = jac @ jac.T + float(damping) ** 2 * np.eye(jac.shape[0])
    return jac.T @ np.linalg.solve(regularized, task_error)


def jacobian_condition_number(jacobian: np.ndarray) -> float:
    singular_values = np.linalg.svd(np.asarray(jacobian, dtype=np.float64), compute_uv=False)
    if singular_values[-1] <= np.finfo(np.float64).eps:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


@dataclass
class CartesianPoseTarget:
    """Persistent base-frame position and flange-frame orientation target."""

    position_world: np.ndarray
    rotation_world: np.ndarray

    @classmethod
    def from_pose(cls, position_world: np.ndarray, rotation_world: np.ndarray) -> "CartesianPoseTarget":
        return cls(
            np.asarray(position_world, dtype=np.float64).copy(),
            np.asarray(rotation_world, dtype=np.float64).reshape(3, 3).copy(),
        )

    def integrate(self, translation_world: np.ndarray, rotation_local: np.ndarray) -> None:
        self.position_world += np.asarray(translation_world, dtype=np.float64)
        self.rotation_world = self.rotation_world @ rotation_vector_to_matrix(rotation_local)

    def error(self, current_position_world: np.ndarray, current_rotation_world: np.ndarray) -> np.ndarray:
        position_error = self.position_world - np.asarray(current_position_world, dtype=np.float64)
        rotation_error_world = rotation_matrix_to_vector(
            self.rotation_world @ np.asarray(current_rotation_world, dtype=np.float64).reshape(3, 3).T
        )
        return np.concatenate([position_error, rotation_error_world])


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
