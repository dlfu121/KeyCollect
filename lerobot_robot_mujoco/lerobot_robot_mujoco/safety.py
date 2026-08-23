"""Safety utilities for joint target clipping and validation."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def unwrap_revolute_targets(
    current: np.ndarray,
    target: np.ndarray,
    period: float = 2.0 * np.pi,
) -> np.ndarray:
    """Choose periodic revolute targets nearest to the current angles."""
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if current.shape != target.shape:
        raise ValueError("current and target must have the same shape")
    return target + period * np.round((current - target) / period)


def clip_joint_step(
    current: np.ndarray,
    target: np.ndarray,
    max_step: float,
) -> np.ndarray:
    """Limit per-step joint position change.

    Args:
        current: Current joint positions.
        target: Requested target positions.
        max_step: Maximum allowed change per joint per step.

    Returns:
        Clipped target positions.
    """
    delta = target - current
    clipped_delta = np.clip(delta, -max_step, max_step)
    return current + clipped_delta


def check_nan_inf(data: np.ndarray, name: str = "data") -> bool:
    """Check for NaN or Inf values. Returns True if valid."""
    if np.any(np.isnan(data)):
        logger.error("NaN detected in %s", name)
        return False
    if np.any(np.isinf(data)):
        logger.error("Inf detected in %s", name)
        return False
    return True


def validate_quaternion(quat: np.ndarray) -> bool:
    """Check quaternion is valid (unit length)."""
    norm = np.linalg.norm(quat)
    return abs(norm - 1.0) < 0.01
