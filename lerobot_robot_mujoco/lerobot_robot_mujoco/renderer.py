"""Camera renderer utilities for MuJoCo simulation."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def validate_rgb_image(image: np.ndarray, expected_shape: tuple[int, int, int] | None = None) -> bool:
    """Validate an RGB image array.

    Args:
        image: Image array to validate.
        expected_shape: Optional (H, W, 3) to check against.

    Returns:
        True if valid.
    """
    if image.dtype != np.uint8:
        logger.error("Image dtype is %s, expected uint8", image.dtype)
        return False
    if image.ndim != 3 or image.shape[2] != 3:
        logger.error("Image shape is %s, expected (H, W, 3)", image.shape)
        return False
    if expected_shape is not None and image.shape != expected_shape:
        logger.error("Image shape %s != expected %s", image.shape, expected_shape)
        return False
    return True
