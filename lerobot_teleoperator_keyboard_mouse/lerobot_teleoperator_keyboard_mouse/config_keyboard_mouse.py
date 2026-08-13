"""Configuration for the Keyboard+Mouse Teleoperator."""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("keyboard_mouse")
@dataclass
class KeyboardMouseTeleopConfig(TeleoperatorConfig):
    """Configuration for keyboard + mouse end-effector teleoperation.

    Attributes:
        translation_step_m: Translation step per key press (meters).
        rotation_step_rad: Rotation step per key press (radians).
        gripper_step: Gripper step per scroll event.
        mouse_sensitivity: Mouse sensitivity multiplier.
        use_gripper: Whether to include gripper control.
    """

    translation_step_m: float = 0.005
    rotation_step_rad: float = 0.03
    gripper_step: float = 0.02
    mouse_sensitivity: float = 1.0
    use_gripper: bool = True
