#!/usr/bin/env python3
"""LeRobot recording entry point with MuJoCo episode-reset controls.

This keeps LeRobot's dataset/video implementation intact while changing the
interactive controls for this project:

* q / n / Right: finish and save the current episode;
* r / Left: discard and re-record the current episode;
* Esc: save the current episode and stop the complete recording session.

After every normally-returning recording loop (early key press or time limit),
the MuJoCo robot is reset.  ``MuJoCoRobot.reset_simulation`` restores the XML
home keyframe and then applies the configured screwdriver randomization.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lerobot.scripts import lerobot_record
from lerobot.utils.keyboard_input import apply_recording_control, create_key_listener


_upstream_record_loop: Callable[..., Any] = lerobot_record.record_loop


def init_episode_keyboard_listener():
    """Create controls where q completes one episode instead of quitting all."""
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    def on_key(name: str) -> None:
        key = name.lower()
        if key in ("right", "n", "q"):
            apply_recording_control("right", events)
        elif key in ("left", "r"):
            apply_recording_control("left", events)
        elif key == "esc":
            apply_recording_control("esc", events)

    listener = create_key_listener(
        on_key,
        controls_help="q/n/Right=save episode, r/Left=re-record, Esc=save and quit",
    )
    return listener, events


def record_loop_with_mujoco_reset(*args: Any, **kwargs: Any):
    """Run LeRobot's loop and reset MuJoCo after a recorded episode.

    LeRobot calls the same loop a second time without a dataset for its manual
    hardware-reset countdown.  A MuJoCo reset is instantaneous, so that phase
    is skipped here; otherwise live glove deltas would move the freshly reset
    robot during ``dataset.reset_time_s``.
    """
    robot = kwargs.get("robot")
    dataset = kwargs.get("dataset")
    can_reset = callable(getattr(robot, "reset_simulation", None))

    if dataset is None and can_reset:
        return None

    result = _upstream_record_loop(*args, **kwargs)
    if dataset is not None and can_reset:
        robot.reset_simulation()
        print("MuJoCo scene reset; screwdrivers randomized for the next episode.")
    return result


def main() -> None:
    lerobot_record.init_keyboard_listener = init_episode_keyboard_listener
    lerobot_record.record_loop = record_loop_with_mujoco_reset
    lerobot_record.main()


if __name__ == "__main__":
    main()
