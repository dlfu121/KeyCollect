#!/usr/bin/env python3
"""Run LeRobot ACT training with support for single-channel depth cameras."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.act_depth_adapter import install_act_depth_adapter


def ensure_metric_depth_training_args(args: list[str]) -> None:
    """Keep decoded depth in metres so training matches live MuJoCo inference."""
    option = "--dataset.depth_output_unit"
    value = None
    for index, arg in enumerate(args):
        if arg == option:
            if index + 1 >= len(args):
                raise ValueError(f"{option} requires a value")
            value = args[index + 1]
        elif arg.startswith(f"{option}="):
            value = arg.split("=", 1)[1]

    if value is None:
        args.append(f"{option}=m")
    elif value != "m":
        raise ValueError(
            f"{option} must be 'm' for KeyCollect depth training, got {value!r}."
        )


def main() -> None:
    ensure_metric_depth_training_args(sys.argv)
    install_act_depth_adapter()

    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    lerobot_train_main()


if __name__ == "__main__":
    main()
