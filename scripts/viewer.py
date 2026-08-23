#!/usr/bin/env python
"""Open the demo scene in an interactive MuJoCo viewer."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "lerobot_robot_mujoco"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lerobot_robot_mujoco.simulation import MuJoCoSimulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene",
        nargs="?",
        default=str(ROOT / "assets" / "scene" / "demo_scene.xml"),
        help="Path to a MuJoCo XML scene.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to run before exiting. Use 0 to run until the viewer closes.",
    )
    parser.add_argument(
        "--physics-dt",
        type=float,
        default=0.002,
        help="Expected physics timestep in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sim = MuJoCoSimulation(args.scene, physics_dt=args.physics_dt)

    try:
        sim.load()
        sim.reset()
        sim.launch_viewer()

        start = time.monotonic()
        while sim.sync_viewer():
            sim.step()
            if args.duration > 0 and time.monotonic() - start >= args.duration:
                break
            time.sleep(args.physics_dt)
    finally:
        sim.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
