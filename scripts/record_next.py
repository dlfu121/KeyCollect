#!/usr/bin/env python3
"""Run the RM65 recording entry point with the first unused run number."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/media/ee304/FDL")
CONFIG_PATH = ROOT / "config" / "record_mujoco.yaml"
RECORD_SCRIPT = ROOT / "scripts" / "record_mujoco.py"
RUN_PREFIXES = {
    "depth": "rm65_dexhand_depth_run_",
    "rgb": "rm65_dexhand_run_",
}


def next_run_index(data_dir: Path = DATA_DIR, profile: str = "depth") -> int:
    """Return the first positive run index without an existing dataset path."""
    run_prefix = RUN_PREFIXES[profile]
    run_pattern = re.compile(rf"^{re.escape(run_prefix)}(\d+)$")
    reservation_pattern = re.compile(rf"^\.{re.escape(run_prefix)}(\d+)\.reserve$")
    occupied: set[int] = set()
    if data_dir.exists():
        for path in data_dir.iterdir():
            match = run_pattern.fullmatch(path.name)
            if match is None:
                match = reservation_pattern.fullmatch(path.name)
            if match is not None:
                occupied.add(int(match.group(1)))
    index = 1
    while index in occupied:
        index += 1
    return index


def build_record_command(
    index: int, extra_args: list[str], profile: str = "depth"
) -> tuple[list[str], Path, str]:
    """Build the recording command using the project's episode-control wrapper.

    ``record_mujoco.py`` monkey-patches LeRobot's recording loop so that q/n/
    Right saves one episode and advances, while r/Left discards and retries it.
    Invoking the installed ``lerobot-record`` executable directly would bypass
    those controls and use LeRobot's default session-level keyboard behavior.
    """
    if not RECORD_SCRIPT.is_file():
        raise RuntimeError(f"Recording entry point was not found: {RECORD_SCRIPT}")

    run_name = f"{RUN_PREFIXES[profile]}{index:03d}"
    dataset_root = DATA_DIR / run_name
    repo_id = f"local/{run_name}"
    command = [
        sys.executable,
        str(RECORD_SCRIPT),
        f"--config_path={CONFIG_PATH}",
        *extra_args,
        *(["--robot.depth_camera_names=[]"] if profile == "rgb" else []),
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
    ]
    return command, dataset_root, repo_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select the first unused depth or RGB dataset directory and "
            "launch the MuJoCo recording entry point."
        )
    )
    parser.add_argument(
        "--profile",
        choices=tuple(RUN_PREFIXES),
        default="depth",
        help="Record RGB+depth by default; use 'rgb' for the legacy two-RGB schema.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected dataset and command without starting recording.",
    )
    args, extra_args = parser.parse_known_args()

    if args.dry_run:
        index = next_run_index(profile=args.profile)
        command, dataset_root, repo_id = build_record_command(
            index, extra_args, profile=args.profile
        )
        print(f"Selected dataset: {repo_id}", flush=True)
        print(f"Dataset root:     {dataset_root}", flush=True)
        print("Command:", flush=True)
        print("  " + shlex.join(command), flush=True)
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Use a new lock name so a still-running launcher from the old version,
    # which held .record_next.lock for the whole recording, cannot deadlock
    # this corrected short-lock implementation.
    run_prefix = RUN_PREFIXES[args.profile]
    lock_path = DATA_DIR / f".record_next_{args.profile}_number.lock"
    with lock_path.open("w") as lock_file:
        # Only hold the lock while selecting and reserving a run number. The
        # previous implementation held it for the entire recording, which made
        # a second invocation appear to freeze with no output.
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        index = next_run_index(profile=args.profile)
        command, dataset_root, repo_id = build_record_command(
            index, extra_args, profile=args.profile
        )
        reservation_path = DATA_DIR / f".{run_prefix}{index:03d}.reserve"
        reservation_path.write_text(f"pid={os.getpid()}\n", encoding="utf-8")

    print(f"Selected dataset: {repo_id}", flush=True)
    print(f"Dataset root:     {dataset_root}", flush=True)
    print("Command:", flush=True)
    print("  " + shlex.join(command), flush=True)
    env = os.environ.copy()
    # The fixed-camera panel uses a desktop GLFW context. Override an inherited
    # headless backend such as ``osmesa`` so MuJoCo and the panel use X11.
    env["MUJOCO_GL"] = "glfw"
    try:
        return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
    finally:
        # The dataset directory itself reserves successful/partial recordings.
        # Remove the temporary reservation when this launcher exits.
        reservation_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
