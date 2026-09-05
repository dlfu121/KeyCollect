#!/usr/bin/env python3
"""Merge compatible recorded RM65 datasets into one LeRobot dataset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

ROOT = Path(__file__).resolve().parents[1]
RUN_PREFIXES = {
    "depth": "rm65_dexhand_depth_run_",
    "rgb": "rm65_dexhand_run_",
}


def collect_runs(data_root: Path, profile: str = "depth") -> list[Path]:
    run_pattern = re.compile(rf"^{re.escape(RUN_PREFIXES[profile])}(\d+)$")
    runs = sorted(
        (path for path in data_root.iterdir() if run_pattern.fullmatch(path.name)),
        key=lambda path: int(run_pattern.fullmatch(path.name).group(1)),
    )
    if not runs:
        raise RuntimeError(
            f"No {RUN_PREFIXES[profile]}* datasets found in {data_root}"
        )
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge all rm65_dexhand_run_NNN datasets under ROOT into one."
    )
    parser.add_argument(
        "--profile",
        choices=tuple(RUN_PREFIXES),
        default="depth",
        help="Merge RGB+depth runs by default; use 'rgb' for legacy runs.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/media/ee304/FDL"),
        help="Directory containing the run_NNN dataset folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (profile-specific default under data/).",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Repo id (profile-specific default when omitted).",
    )
    parser.add_argument(
        "--exclude",
        type=int,
        nargs="*",
        default=[],
        help="Run numbers to skip (e.g. --exclude 1 2 skips run_001 and run_002).",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        suffix = "rm65_dexhand_depth_merged" if args.profile == "depth" else "rm65_dexhand_merged"
        args.output_dir = ROOT / "data" / suffix
    if args.repo_id is None:
        name = "rm65_dexhand_depth_merged" if args.profile == "depth" else "rm65_dexhand_merged"
        args.repo_id = f"local/{name}"

    excluded = set(args.exclude)
    run_pattern = re.compile(rf"^{re.escape(RUN_PREFIXES[args.profile])}(\d+)$")
    runs = [
        run
        for run in collect_runs(args.data_root, args.profile)
        if int(run_pattern.fullmatch(run.name).group(1)) not in excluded
    ]
    if not runs:
        raise RuntimeError("No datasets left to merge after applying --exclude.")
    repo_ids = [f"local/{run.name}" for run in runs]
    print(f"Merging {len(runs)} datasets into {args.repo_id} @ {args.output_dir}")
    for run in runs:
        print(f"  {run.name}")

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=args.repo_id,
        roots=runs,
        aggr_root=args.output_dir,
    )
    print(f"Merged dataset ready: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
