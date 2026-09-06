#!/usr/bin/env python3
"""Execute fixed ACT cases and aggregate only measurements actually produced."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "artifacts/evaluation_cases.jsonl")
    parser.add_argument("--protocol", type=Path, default=ROOT / "config/task_evaluation.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/act_evaluation_runs")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/infer_act_rgb30.yaml")
    parser.add_argument("--training-run-manifest", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--case-indices", nargs="*", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line]
    indices = args.case_indices if args.case_indices is not None else list(range(len(cases)))
    if not indices or any(index < 0 or index >= len(cases) for index in indices):
        raise ValueError("case indices are empty or out of range")
    duration = float(yaml.safe_load(args.protocol.read_text(encoding="utf-8"))["episode_duration_s"])
    commands: list[list[str]] = []
    for index in indices:
        run_dir = args.output / cases[index]["case_id"]
        result_path = run_dir / "result.json"
        if result_path.exists() and not args.dry_run:
            raise FileExistsError(f"Refusing to overwrite completed result: {result_path}")
        command = [
            sys.executable,
            str(ROOT / "scripts/infer_mujoco.py"),
            "--headless",
            "--duration", str(duration),
            "--checkpoint", str(args.checkpoint),
            "--dataset-root", str(args.dataset_root),
            "--config", str(args.config),
            "--device", args.device,
            "--evaluation-cases", str(args.cases),
            "--case-index", str(index),
            "--evaluation-protocol", str(args.protocol),
            "--evaluation-result", str(result_path),
            "--runtime-log", str(run_dir / "runtime.jsonl"),
            "--save-dir", str(run_dir / "video"),
        ]
        if args.training_run_manifest is not None:
            command += ["--training-run-manifest", str(args.training_run_manifest)]
        commands.append(command)

    if args.dry_run:
        print(json.dumps({"case_count": len(commands), "commands": commands}, indent=2))
        return 0

    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    results = [json.loads((args.output / cases[index]["case_id"] / "result.json").read_text()) for index in indices]
    independent = all(result["independent_test_eligible"] for result in results)
    summary = {
        "status": "completed_independent_evaluation" if independent else "completed_diagnostic_evaluation",
        "case_count": len(results),
        "successes": sum(bool(result["success"]) for result in results),
        "success_rate": sum(bool(result["success"]) for result in results) / len(results),
        "independent_test_eligible": independent,
        "result_files": [str(args.output / cases[index]["case_id"] / "result.json") for index in indices],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
