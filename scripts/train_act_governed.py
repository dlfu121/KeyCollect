#!/usr/bin/env python3
"""Launch ACT training with an immutable episode split and auditable run record.

Arguments after ``--`` are forwarded to LeRobot.  Split, seed, output directory,
validation cadence, and disabled environment evaluation are controlled here so
the independent test episodes cannot accidentally enter training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def installed_versions() -> dict[str, str]:
    names = ("lerobot", "torch", "torchvision", "mujoco", "numpy", "pyarrow", "safetensors")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def hardware_record() -> dict[str, Any]:
    result: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except ImportError:
        result["cuda_available"] = False
        result["cuda_devices"] = []
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("rgb", "rgbd"), default="rgb")
    parser.add_argument("--split-file", type=Path, default=ROOT / "artifacts/dataset_audit/split.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if args.eval_steps <= 0:
        raise ValueError("--eval-steps must be positive so validation loss is recorded")

    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    train = [int(value) for value in split["train"]]
    validation = [int(value) for value in split["validation"]]
    test = [int(value) for value in split["test"]]
    selected = train + validation
    if split.get("unit") != "complete_episode":
        raise ValueError("split unit must be complete_episode")
    if any(set(left) & set(right) for left, right in ((train, validation), (train, test), (validation, test))):
        raise ValueError("train/validation/test episode sets overlap")
    if selected != list(range(len(selected))):
        raise ValueError("LeRobot held-out-tail split requires contiguous train then validation episodes")

    # Choose a point strictly inside the interval that makes ceil(N*p) equal
    # the requested validation count, avoiding floating-point boundary issues.
    eval_split = (len(validation) - 0.5) / len(selected)
    controlled_prefixes = (
        "--dataset.episodes", "--dataset.eval_split", "--seed", "--output_dir",
        "--eval_steps", "--env_eval_freq",
    )
    conflicts = [item for item in forwarded if any(item == key or item.startswith(key + "=") for key in controlled_prefixes)]
    if conflicts:
        raise ValueError(f"governed arguments cannot be overridden: {conflicts}")

    entry = ROOT / "scripts" / ("train_act_depth.py" if args.profile == "rgbd" else "_train_act_upstream.py")
    if args.profile == "rgb":
        command = [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    else:
        command = [sys.executable, str(entry)]
    command += [
        f"--dataset.episodes={json.dumps(selected, separators=(',', ':'))}",
        f"--dataset.eval_split={eval_split:.12f}",
        f"--seed={args.seed}",
        f"--output_dir={args.output_dir}",
        f"--eval_steps={args.eval_steps}",
        "--env_eval_freq=0",
        *forwarded,
    ]

    manifest_path = args.output_dir / "keycollect_training_run.json"
    log_path = args.output_dir / "training.log"
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "planned" if args.dry_run else "running",
        "profile": args.profile,
        "command": command,
        "split_file": str(args.split_file),
        "split_file_sha256": file_sha256(args.split_file),
        "episodes": {"train": train, "validation": validation, "independent_test_excluded": test},
        "seed": args.seed,
        "dependencies": installed_versions(),
        "dependency_lock": str(ROOT / "uv.lock"),
        "dependency_lock_sha256": file_sha256(ROOT / "uv.lock"),
        "hardware": hardware_record(),
        "started_at_unix_s": None if args.dry_run else time.time(),
        "duration_s": "PENDING_EXECUTION" if args.dry_run else None,
        "checkpoint_hashes": "PENDING_EXECUTION",
    }
    write_manifest(manifest_path, record)
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    record["duration_s"] = time.perf_counter() - started
    record["finished_at_unix_s"] = time.time()
    record["return_code"] = return_code
    record["status"] = "completed" if return_code == 0 else "failed"
    record["train_and_validation_loss_log"] = str(log_path)
    record["train_and_validation_loss_log_sha256"] = file_sha256(log_path)
    checkpoints = sorted(args.output_dir.glob("checkpoints/*/pretrained_model/model.safetensors"))
    record["checkpoint_hashes"] = {str(path.relative_to(args.output_dir)): file_sha256(path) for path in checkpoints}
    write_manifest(manifest_path, record)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
