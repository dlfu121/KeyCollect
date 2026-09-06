#!/usr/bin/env python3
"""Create an evidence-only ACT training/checkpoint manifest."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from keycollect.act_validation import sha256_file, validate_act_artifacts


def version(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/train/act_rm65_dexhand/checkpoints/040000/pretrained_model")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/rm65_dexhand_merged")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "act_training_manifest.json")
    args = parser.parse_args()
    robot = yaml.safe_load((ROOT / "config/infer_act_rgb30.yaml").read_text(encoding="utf-8"))["robot"]
    validation = validate_act_artifacts(args.dataset, args.checkpoint, robot)
    train_config = json.loads((args.checkpoint / "train_config.json").read_text(encoding="utf-8"))
    training_log = args.checkpoint.parents[1] / "trainlog.txt"
    manifest = {
        "status": "checkpoint_artifacts_valid_but_training_and_evaluation_record_incomplete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": 40000,
        "checkpoint_sha256": validation["checkpoint_sha256"],
        "checkpoint_files": {
            path.name: sha256_file(path) for path in sorted(args.checkpoint.iterdir()) if path.is_file()
        },
        "random_seed": train_config.get("seed"),
        "complete_train_config": train_config,
        "dataset_declared_split": "train=0:127 (all episodes)",
        "governed_split_available_after_training": "artifacts/dataset_audit/split.json",
        "train_loss_history": "UNAVAILABLE_EMPTY_TRAINLOG",
        "validation_loss_history": "UNAVAILABLE_NO_VALIDATION_SPLIT_OR_EVAL_STEPS",
        "training_hardware": "UNAVAILABLE_NOT_RECORDED",
        "training_duration": "UNAVAILABLE_NOT_RECORDED",
        "dependency_versions_at_training_time": "UNAVAILABLE_NOT_RECORDED",
        "current_audit_host": platform.platform(),
        "current_python": platform.python_version(),
        "current_git_revision": version(["git", "rev-parse", "HEAD"]),
        "independent_test_evaluation": "PENDING_RETRAIN_WITH_GOVERNED_SPLIT_AND_RUN_FIXED_TASKS",
        "grasp_success_rate": "PENDING_MEASUREMENT",
        "training_log_bytes": training_log.stat().st_size if training_log.exists() else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "checkpoint_sha256", "grasp_success_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
