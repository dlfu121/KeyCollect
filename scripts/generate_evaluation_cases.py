#!/usr/bin/env python3
"""Generate the complete deterministic ACT task matrix as JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from keycollect.evaluation import generate_cases, load_protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "config/task_evaluation.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/evaluation_cases.jsonl")
    args = parser.parse_args()
    cases = generate_cases(load_protocol(args.protocol))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases), encoding="utf-8")
    print(f"Wrote {len(cases)} fixed cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
