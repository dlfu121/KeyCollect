#!/usr/bin/env python3
"""Audit a LeRobot dataset and emit episode-safe governance manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "rm65_dexhand_merged")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "dataset_audit")
    parser.add_argument("--expected-episodes", type=int, default=127)
    parser.add_argument("--expected-frames", type=int, default=39673)
    parser.add_argument("--require-governance", action="store_true")
    args = parser.parse_args()

    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Dataset audit requires pyarrow; install the locked audit dependencies.") from exc

    info_path = args.dataset / "meta" / "info.json"
    frame_path = args.dataset / "data" / "chunk-000" / "file-000.parquet"
    episode_path = args.dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    frames = pq.read_table(
        frame_path,
        columns=["action", "observation.state", "timestamp", "frame_index", "episode_index", "index"],
    )
    episodes = pq.read_table(
        episode_path,
        columns=["episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index"],
    ).to_pylist()

    action_lengths = set(pc.unique(pc.list_value_length(frames["action"])).to_pylist())
    state_lengths = set(pc.unique(pc.list_value_length(frames["observation.state"])).to_pylist())
    episode_indices = frames["episode_index"].to_pylist()
    frame_indices = frames["frame_index"].to_pylist()
    timestamps = frames["timestamp"].to_pylist()
    indices = frames["index"].to_pylist()
    errors: list[str] = []
    governance_path = args.dataset / "meta" / "keycollect_episodes.jsonl"
    timing_path = args.dataset / "meta" / "keycollect_frame_timing.jsonl"
    governance: dict[int, dict[str, Any]] = {}
    if governance_path.exists():
        for line_number, line in enumerate(governance_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            episode_index = int(value["episode_index"])
            if episode_index in governance:
                errors.append(f"duplicate governance episode {episode_index} at line {line_number}")
            governance[episode_index] = value
    timing_by_episode: dict[int, list[dict[str, Any]]] = {}
    if timing_path.exists():
        for line in timing_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("record_type") == "frame_timing":
                timing_by_episode.setdefault(int(value["episode_index"]), []).append(value)
    if frames.num_rows != args.expected_frames:
        errors.append(f"frame count {frames.num_rows} != expected {args.expected_frames}")
    if len(episodes) != args.expected_episodes:
        errors.append(f"episode count {len(episodes)} != expected {args.expected_episodes}")
    if action_lengths != {26} or state_lengths != {39}:
        errors.append(f"vector dimensions action={action_lengths}, state={state_lengths}")
    if indices != list(range(frames.num_rows)):
        errors.append("global index is not contiguous")
    if sum(int(row["length"]) for row in episodes) != frames.num_rows:
        errors.append("episode lengths do not sum to total frame count")
    if args.require_governance and len(governance) != len(episodes):
        errors.append(f"governance rows {len(governance)} != episodes {len(episodes)}")

    args.output.mkdir(parents=True, exist_ok=True)
    scene_path = ROOT / "assets" / "scenes" / "rm65_dexhand_scene.xml"
    scene_hash = sha256(scene_path)
    # Contiguous complete-episode split. The 40k checkpoint predates this split
    # and is therefore explicitly ineligible for independent-test claims.
    train_end = int(len(episodes) * 0.80)
    validation_end = train_end + int(len(episodes) * 0.10)
    split = {
        "schema_version": 1,
        "unit": "complete_episode",
        "seed": 1000,
        "train": list(range(0, train_end)),
        "validation": list(range(train_end, validation_end)),
        "test": list(range(validation_end, len(episodes))),
        "applies_to_existing_40k_checkpoint": False,
    }
    (args.output / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

    fields = [
        "episode_index", "frame_count", "from_index", "to_index_exclusive", "start_timestamp_s",
        "end_timestamp_s", "split", "task", "success", "failure_reason", "object_initial_pose",
        "scene_config", "scene_sha256", "random_seed", "collection_batch", "dataset_version",
        "schema_name", "source_time_recorded", "receive_time_recorded", "observation_time_recorded",
        "action_execution_time_recorded", "annotation_status",
    ]
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        local_frames = frame_indices[start:end]
        local_episodes = episode_indices[start:end]
        if local_frames != list(range(end - start)) or set(local_episodes) != {episode_index}:
            errors.append(f"episode {episode_index} frame/index boundary is inconsistent")
        expected_timestamps = [index / float(info["fps"]) for index in range(end - start)]
        if any(abs(actual - expected) > 1e-4 for actual, expected in zip(timestamps[start:end], expected_timestamps)):
            errors.append(f"episode {episode_index} timestamps do not match declared fps={info['fps']}")
        split_name = next(name for name in ("train", "validation", "test") if episode_index in split[name])
        provenance = governance.get(episode_index)
        timing_records = timing_by_episode.get(episode_index, [])
        timing_complete = len(timing_records) == end - start and all(
            all(key in record for key in ("source_time", "receive_time", "observation_time", "action_execution_time"))
            for record in timing_records
        )
        missing = "UNAVAILABLE_NOT_RECORDED"
        rows.append({
            "episode_index": episode_index,
            "frame_count": end - start,
            "from_index": start,
            "to_index_exclusive": end,
            "start_timestamp_s": timestamps[start] if start < end else "",
            "end_timestamp_s": timestamps[end - 1] if start < end else "",
            "split": split_name,
            "task": " | ".join(episode["tasks"] or []),
            "success": provenance.get("success") if provenance else "PENDING_ANNOTATION",
            "failure_reason": provenance.get("failure_reason") if provenance else "PENDING_ANNOTATION",
            "object_initial_pose": json.dumps(provenance.get("object_initial_pose")) if provenance else missing,
            # The legacy parquet metadata did not capture scene provenance.
            # Do not attribute today's generated artifact to historical data.
            "scene_config": json.dumps(provenance.get("scene_config"), sort_keys=True) if provenance else missing,
            "scene_sha256": provenance.get("scene_sha256", missing) if provenance else missing,
            "random_seed": provenance.get("random_seed", missing) if provenance else missing,
            "collection_batch": provenance.get("collection_batch", missing) if provenance else missing,
            "dataset_version": provenance.get("dataset_version", info.get("codebase_version", "UNKNOWN")) if provenance else info.get("codebase_version", "UNKNOWN"),
            "schema_name": provenance.get("dataset_version", "keycollect_act_rgb_v1") if provenance else "keycollect_act_rgb_v1",
            "source_time_recorded": timing_complete,
            "receive_time_recorded": timing_complete,
            "observation_time_recorded": timing_complete,
            "action_execution_time_recorded": timing_complete,
            "annotation_status": "COMPLETE" if provenance and timing_complete and provenance.get("success") is not None else "INCOMPLETE_LEGACY_METADATA" if provenance is None else "INCOMPLETE_GOVERNANCE",
        })
    with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "status": "passed_integrity_checks" if not errors else "failed_integrity_checks",
        "dataset": str(args.dataset),
        "metadata_total_episodes": info.get("total_episodes"),
        "metadata_total_frames": info.get("total_frames"),
        "audited_episode_rows": len(episodes),
        "audited_frame_rows": frames.num_rows,
        "fps": info.get("fps"),
        "action_lengths": sorted(action_lengths),
        "state_lengths": sorted(state_lengths),
        "frame_parquet_sha256": sha256(frame_path),
        "episode_parquet_sha256": sha256(episode_path),
        "current_generated_scene": str(scene_path.relative_to(ROOT)),
        "current_generated_scene_sha256": scene_hash,
        "integrity_errors": errors,
        "governance_missing_for_all_legacy_episodes": [
            "success", "failure_reason", "object_initial_pose", "scene_config", "scene_sha256",
            "random_seed", "collection_batch",
            "source_time", "receive_time", "observation_time", "action_execution_time",
        ],
        "governance_sidecar": str(governance_path) if governance_path.exists() else "UNAVAILABLE_NOT_RECORDED",
        "governance_rows": len(governance),
        "timing_sidecar": str(timing_path) if timing_path.exists() else "UNAVAILABLE_NOT_RECORDED",
        "existing_dataset_declared_splits": info.get("splits", {}),
        "governed_split": {key: len(value) for key, value in split.items() if isinstance(value, list)},
        "existing_40k_checkpoint_independent_test_eligible": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
