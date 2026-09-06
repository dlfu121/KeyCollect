from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import record_mujoco
from scripts.merge_datasets import merge_governance_sidecars


class _Dataset:
    def __init__(self, root: Path):
        self.root = root
        self.num_episodes = 3


class RecordingGovernanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_save = record_mujoco._upstream_save_episode
        record_mujoco._pending_episode_metadata.clear()

    def tearDown(self) -> None:
        record_mujoco._upstream_save_episode = self.original_save
        record_mujoco._pending_episode_metadata.clear()

    def test_metadata_is_committed_only_after_dataset_save_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = _Dataset(Path(temporary))
            record_mujoco._pending_episode_metadata[id(dataset)] = {"episode_index": 3}

            def fail(_dataset):
                raise RuntimeError("save failed")

            record_mujoco._upstream_save_episode = fail
            with self.assertRaisesRegex(RuntimeError, "save failed"):
                record_mujoco.save_episode_with_keycollect_metadata(dataset)
            sidecar = dataset.root / "meta/keycollect_episodes.jsonl"
            self.assertFalse(sidecar.exists())
            self.assertIn(id(dataset), record_mujoco._pending_episode_metadata)

            record_mujoco._upstream_save_episode = lambda _dataset: "saved"
            self.assertEqual(record_mujoco.save_episode_with_keycollect_metadata(dataset), "saved")
            self.assertEqual(json.loads(sidecar.read_text()), {"episode_index": 3})
            self.assertNotIn(id(dataset), record_mujoco._pending_episode_metadata)

    def test_merge_remaps_episode_and_timing_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [root / "run_001", root / "run_002"]
            for run, episode_count in zip(runs, (2, 1)):
                (run / "meta").mkdir(parents=True)
                (run / "meta/info.json").write_text(json.dumps({"total_episodes": episode_count}))
                (run / "meta/keycollect_episodes.jsonl").write_text(
                    "".join(json.dumps({"episode_index": index}) + "\n" for index in range(episode_count))
                )
                (run / "meta/keycollect_frame_timing.jsonl").write_text(
                    json.dumps({"episode_index": 0, "record_type": "frame_timing"}) + "\n"
                )
            output = root / "merged"
            merge_governance_sidecars(runs, output)
            episodes = [
                json.loads(line)
                for line in (output / "meta/keycollect_episodes.jsonl").read_text().splitlines()
            ]
            timing = [
                json.loads(line)
                for line in (output / "meta/keycollect_frame_timing.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["episode_index"] for row in episodes], [0, 1, 2])
            self.assertEqual([row["episode_index"] for row in timing], [0, 2])


if __name__ == "__main__":
    unittest.main()
