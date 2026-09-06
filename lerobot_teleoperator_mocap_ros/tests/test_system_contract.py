from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from keycollect.act_validation import validate_act_artifacts
from keycollect.contract import CONTRACT, ContractError, per_cycle_limits
from keycollect.evaluation import LiftSuccessTracker, generate_cases, load_protocol


class SystemContractTest(unittest.TestCase):
    def test_rate_limits_scale_with_actual_control_period(self) -> None:
        limits_24 = per_cycle_limits(24)
        limits_30 = per_cycle_limits(30)
        self.assertAlmostEqual(limits_24["translation_m"], 0.005)
        self.assertAlmostEqual(limits_24["rotation_rad"], 0.04)
        self.assertAlmostEqual(limits_24["finger_rad"], 0.05)
        self.assertAlmostEqual(limits_24["translation_m"] * 24, limits_30["translation_m"] * 30)
        self.assertEqual(len(CONTRACT["act"]["state_units"]), 39)
        self.assertEqual(len(CONTRACT["act"]["action_units"]), 26)

    def test_generated_scene_matches_home_camera_and_timestep_contract(self) -> None:
        root = ET.parse(ROOT / "assets/scenes/rm65_dexhand_scene.xml").getroot()
        option = root.find("option")
        self.assertAlmostEqual(float(option.get("timestep")), CONTRACT["timing"]["mujoco_integration_step_s"])
        home = root.find("./keyframe/key[@name='home']")
        qpos = [float(value) for value in home.get("qpos").split()]
        expected_home = CONTRACT["geometry"]["arm_home_deg"] + CONTRACT["geometry"]["dexhand_home_deg"]
        import math
        for actual, expected_deg in zip(qpos[:26], expected_home):
            self.assertAlmostEqual(actual, math.radians(expected_deg), places=6)
        for name in ("table_camera", "wrist_overhead_camera"):
            camera = root.find(f".//camera[@name='{name}']")
            expected = CONTRACT["geometry"][name]
            self.assertEqual([float(x) for x in camera.get("pos").split()], expected["position_m"])
            self.assertEqual([float(x) for x in camera.get("xyaxes").split()], expected["xyaxes"])
            self.assertAlmostEqual(float(camera.get("fovy")), expected["fovy_deg"])

    def test_existing_act_artifacts_pass_and_frequency_mismatch_fails_closed(self) -> None:
        checkpoint = ROOT / "outputs/train/act_rm65_dexhand/checkpoints/040000/pretrained_model"
        dataset = ROOT / "data/rm65_dexhand_merged"
        robot = yaml.safe_load((ROOT / "config/infer_act_rgb30.yaml").read_text())["robot"]
        report = validate_act_artifacts(dataset, checkpoint, robot)
        self.assertEqual(report["state_dimension"], 39)
        self.assertEqual(report["action_dimension"], 26)
        mismatched = copy.deepcopy(robot)
        mismatched["control_fps"] = 24
        with self.assertRaisesRegex(ContractError, "frequency mismatch"):
            validate_act_artifacts(dataset, checkpoint, mismatched)

    def test_rgbd_profile_is_distinct_from_legacy_rgb(self) -> None:
        recording = yaml.safe_load((ROOT / "config/record_mujoco.yaml").read_text())
        legacy = yaml.safe_load((ROOT / "config/infer_act_rgb30.yaml").read_text())
        self.assertEqual(recording["dataset"]["fps"], 24)
        self.assertEqual(legacy["robot"]["control_fps"], 30)
        self.assertEqual(recording["robot"]["depth_camera_names"], ["table_camera"])
        self.assertEqual(legacy["robot"]["depth_camera_names"], [])

    def test_fixed_evaluation_matrix_and_success_hold_are_deterministic(self) -> None:
        protocol = load_protocol(ROOT / "config/task_evaluation.yaml")
        cases = generate_cases(protocol)
        self.assertEqual(len(cases), 100)
        self.assertEqual(len({case["case_id"] for case in cases}), 100)
        self.assertEqual(len({case["random_seed"] for case in cases}), 100)
        tracker = LiftSuccessTracker(initial_z=1.0, minimum_lift_m=0.05, minimum_hold_s=2.0)
        self.assertFalse(tracker.successful(0.0, 1.06))
        self.assertFalse(tracker.successful(1.9, 1.06))
        self.assertTrue(tracker.successful(2.0, 1.06))
        tracker.update(2.1, 1.01)
        self.assertTrue(tracker.slipped_after_threshold)

    def test_governed_training_excludes_complete_test_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/train_act_governed.py"),
                    "--output-dir", str(output),
                    "--dry-run",
                    "--policy.type=act",
                    "--dataset.repo_id=local/rm65_dexhand_merged",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output / "keycollect_training_run.json").read_text())
            episodes = manifest["episodes"]
            self.assertEqual(len(episodes["train"]), 101)
            self.assertEqual(len(episodes["validation"]), 12)
            self.assertEqual(len(episodes["independent_test_excluded"]), 14)
            selected = set(episodes["train"] + episodes["validation"])
            self.assertFalse(selected & set(episodes["independent_test_excluded"]))


if __name__ == "__main__":
    unittest.main()
