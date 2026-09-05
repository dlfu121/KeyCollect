#!/usr/bin/env python3
"""Run a trained ACT checkpoint to control the RM65 + DexHand MuJoCo scene.

Reuses the same `MuJoCoRobot` plugin used for data collection, so observations
and actions match the training distribution exactly:

* `observation.state` = [6 arm pos, 6 arm vel, 20 finger pos, 7 ee_pose] (float32)
* RGB observations = float32 CHW in [0, 1]
* depth observations = float32 CHW in metres
* action = 26 delta commands (6 cartesian + 20 finger deltas)

Inference uses CUDA when requested and available. ACT caches an action chunk,
so it does not need to run a new network forward pass on every control step.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors

from lerobot_robot_mujoco import MuJoCoRobot, MuJoCoRobotConfig
from lerobot.cameras import CameraConfig
from scripts.act_depth_adapter import install_act_depth_adapter


class _RenderCameraConfig(CameraConfig):
    """Minimal concrete camera config (MuJoCo only reads width/height/fps)."""

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
EE_POSE_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")


def find_checkpoint(checkpoint: Path) -> Path:
    pretrained = Path(checkpoint) / "pretrained_model"
    if pretrained.exists():
        return pretrained
    steps = sorted(Path(checkpoint).glob("checkpoints/*/pretrained_model"))
    if not steps:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint}")
    return steps[-1]


def load_robot_config(yaml_path: Path) -> MuJoCoRobotConfig:
    raw = yaml.safe_load(yaml_path.read_text())["robot"]
    raw["cameras"] = {
        name: _RenderCameraConfig(
            width=cfg.get("width"),
            height=cfg.get("height"),
            fps=cfg.get("fps"),
        )
        for name, cfg in raw["cameras"].items()
    }
    for drop in ("type", "id"):
        raw.pop(drop, None)
    return MuJoCoRobotConfig(**raw)


def build_state(obs: dict) -> np.ndarray:
    arm_pos = np.asarray([obs[f"{j}.pos"] for j in ARM_JOINTS], dtype=np.float32)
    arm_vel = np.asarray([obs[f"{j}.vel"] for j in ARM_JOINTS], dtype=np.float32)
    finger_pos = np.asarray(
        [obs[k] for k in obs if k.endswith(".pos") and k.startswith("r_f_")],
        dtype=np.float32,
    )
    ee_pose = np.asarray([obs[f"ee_pose.{n}"] for n in EE_POSE_NAMES], dtype=np.float32)
    return np.concatenate([arm_pos, arm_vel, finger_pos, ee_pose])


def img_to_float_chw(img: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(img)
    t = torch.from_numpy(array).float()
    if array.dtype == np.uint8:
        t = t / 255.0
    return t.permute(2, 0, 1).contiguous()


def make_observation(obs: dict, cameras: list[str]) -> dict:
    return {
        **{f"observation.images.{cam}": img_to_float_chw(obs[cam]) for cam in cameras},
        "observation.state": torch.from_numpy(build_state(obs)),
    }


def action_to_dict(action: np.ndarray, gripper_joints: list[str]) -> dict:
    delta = {
        "delta_x": float(action[0]),
        "delta_y": float(action[1]),
        "delta_z": float(action[2]),
        "delta_roll": float(action[3]),
        "delta_pitch": float(action[4]),
        "delta_yaw": float(action[5]),
    }
    for i, joint in enumerate(gripper_joints):
        delta[f"{joint}.delta"] = float(action[6 + i])
    return delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/train/act_rm65_dexhand/checkpoints/last",
        help="Checkpoint dir (containing pretrained_model/) or pretrained_model dir itself.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "record_mujoco.yaml",
        help="Robot config YAML (reuses the recording robot section).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data" / "rm65_dexhand_merged",
        help="Merged dataset root, used for policy feature shapes.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to run before exiting (0 = until the viewer closes).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Policy device; 'auto' selects CUDA when available.",
    )
    parser.add_argument(
        "--no-randomize",
        action="store_true",
        help="Keep screwdrivers at their authored pose instead of randomizing.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed for screwdriver randomization.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, save the table camera frames as a video + PNG previews.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    install_act_depth_adapter()
    pretrained_dir = find_checkpoint(args.checkpoint)
    print(f"Using checkpoint: {pretrained_dir}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but torch.cuda.is_available() is false."
        )
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device == "auto":
        device = "cpu"
        print("CUDA unavailable; running inference on CPU.")

    ds_meta = LeRobotDatasetMetadata(
        "local/rm65_dexhand_merged", root=args.dataset_root
    )
    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_dir)
    policy_cfg.pretrained_path = str(pretrained_dir)
    policy_cfg.device = device
    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    print(f"Policy ready: {policy.config.type} on {device}")

    robot_cfg = load_robot_config(args.config)
    if args.no_randomize:
        robot_cfg.randomize_screwdrivers = False
    robot = MuJoCoRobot(robot_cfg)
    robot.connect()
    cameras = [
        key.removeprefix("observation.images.")
        for key in policy.config.image_features
    ]
    missing_cameras = [camera for camera in cameras if camera not in robot.observation_features]
    if missing_cameras:
        raise ValueError(
            f"Checkpoint requires unavailable camera observations: {missing_cameras}"
        )
    gripper_joints = list(robot_cfg.gripper_joint_names)
    period = 1.0 / max(1, robot_cfg.control_fps)
    print(f"Controlling {cameras} cameras at {robot_cfg.control_fps} Hz "
          f"({len(gripper_joints)} finger joints).")

    writer = None
    start = time.monotonic()
    step = 0
    try:
        while robot.is_connected:
            if not robot.simulation.sync_viewer():
                break
            obs = robot.get_observation()
            policy_obs = make_observation(obs, cameras)
            with torch.inference_mode():
                action_tensor = policy.select_action(preprocessor(policy_obs))
            action_vector = postprocessor(action_tensor)[0].numpy()
            robot.send_action(action_to_dict(action_vector, gripper_joints))

            if args.save_dir is not None:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                frame = obs[cameras[0]]
                if writer is None:
                    import cv2
                    writer = cv2.VideoWriter(
                        str(args.save_dir / "infer.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        robot_cfg.control_fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                writer.write(frame)

            step += 1
            if args.duration > 0 and time.monotonic() - start >= args.duration:
                break
            time.sleep(period)
    finally:
        if writer is not None:
            writer.release()
            print(f"Saved video: {args.save_dir / 'infer.mp4'}")
        robot.disconnect()

    print(f"Inference finished after {step} steps ({time.monotonic() - start:.1f} s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
