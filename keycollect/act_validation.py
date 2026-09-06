"""Strict, dependency-light validation of ACT dataset/checkpoint/runtime artifacts."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .contract import CONTRACT, ContractError, action_names, state_names, tactile_state_names


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"Required ACT artifact is missing: {path}") from exc


def _safetensor_shapes(path: Path) -> dict[str, list[int]]:
    try:
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size))
    except (FileNotFoundError, ValueError, struct.error, json.JSONDecodeError) as exc:
        raise ContractError(f"Invalid or missing safetensors artifact: {path}") from exc
    return {key: value["shape"] for key, value in header.items() if key != "__metadata__"}


def _safetensor_f32(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        data_start = 8 + header_size
        result: dict[str, np.ndarray] = {}
        for name in names:
            descriptor = header.get(name)
            if descriptor is None or descriptor.get("dtype") != "F32":
                raise ContractError(f"Missing F32 statistic {name} in {path}")
            start, end = descriptor["data_offsets"]
            stream.seek(data_start + start)
            result[name] = np.frombuffer(stream.read(end - start), dtype="<f4").copy()
        return result


def _shape(feature: dict[str, Any]) -> list[int]:
    value = feature.get("shape")
    if not isinstance(value, list):
        raise ContractError(f"Feature has no valid shape: {feature}")
    return [int(item) for item in value]


def validate_act_artifacts(
    dataset_root: Path,
    pretrained_dir: Path,
    robot_config: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed on schema, frequency, camera, processor, or stats mismatch."""
    info = _json(dataset_root / "meta" / "info.json")
    stats = _json(dataset_root / "meta" / "stats.json")
    policy = _json(pretrained_dir / "config.json")
    train = _json(pretrained_dir / "train_config.json")
    pre = _json(pretrained_dir / "policy_preprocessor.json")
    post = _json(pretrained_dir / "policy_postprocessor.json")
    errors: list[str] = []

    features = info.get("features", {})
    state = features.get("observation.state", {})
    action = features.get("action", {})
    base_state, expected_action = state_names(), action_names()
    dataset_state_names = state.get("names", [])
    has_tactile = any(str(name).startswith("tactile.") for name in dataset_state_names)
    expected_state = base_state + (tactile_state_names() if has_tactile else [])
    state_dimension = len(expected_state)
    if len(CONTRACT["act"].get("state_units", [])) != len(base_state):
        errors.append("system contract must provide one unit for each state element")
    if len(CONTRACT["act"].get("action_units", [])) != len(expected_action):
        errors.append("system contract must provide one unit for each action element")
    if _shape(state) != [state_dimension] or dataset_state_names != expected_state:
        errors.append(f"dataset observation.state is not the ordered {state_dimension}-dimensional contract")
    if _shape(action) != [len(expected_action)] or action.get("names") != expected_action:
        errors.append("dataset action is not the ordered 26-dimensional contract")
    if bool(robot_config.get("tactile_enabled", False)) != has_tactile:
        errors.append("robot tactile observation setting differs from dataset state schema")
    expected_arm_joints = [f"joint_{index}" for index in range(1, 7)]
    expected_gripper_joints = [name.removesuffix(".delta") for name in expected_action[6:]]
    if robot_config.get("arm_joint_names") != expected_arm_joints:
        errors.append("robot arm joint order differs from ACT state contract")
    if robot_config.get("gripper_joint_names") != expected_gripper_joints:
        errors.append("robot gripper joint order differs from ACT state/action contract")
    if robot_config.get("ee_site_name") != CONTRACT["act"]["ee_pose_frame"]:
        errors.append("robot end-effector observation frame differs from ACT contract")

    expected_cameras = list(CONTRACT["act"]["rgb_cameras"])
    depth_key = str(CONTRACT["act"]["depth_camera"])
    has_depth = f"observation.images.{depth_key}" in features
    image_prefix = "observation.images."
    dataset_cameras = sorted(
        key[len(image_prefix) :]
        for key in features
        if key.startswith("observation.images.") and not key.endswith("_depth")
    )
    if dataset_cameras != sorted(expected_cameras):
        errors.append(f"dataset RGB cameras {dataset_cameras} != {expected_cameras}")
    for camera in expected_cameras:
        camera_feature = features.get(f"observation.images.{camera}", {})
        if _shape(camera_feature) != [480, 640, 3]:
            errors.append(f"dataset camera {camera} must be HWC 480x640x3")
    if has_depth:
        if _shape(features[f"observation.images.{depth_key}"]) != [480, 640, 1]:
            errors.append(f"dataset depth camera {depth_key} must be HWC 480x640x1")
        expected_base_depth = depth_key.removesuffix("_depth")
        if robot_config.get("depth_camera_names") != [expected_base_depth]:
            errors.append(
                f"robot depth cameras {robot_config.get('depth_camera_names')} != {[expected_base_depth]}"
            )
    elif robot_config.get("depth_camera_names"):
        errors.append("RGB dataset cannot run with depth observations enabled")
    if has_tactile and not has_depth:
        errors.append("KeyCollect tactile schema must use the independent RGB-D+tactile profile")

    dataset_fps = int(info.get("fps", 0))
    robot_fps = int(robot_config.get("control_fps", 0))
    camera_fps = {int(value.get("fps", 0)) for value in robot_config.get("cameras", {}).values()}
    if dataset_fps <= 0 or robot_fps != dataset_fps or camera_fps != {dataset_fps}:
        errors.append(
            f"frequency mismatch: dataset={dataset_fps}, control={robot_fps}, cameras={sorted(camera_fps)}"
        )

    policy_inputs = policy.get("input_features", {})
    if policy.get("type") != "act" or train.get("policy", {}).get("type") != "act":
        errors.append("checkpoint/train_config policy type must both be act")
    if int(policy.get("chunk_size", 0)) != int(CONTRACT["act"]["chunk_size"]):
        errors.append("checkpoint ACT chunk_size differs from system contract")
    if int(policy.get("n_action_steps", 0)) != int(CONTRACT["act"]["online_action_steps"]):
        errors.append("checkpoint ACT n_action_steps differs from system contract")
    if _shape(policy_inputs.get("observation.state", {})) != [state_dimension]:
        errors.append(f"checkpoint state input is not {state_dimension}-dimensional")
    if _shape(policy.get("output_features", {}).get("action", {})) != [26]:
        errors.append("checkpoint action output is not 26-dimensional")
    policy_cameras = sorted(
        key[len(image_prefix) :]
        for key in policy_inputs
        if key.startswith("observation.images.")
    )
    expected_policy_cameras = expected_cameras + ([depth_key] if has_depth else [])
    if policy_cameras != sorted(expected_policy_cameras):
        errors.append(f"checkpoint cameras {policy_cameras} != {expected_policy_cameras}")
    for camera in expected_cameras:
        if _shape(policy_inputs.get(f"observation.images.{camera}", {})) != [3, 480, 640]:
            errors.append(f"checkpoint camera {camera} must be CHW 3x480x640")
    if has_depth and _shape(policy_inputs.get(f"observation.images.{depth_key}", {})) != [1, 480, 640]:
        errors.append(f"checkpoint depth camera {depth_key} must be CHW 1x480x640")
    if policy.get("normalization_mapping") != {
        "VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"
    }:
        errors.append("checkpoint normalization mapping is not the ACT mean/std contract")

    train_policy = train.get("policy", {})
    if train_policy.get("input_features") != policy.get("input_features") or train_policy.get(
        "output_features"
    ) != policy.get("output_features"):
        errors.append("train_config policy features differ from checkpoint config")

    pre_state = pretrained_dir / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    post_state = pretrained_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    pre_shapes = _safetensor_shapes(pre_state)
    post_shapes = _safetensor_shapes(post_state)
    required_shapes = {
        "observation.state.mean": [state_dimension],
        "observation.state.std": [state_dimension],
        "action.mean": [26],
        "action.std": [26],
    }
    for key, shape in required_shapes.items():
        if pre_shapes.get(key) != shape:
            errors.append(f"preprocessor statistic {key} has shape {pre_shapes.get(key)}, expected {shape}")
    for key in ("action.mean", "action.std"):
        if post_shapes.get(key) != required_shapes[key]:
            errors.append(f"postprocessor statistic {key} has shape {post_shapes.get(key)}, expected {required_shapes[key]}")
    for key, size in (("observation.state", state_dimension), ("action", 26)):
        if len(stats.get(key, {}).get("mean", [])) != size or len(stats.get(key, {}).get("std", [])) != size:
            errors.append(f"dataset normalization statistics for {key} do not have length {size}")
    statistic_names = [
        "observation.state.mean", "observation.state.std", "action.mean", "action.std"
    ]
    pre_values = _safetensor_f32(pre_state, statistic_names)
    post_values = _safetensor_f32(post_state, ["action.mean", "action.std"])
    for tensor_name in statistic_names:
        feature_name, statistic = tensor_name.rsplit(".", 1)
        expected = np.asarray(stats.get(feature_name, {}).get(statistic, []), dtype=np.float32)
        if pre_values[tensor_name].shape != expected.shape or not np.allclose(
            pre_values[tensor_name], expected, rtol=0.0, atol=1e-6
        ):
            errors.append(f"preprocessor {tensor_name} values differ from dataset stats")
    for tensor_name, values in post_values.items():
        if not np.array_equal(values, pre_values[tensor_name]):
            errors.append(f"pre/postprocessor {tensor_name} values differ")

    pre_names = [step.get("registry_name") for step in pre.get("steps", [])]
    post_names = [step.get("registry_name") for step in post.get("steps", [])]
    if pre_names != list(CONTRACT["act"]["preprocessor_chain"]):
        errors.append(f"checkpoint preprocessor chain differs from contract: {pre_names}")
    if post_names != list(CONTRACT["act"]["postprocessor_chain"]):
        errors.append(f"checkpoint postprocessor chain differs from contract: {post_names}")

    model_path = pretrained_dir / "model.safetensors"
    model_shapes = _safetensor_shapes(model_path)
    if model_shapes.get("model.encoder_robot_state_input_proj.weight", [None, None])[-1] != state_dimension:
        errors.append(f"model tensor state projection is not {state_dimension}-dimensional")
    if model_shapes.get("model.action_head.bias") != [26]:
        errors.append("model tensor action head is not 26-dimensional")

    if errors:
        raise ContractError("ACT compatibility validation failed:\n- " + "\n- ".join(errors))
    return {
        "status": "passed",
        "dataset_fps": dataset_fps,
        "state_dimension": state_dimension,
        "action_dimension": 26,
        "cameras": expected_policy_cameras,
        "schema_name": (
            "keycollect_act_rgbd_tactile_v1"
            if has_tactile
            else CONTRACT["act"]["schema_name_rgbd" if has_depth else "schema_name_rgb"]
        ),
        "chunk_size": int(policy.get("chunk_size", 0)),
        "n_action_steps": int(policy.get("n_action_steps", 0)),
        "checkpoint_sha256": sha256_file(model_path),
        "preprocessor_sha256": sha256_file(pre_state),
        "postprocessor_sha256": sha256_file(post_state),
    }
