"""Deterministic fixed-case generation and success tracking for ACT rollouts."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import yaml


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError(f"Unsupported evaluation protocol: {path}")
    return protocol


def generate_cases(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    positions = protocol["position_sets_xy_m"]
    center = positions["center"]
    cases: list[dict[str, Any]] = []
    combinations = itertools.product(
        positions.items(),
        protocol["orientation_deg_from_world_y"],
        protocol["visibility"].items(),
        protocol["distribution"].items(),
    )
    for index, ((position_name, xy), angle_deg, (visibility_name, visibility), (distribution_name, distribution)) in enumerate(combinations):
        scale = float(distribution["radial_scale_about_center"])
        scaled_xy = [center[axis] + (xy[axis] - center[axis]) * scale for axis in range(2)]
        yaw = math.pi / 2.0 + math.radians(float(angle_deg))
        object_pose = [
            float(scaled_xy[0]), float(scaled_xy[1]), float(protocol["object_root_z_m"]),
            0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0),
        ]
        cases.append({
            "schema_version": 1,
            "case_id": f"case_{index:03d}",
            "position_set": position_name,
            "orientation_deg_from_world_y": float(angle_deg),
            "visibility": visibility_name,
            "distribution": distribution_name,
            "random_seed": int(protocol["random_seed_base"]) + index,
            "object_body": protocol["object_body"],
            "object_pose_xyzw": object_pose,
            "occluder": dict(visibility),
        })
    return cases


class LiftSuccessTracker:
    def __init__(self, initial_z: float, minimum_lift_m: float, minimum_hold_s: float):
        self.initial_z = initial_z
        self.minimum_lift_m = minimum_lift_m
        self.minimum_hold_s = minimum_hold_s
        self.above_since: float | None = None
        self.ever_above = False
        self.slipped_after_threshold = False

    def update(self, simulation_time_s: float, object_z: float) -> None:
        above = object_z - self.initial_z >= self.minimum_lift_m
        if above:
            self.ever_above = True
            if self.above_since is None:
                self.above_since = simulation_time_s
        elif self.above_since is not None:
            self.slipped_after_threshold = True
            self.above_since = None

    def successful(self, simulation_time_s: float, object_z: float) -> bool:
        self.update(simulation_time_s, object_z)
        return (
            self.above_since is not None
            and simulation_time_s - self.above_since >= self.minimum_hold_s
            and object_z - self.initial_z >= self.minimum_lift_m
        )
