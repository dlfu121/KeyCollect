#!/usr/bin/env python
"""Build a combined arm + dexterous hand URDF scene for MuJoCo."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rewrite_mesh_paths(root: ET.Element, asset_dir: str) -> None:
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        mesh_name = Path(filename).name
        mesh.set("filename", f"../{asset_dir}/{mesh_name}")


def add_mujoco_defaults(robot: ET.Element) -> None:
    for mujoco_node in robot.findall("mujoco"):
        robot.remove(mujoco_node)

    mujoco_node = ET.SubElement(robot, "mujoco")
    ET.SubElement(
        mujoco_node,
        "compiler",
        {
            "angle": "radian",
            "balanceinertia": "true",
            "discardvisual": "false",
        },
    )


def append_simulation_extras(robot: ET.Element) -> None:
    link = ET.SubElement(robot, "link", {"name": "floor_link"})
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", {"xyz": "0.5 0 0", "rpy": "0 0 0"})
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "box", {"size": "2.0 2.0 0.02"})

    floor_joint = ET.SubElement(robot, "joint", {"name": "world_to_floor", "type": "fixed"})
    floor_joint.extend(
        [
            ET.Element("origin", {"xyz": "0 0 -0.02", "rpy": "0 0 0"}),
            ET.Element("parent", {"link": "world"}),
            ET.Element("child", {"link": "floor_link"}),
        ]
    )

    light_link = ET.SubElement(robot, "link", {"name": "light_link"})
    light_joint = ET.SubElement(robot, "joint", {"name": "world_to_light", "type": "fixed"})
    light_joint.extend(
        [
            ET.Element("origin", {"xyz": "0 0 3", "rpy": "0 0 0"}),
            ET.Element("parent", {"link": "world"}),
            ET.Element("child", {"link": "light_link"}),
        ]
    )

    for name, xyz, rpy in (
        ("camera_front", "1.0 0 0.8", "0 1.0472 1.5708"),
        ("camera_side", "0 1.0 0.8", "1.0472 0 3.1416"),
        ("camera_top", "0.4 0 1.5", "0 0 0"),
    ):
        cam_link = ET.SubElement(robot, "link", {"name": f"{name}_link"})
        cam_joint = ET.SubElement(robot, "joint", {"name": f"world_to_{name}", "type": "fixed"})
        cam_joint.extend(
            [
                ET.Element("origin", {"xyz": xyz, "rpy": rpy}),
                ET.Element("parent", {"link": "world"}),
                ET.Element("child", {"link": f"{name}_link"}),
            ]
        )


def append_world_extras(robot: ET.Element) -> None:
    ET.SubElement(
        robot,
        "link",
        {"name": "world"},
    )


def append_actuators(robot: ET.Element) -> None:
    transmissions = list(robot.findall("transmission"))
    for node in transmissions:
        robot.remove(node)

    for joint in robot.findall("joint"):
        if joint.get("type") not in {"revolute", "continuous", "prismatic"}:
            continue
        name = joint.get("name")
        if not name:
            continue
        transmission = ET.SubElement(robot, "transmission", {"name": f"{name}_transmission"})
        ET.SubElement(transmission, "type").text = "transmission_interface/SimpleTransmission"
        joint_node = ET.SubElement(transmission, "joint", {"name": name})
        ET.SubElement(joint_node, "hardwareInterface").text = "PositionJointInterface"
        actuator = ET.SubElement(transmission, "actuator", {"name": f"{name}_actuator"})
        ET.SubElement(actuator, "hardwareInterface").text = "PositionJointInterface"
        ET.SubElement(actuator, "mechanicalReduction").text = "1"
    ET.SubElement(
        robot,
        "joint",
        {"name": "world_to_base", "type": "fixed"},
    ).extend(
        [
            ET.Element("origin", {"xyz": "0 0 0", "rpy": "0 0 0"}),
            ET.Element("parent", {"link": "world"}),
            ET.Element("child", {"link": "base_link"}),
        ]
    )


def append_hand_to_arm(
    arm_root: ET.Element,
    hand_root: ET.Element,
    parent_link: str,
    child_link: str,
    mount_xyz: str,
    mount_rpy: str,
) -> None:
    for node in list(hand_root):
        if node.tag == "mujoco":
            continue
        arm_root.append(node)

    mount_joint = ET.SubElement(
        arm_root,
        "joint",
        {"name": "wrist_to_hand", "type": "fixed"},
    )
    mount_joint.extend(
        [
            ET.Element("origin", {"xyz": mount_xyz, "rpy": mount_rpy}),
            ET.Element("parent", {"link": parent_link}),
            ET.Element("child", {"link": child_link}),
        ]
    )

    ee_link = ET.SubElement(arm_root, "link", {"name": "ee_site_link"})
    inertial = ET.SubElement(ee_link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.001"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "1e-9",
            "ixy": "0",
            "ixz": "0",
            "iyy": "1e-9",
            "iyz": "0",
            "izz": "1e-9",
        },
    )

    ee_joint = ET.SubElement(arm_root, "joint", {"name": "ee_site", "type": "fixed"})
    ee_joint.extend(
        [
            ET.Element("origin", {"xyz": "0 0 0.20", "rpy": "0 0 0"}),
            ET.Element("parent", {"link": child_link}),
            ET.Element("child", {"link": "ee_site_link"}),
        ]
    )


def build_scene(args: argparse.Namespace) -> None:
    arm_tree = ET.parse(args.arm)
    hand_tree = ET.parse(args.hand)
    arm_root = arm_tree.getroot()
    hand_root = hand_tree.getroot()

    arm_root.set("name", "rm65_dexhand")
    rewrite_mesh_paths(arm_root, "arm")
    rewrite_mesh_paths(hand_root, "hand")
    add_mujoco_defaults(arm_root)
    append_hand_to_arm(
        arm_root,
        hand_root,
        args.arm_mount_link,
        args.hand_root_link,
        " ".join(str(v) for v in args.hand_mount_xyz),
        " ".join(str(v) for v in args.hand_mount_rpy),
    )
    append_world_extras(arm_root)
    append_simulation_extras(arm_root)
    append_actuators(arm_root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(arm_tree, space="  ")
    arm_tree.write(args.output, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, default=ROOT / "assets" / "arm" / "RM65-6F.urdf")
    parser.add_argument(
        "--hand",
        type=Path,
        default=ROOT / "assets" / "hand" / "dexhand021_right_simplified.urdf",
    )
    parser.add_argument("--arm-mount-link", default="link_6")
    parser.add_argument("--hand-root-link", default="right_hand_base")
    parser.add_argument(
        "--hand-mount-xyz",
        nargs=3,
        type=float,
        default=[0.0, 0.0, -0.08],
        help="Fixed joint xyz from arm mount link to hand root link.",
    )
    parser.add_argument(
        "--hand-mount-rpy",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        help="Fixed joint rpy from arm mount link to hand root link.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets" / "scene" / "rm65_dexhand_scene.urdf",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_scene(args)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
