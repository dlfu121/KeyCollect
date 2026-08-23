#!/usr/bin/env python
"""Build a dog base with two RM65 arm assemblies for MuJoCo."""

from __future__ import annotations

import argparse
import copy
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


def append_world_extras(robot: ET.Element) -> None:
    ET.SubElement(robot, "link", {"name": "world"})


def append_scene_extras(robot: ET.Element) -> None:
    link = ET.SubElement(robot, "link", {"name": "floor_link"})
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "box", {"size": "3.0 3.0 0.02"})

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
        ("camera_front", "1.5 0 0.9", "0 1.0472 1.5708"),
        ("camera_side", "0 1.5 0.9", "1.0472 0 3.1416"),
        ("camera_top", "0.2 0 1.8", "0 0 0"),
    ):
        ET.SubElement(robot, "link", {"name": f"{name}_link"})
        cam_joint = ET.SubElement(robot, "joint", {"name": f"world_to_{name}", "type": "fixed"})
        cam_joint.extend(
            [
                ET.Element("origin", {"xyz": xyz, "rpy": rpy}),
                ET.Element("parent", {"link": "world"}),
                ET.Element("child", {"link": f"{name}_link"}),
            ]
        )


def append_actuators(robot: ET.Element) -> None:
    for node in list(robot.findall("transmission")):
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


def prefix_urdf_tree(root: ET.Element, prefix: str) -> ET.Element:
    cloned = copy.deepcopy(root)
    name_map: dict[str, str] = {}

    for link in cloned.findall("link"):
        name = link.get("name")
        if not name:
            continue
        name_map[name] = f"{prefix}{name}"
        link.set("name", name_map[name])

    for joint in cloned.findall("joint"):
        name = joint.get("name")
        if name:
            joint.set("name", f"{prefix}{name}")
        for node in joint.findall("parent"):
            link = node.get("link")
            if link and link in name_map:
                node.set("link", name_map[link])
        for node in joint.findall("child"):
            link = node.get("link")
            if link and link in name_map:
                node.set("link", name_map[link])

    for transmission in cloned.findall("transmission"):
        name = transmission.get("name")
        if name:
            transmission.set("name", f"{prefix}{name}")
        joint = transmission.find("joint")
        if joint is not None:
            joint_name = joint.get("name")
            if joint_name:
                joint.set("name", f"{prefix}{joint_name}")
        actuator = transmission.find("actuator")
        if actuator is not None:
            actuator_name = actuator.get("name")
            if actuator_name:
                actuator.set("name", f"{prefix}{actuator_name}")

    for node in cloned.findall("mujoco"):
        cloned.remove(node)

    return cloned


def append_dog_base(robot: ET.Element) -> None:
    dog_link = ET.SubElement(robot, "link", {"name": "dog_base_link"})
    inertial = ET.SubElement(dog_link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0.12", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "20"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "0.8",
            "ixy": "0",
            "ixz": "0",
            "iyy": "1.0",
            "iyz": "0",
            "izz": "0.9",
        },
    )

    for idx, (xyz, rpy) in enumerate(
        (
            ("0 0 0.12", "0 0 0"),
            ("0.12 0.07 0.06", "0 0 0"),
            ("0.12 -0.07 0.06", "0 0 0"),
            ("-0.14 0.07 0.05", "0 0 0"),
            ("-0.14 -0.07 0.05", "0 0 0"),
            ("0 0 0.22", "0 0 0"),
        )
    ):
        geom = ET.SubElement(dog_link, "visual")
        ET.SubElement(geom, "origin", {"xyz": xyz, "rpy": rpy})
        geometry = ET.SubElement(geom, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": f"../dog/dog_visual_{idx}.STL"})

    base_joint = ET.SubElement(robot, "joint", {"name": "world_to_dog", "type": "fixed"})
    base_joint.extend(
        [
            ET.Element("origin", {"xyz": "0 0 0.18", "rpy": "0 0 0"}),
            ET.Element("parent", {"link": "world"}),
            ET.Element("child", {"link": "dog_base_link"}),
        ]
    )


def mount_arm(robot: ET.Element, arm_root: ET.Element, prefix: str, mount_xyz: str, mount_rpy: str) -> str:
    cloned = prefix_urdf_tree(arm_root, prefix)
    for node in list(cloned):
        robot.append(node)

    base_link = f"{prefix}base_link"
    mount_joint = ET.SubElement(robot, "joint", {"name": f"{prefix}base_mount", "type": "fixed"})
    mount_joint.extend(
        [
            ET.Element("origin", {"xyz": mount_xyz, "rpy": mount_rpy}),
            ET.Element("parent", {"link": "dog_base_link"}),
            ET.Element("child", {"link": base_link}),
        ]
    )

    ee_link = ET.SubElement(robot, "link", {"name": f"{prefix}ee_site_link"})
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

    ee_joint = ET.SubElement(robot, "joint", {"name": f"{prefix}ee_site", "type": "fixed"})
    ee_joint.extend(
        [
            ET.Element("origin", {"xyz": "0 0 0.20", "rpy": "0 0 0"}),
            ET.Element("parent", {"link": f"{prefix}link_6"}),
            ET.Element("child", {"link": f"{prefix}ee_site_link"}),
        ]
    )

    return f"{prefix}ee_site"


def build_scene(args: argparse.Namespace) -> None:
    arm_tree = ET.parse(args.arm)
    arm_root = arm_tree.getroot()
    rewrite_mesh_paths(arm_root, "arm")

    robot = ET.Element("robot", {"name": "dog_dual_arm"})
    add_mujoco_defaults(robot)
    append_world_extras(robot)
    append_dog_base(robot)

    left_ee = mount_arm(robot, arm_root, "left_", "-0.15 0.24 0.38", "0 0 1.5708")
    right_ee = mount_arm(robot, arm_root, "right_", "-0.15 -0.24 0.38", "0 0 -1.5708")

    append_scene_extras(robot)
    append_actuators(robot)

    # Keep the end-effector sites easy to find in the final scene.
    robot.set("name", "dog_dual_arm")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)

    print(f"Wrote {args.output}")
    print(f"End-effector sites: {left_ee}, {right_ee}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, default=ROOT / "assets" / "arm" / "RM65-6F.urdf")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets" / "scene" / "dog_dual_arm_scene.urdf",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_scene(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
