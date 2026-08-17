#!/usr/bin/env python
"""Build a dog-mounted arm + dexterous hand scene for MuJoCo."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARM_HOME_QPOS = np.array([0.0303, -1.5190, -0.0250, 0.0018, -1.4745, 0.0319], dtype=np.float64)


def add_box(
    robot: ET.Element,
    name: str,
    parent: str,
    xyz: str,
    size: str,
    rgba: str,
    rpy: str = "0 0 0",
) -> None:
    link = ET.SubElement(robot, "link", {"name": name})
    for tag in ("visual", "collision"):
        node = ET.SubElement(link, tag)
        ET.SubElement(node, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(node, "geometry")
        ET.SubElement(geometry, "box", {"size": size})
        if tag == "visual":
            material = ET.SubElement(node, "material", {"name": f"{name}_mat"})
            ET.SubElement(material, "color", {"rgba": rgba})

    joint = ET.SubElement(robot, "joint", {"name": f"{parent}_to_{name}", "type": "fixed"})
    joint.extend(
        [
            ET.Element("origin", {"xyz": xyz, "rpy": rpy}),
            ET.Element("parent", {"link": parent}),
            ET.Element("child", {"link": name}),
        ]
    )


def add_cylinder(
    robot: ET.Element,
    name: str,
    parent: str,
    xyz: str,
    radius: float,
    length: float,
    rgba: str,
    rpy: str = "0 0 0",
) -> None:
    link = ET.SubElement(robot, "link", {"name": name})
    for tag in ("visual", "collision"):
        node = ET.SubElement(link, tag)
        ET.SubElement(node, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(node, "geometry")
        ET.SubElement(geometry, "cylinder", {"radius": str(radius), "length": str(length)})
        if tag == "visual":
            material = ET.SubElement(node, "material", {"name": f"{name}_mat"})
            ET.SubElement(material, "color", {"rgba": rgba})

    joint = ET.SubElement(robot, "joint", {"name": f"{parent}_to_{name}", "type": "fixed"})
    joint.extend(
        [
            ET.Element("origin", {"xyz": xyz, "rpy": rpy}),
            ET.Element("parent", {"link": parent}),
            ET.Element("child", {"link": name}),
        ]
    )


def rewrite_mesh_paths(root: ET.Element, asset_dir: str, scale: str | None = None) -> None:
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        mesh_name = Path(filename).name
        mesh.set("filename", f"../{asset_dir}/{mesh_name}")
        if scale is not None and mesh.get("scale") is None:
            mesh.set("scale", scale)


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
    add_box(robot, "floor_link", "world", "0 0 -0.01", "12 12 0.02", "0.22 0.22 0.24 1")
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
        ("camera_front", "2.4 -2.2 1.2", "0 1.0472 0.7854"),
        ("camera_side", "0 3.0 1.4", "1.0472 0 3.1416"),
        ("camera_top", "0.3 0 3.0", "0 0 0"),
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


def append_robot_stand_table(robot: ET.Element) -> None:
    add_box(robot, "robot_stand_top", "world", "-0.7 0 0.58", "0.70 0.55 0.04", "0.42 0.42 0.45 1")
    for name, x, y in (
        ("robot_stand_leg_fl", "-0.98", "-0.22"),
        ("robot_stand_leg_fr", "-0.42", "-0.22"),
        ("robot_stand_leg_bl", "-0.98", "0.22"),
        ("robot_stand_leg_br", "-0.42", "0.22"),
    ):
        add_box(robot, name, "world", f"{x} {y} 0.29", "0.06 0.06 0.58", "0.32 0.32 0.35 1")


def append_work_table_scene(robot: ET.Element) -> None:
    add_box(robot, "work_table_top", "world", "0.18 0 0.48", "1.0 0.8 0.04", "0.50 0.45 0.40 1")
    for name, x, y in (
        ("work_table_leg_fl", "-0.27", "-0.35"),
        ("work_table_leg_fr", "0.63", "-0.35"),
        ("work_table_leg_bl", "-0.27", "0.35"),
        ("work_table_leg_br", "0.63", "0.35"),
    ):
        add_box(robot, name, "world", f"{x} {y} 0.23", "0.08 0.08 0.46", "0.35 0.35 0.38 1")

    add_cylinder(
        robot,
        "screwdriver_red_handle",
        "world",
        "-0.02 0.12 0.525",
        0.018,
        0.18,
        "0.85 0.15 0.15 1",
        "0 1.5708 0",
    )
    add_cylinder(
        robot,
        "screwdriver_red_shaft",
        "world",
        "0.16 0.12 0.525",
        0.006,
        0.16,
        "0.65 0.65 0.70 1",
        "0 1.5708 0",
    )
    add_cylinder(
        robot,
        "screwdriver_blue_handle",
        "world",
        "0.35 -0.16 0.525",
        0.018,
        0.18,
        "0.15 0.35 0.85 1",
        "0 1.5708 0.8",
    )
    add_cylinder(
        robot,
        "screwdriver_blue_shaft",
        "world",
        "0.49 -0.04 0.525",
        0.006,
        0.16,
        "0.65 0.65 0.70 1",
        "0 1.5708 0.8",
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
            ET.Element("origin", {"xyz": "0 0 0.02", "rpy": "0 0 0"}),
            ET.Element("parent", {"link": "robot_stand_top"}),
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


def export_mjcf(urdf_path: Path, mjcf_path: Path) -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    mujoco.mj_saveLastXML(str(mjcf_path), model)

    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    ET.SubElement(root, "option", {"timestep": "0.002", "integrator": "implicitfast"})
    ET.SubElement(root, "statistic", {"center": "0 0 0.7", "extent": "3.2"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {"ambient": "0.35 0.35 0.35", "diffuse": "0.65 0.65 0.65", "specular": "0.2 0.2 0.2"})
    ET.SubElement(visual, "global", {"azimuth": "135", "elevation": "-25"})

    def look_at_xyaxes(pos: tuple[float, float, float], target: tuple[float, float, float]) -> str:
        forward = np.array(target, dtype=np.float64) - np.array(pos, dtype=np.float64)
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        x_axis = np.cross(forward, world_up)
        if np.linalg.norm(x_axis) < 1e-9:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(x_axis, forward)
        y_axis /= np.linalg.norm(y_axis)
        return " ".join(f"{v:.6f}" for v in (*x_axis, *y_axis))

    worldbody = root.find("worldbody")
    if worldbody is not None:
        worldbody.insert(
            0,
            ET.Element(
                "light",
                {
                    "name": "key",
                    "pos": "2 -3 5",
                    "dir": "-0.4 0.5 -1",
                    "directional": "true",
                    "diffuse": "0.8 0.8 0.78",
                    "specular": "0.25 0.25 0.25",
                },
            ),
        )
        worldbody.insert(
            1,
            ET.Element(
                "camera",
                {
                    "name": "table_camera",
                    "pos": "-0.36 0.00 0.66",
                    "xyaxes": look_at_xyaxes((-0.36, 0.0, 0.66), (0.18, 0.0, 0.52)),
                    "fovy": "65",
                },
            ),
        )
        link_6 = root.find(".//body[@name='link_6']")
        if link_6 is not None:
            link_6.insert(
                0,
                ET.Element(
                    "camera",
                    {
                        "name": "wrist_overhead_camera",
                        "pos": "0 0 0.22",
                        "euler": "0 0 0",
                        "fovy": "70",
                    },
                ),
            )

    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    for key in list(keyframe.findall("key")):
        if key.get("name") == "home":
            keyframe.remove(key)
    home_qpos = np.zeros(model.nq, dtype=np.float64)
    home_qpos[: min(len(ARM_HOME_QPOS), model.nq)] = ARM_HOME_QPOS[: min(len(ARM_HOME_QPOS), model.nq)]
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": "home",
            "qpos": " ".join(f"{v:.6f}" for v in home_qpos),
        },
    )

    ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)


def build_scene(args: argparse.Namespace) -> None:
    arm_tree = ET.parse(args.arm)
    hand_tree = ET.parse(args.hand)
    arm_root = arm_tree.getroot()
    hand_root = hand_tree.getroot()

    arm_root.set("name", "rm65_dexhand_table_scene")
    rewrite_mesh_paths(arm_root, "arm")
    rewrite_mesh_paths(hand_root, "hand")
    add_mujoco_defaults(arm_root)
    append_world_extras(arm_root)
    append_robot_stand_table(arm_root)
    append_work_table_scene(arm_root)
    append_hand_to_arm(
        arm_root,
        hand_root,
        args.arm_mount_link,
        args.hand_root_link,
        " ".join(str(v) for v in args.hand_mount_xyz),
        " ".join(str(v) for v in args.hand_mount_rpy),
    )
    append_simulation_extras(arm_root)
    append_actuators(arm_root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    urdf_output = args.output.with_suffix(".urdf") if args.output.suffix == ".xml" else args.output
    ET.indent(arm_tree, space="  ")
    arm_tree.write(urdf_output, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {urdf_output}")

    if args.output.suffix == ".xml":
        export_mjcf(urdf_output, args.output)
        print(f"Wrote {args.output}")


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
        default=ROOT / "assets" / "scenes" / "rm65_dexhand_scene.xml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_scene(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
