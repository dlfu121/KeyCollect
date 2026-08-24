#!/usr/bin/env python
"""Build a dog-mounted arm + dexterous hand scene for MuJoCo."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
# This wrist-roll reference keeps the DexHand palm facing downward at reset.
# Runtime IK still unwraps equivalent J6 angles to avoid unnecessary turns.
ARM_HOME_QPOS = np.deg2rad([0.0, -23.0, -99.0, 0.0, 27.0, 180.0])
# DEXHAND_HOME_QPOS = np.zeros(20, dtype=np.float64)
DEXHAND_HOME_QPOS = np.deg2rad([
    # 拇指：joint1_1 ～ joint1_4
    50.0, 30.0, 0.0, 0.0,

    # 食指：joint2_1 ～ joint2_4
    0.0, 0.0, 0.0, 0.0,

    # 中指：joint3_1 ～ joint3_4
    0.0, 0.0, 0.0, 0.0,

    # 无名指：joint4_1 ～ joint4_4
    0.0, 0.0, 0.0, 0.0,

    # 小指：joint5_1 ～ joint5_4
    0.0, 0.0, 0.0, 0.0,
])
ARM_POSITION_GAINS = (
    (700.0, 55.0),
    (700.0, 55.0),
    (450.0, 40.0),
    (280.0, 25.0),
    (220.0, 20.0),
    (160.0, 15.0),
)
ARM_FORCE_LIMITS = (40.0, 40.0, 24.0, 7.0, 6.0, 5.0)
DEXHAND_POSITION_GAIN = (25.0, 2.0)
DEXHAND_FORCE_LIMIT = 0.5
# The URDF joint_properties tag is not preserved by MuJoCo's URDF importer.
# Keep a small amount of physical damping/inertia on the gram-scale finger
# links so contact cannot produce an unbounded acceleration.
DEXHAND_JOINT_DAMPING = 0.10
DEXHAND_JOINT_ARMATURE = 0.0002
DEXHAND_JOINT_FRICTIONLOSS = 0.005
SCREWDRIVER_HEX_MESH = "screwdriver_hex_handle"
SCREWDRIVER_FLAT_HANDLE_QUAT = "0.683013 0.183013 0.683013 0.183013"

# Work surface layout (metres).  The robot stand top is at z=0.60 m;
# keeping the work surface slightly above it makes the objects easy to reach.
WORK_TABLE_CENTER_X = 0
WORK_TABLE_HEIGHT = 0.7
WORK_TABLE_THICKNESS = 0.04

# Low fixed camera: 8 cm in front of the RM65 base centre and 10 cm above the
# tabletop.  A wide field of view keeps the full randomized screwdriver
# envelope visible despite the deliberately low mounting height.
TABLE_CAMERA_POS = (-0.62, 0.0, WORK_TABLE_HEIGHT + 0.10)
TABLE_CAMERA_TARGET = (-0.04, -0.08, WORK_TABLE_HEIGHT + 0.02)
TABLE_CAMERA_FOVY_DEG = 100

# Keep the legacy observation name, but mount this camera at the palm.  The
# palm-centre site is (0.000175, 0.001232, 0.016614) in link_6; add 20 mm along
# local +X (the outward palm normal) so the lens is not buried in the mesh.
# Aim between the fingertips and table to observe finger closure and contact.
HAND_BACK_CAMERA_POS = (0.020175, 0.001232, 0.016614)
HAND_BACK_CAMERA_TARGET = (0.10, 0.0, 0.20)
HAND_BACK_CAMERA_UP = (-1.0, 0.0, 0.0)
HAND_BACK_CAMERA_FOVY_DEG = 100


def indent_xml(tree: ET.ElementTree, space: str = "  ") -> None:
    """Indent XML on Python 3.8 as well as newer interpreters."""
    if hasattr(ET, "indent"):
        ET.indent(tree, space=space)
        return

    def indent_element(element: ET.Element, level: int = 0) -> None:
        whitespace = "\n" + level * space
        child_whitespace = "\n" + (level + 1) * space
        if len(element):
            if not element.text or not element.text.strip():
                element.text = child_whitespace
            for child in element:
                indent_element(child, level + 1)
                if not child.tail or not child.tail.strip():
                    child.tail = child_whitespace
            element[-1].tail = whitespace
        elif level and (not element.tail or not element.tail.strip()):
            element.tail = whitespace

    indent_element(tree.getroot())


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


def add_screwdriver(
    robot: ET.Element,
    name: str,
    xyz: str,
    handle_rgba: str,
    yaw: float = 0.0,
) -> None:
    """Add a graspable screwdriver with a single floating root body.

    The handle and shaft are fixed children of the root, so they move as one
    rigid object when the free joint is grasped or randomized.
    """
    root = ET.SubElement(robot, "link", {"name": name})
    root_joint = ET.SubElement(robot, "joint", {"name": f"world_to_{name}", "type": "floating"})
    root_joint.extend(
        [
            ET.Element("origin", {"xyz": xyz, "rpy": f"0 0 {yaw:.6f}"}),
            ET.Element("parent", {"link": "world"}),
            ET.Element("child", {"link": name}),
        ]
    )
    # The handle ends at x=0.04 and the shaft starts at the same point.
    add_cylinder(robot, f"{name}_handle", name, "-0.040 0 0", 0.018, 0.160, handle_rgba, "0 1.5708 0")
    add_cylinder(robot, f"{name}_shaft", name, "0.140 0 0", 0.006, 0.200, "0.65 0.65 0.70 1", "0 1.5708 0")


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
            "strippath": "false",
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
    top_center_z = WORK_TABLE_HEIGHT - WORK_TABLE_THICKNESS / 2
    leg_height = WORK_TABLE_HEIGHT - WORK_TABLE_THICKNESS
    leg_center_z = leg_height / 2
    add_box(
        robot,
        "work_table_top",
        "world",
        f"{WORK_TABLE_CENTER_X} 0 {top_center_z:.3f}",
        "1.0 0.8 0.04",
        "0.50 0.45 0.40 1",
    )
    for name, x, y in (
        ("work_table_leg_fl", f"{WORK_TABLE_CENTER_X - 0.45:.3f}", "-0.35"),
        ("work_table_leg_fr", f"{WORK_TABLE_CENTER_X + 0.45:.3f}", "-0.35"),
        ("work_table_leg_bl", f"{WORK_TABLE_CENTER_X - 0.45:.3f}", "0.35"),
        ("work_table_leg_br", f"{WORK_TABLE_CENTER_X + 0.45:.3f}", "0.35"),
    ):
        add_box(robot, name, "world", f"{x} {y} {leg_center_z:.3f}", f"0.08 0.08 {leg_height:.3f}", "0.35 0.35 0.38 1")

    # The flat hex-handle face is 18 mm from the centre; start it directly on
    # the tabletop instead of dropping it from above and inducing a flip.
    object_z = WORK_TABLE_HEIGHT + 0.018
    add_screwdriver(robot, "screwdriver_red", f"-0.08 -0.02 {object_z:.3f}", "0.85 0.15 0.15 1", np.deg2rad(70.0))


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

    # Position servos otherwise need a persistent pose error to generate the
    # torque that holds the arm and hand against gravity. Scope compensation to
    # the robot subtree so free task objects still obey world gravity.
    robot_root = root.find(".//body[@name='link_1']")
    if robot_root is None:
        raise ValueError("Exported MJCF does not contain the RM65 root body 'link_1'.")
    for body in robot_root.iter("body"):
        body.set("gravcomp", "1")

    for joint in root.findall(".//joint"):
        if joint.get("name", "").startswith("r_f_joint"):
            joint.set("damping", f"{DEXHAND_JOINT_DAMPING:g}")
            joint.set("armature", f"{DEXHAND_JOINT_ARMATURE:g}")
            joint.set("frictionloss", f"{DEXHAND_JOINT_FRICTIONLOSS:g}")

    # Fixed URDF links are fused during import.  This is the hand-base mesh
    # centre expressed in link_6 coordinates after MuJoCo applies the STL's
    # reference-frame offset; expose it for palm-centred object randomization.
    hand_body = root.find(".//body[@name='link_6']")
    if hand_body is None:
        raise ValueError("Exported MJCF does not contain the wrist body 'link_6'.")
    ET.SubElement(
        hand_body,
        "site",
        {
            "name": "dexhand_palm_center",
            "pos": "0.000175 0.001232 0.016614",
            "size": "0.001",
            "rgba": "0 0 0 0",
        },
    )

    # MuJoCo's URDF importer has no native hexagonal primitive.  Replace the
    # visual and collision handles of the screwdriver with a convex hex mesh.
    # The extra 30-degree roll puts a complete hex face, rather than an edge,
    # against the tabletop so the screwdriver does not tip at reset.
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "mesh",
        {"name": SCREWDRIVER_HEX_MESH, "file": "screwdriver_hex_handle.obj"},
    )
    for body_name in ("screwdriver_red",):
        screwdriver = root.find(f".//body[@name='{body_name}']")
        if screwdriver is None:
            raise ValueError(f"Exported MJCF does not contain '{body_name}'.")
        handle_geoms = [
            geom
            for geom in screwdriver.findall("geom")
            if geom.get("type") == "cylinder" and geom.get("pos") == "-0.04 0 0"
        ]
        if len(handle_geoms) != 2:
            raise ValueError(f"Expected two {body_name} handle geoms, got {len(handle_geoms)}.")
        for geom in handle_geoms:
            geom.set("type", "mesh")
            geom.set("mesh", SCREWDRIVER_HEX_MESH)
            geom.set("quat", SCREWDRIVER_FLAT_HANDLE_QUAT)
            geom.attrib.pop("size", None)

    # Compliant contacts and a 1 ms physics step prevent a position-controlled
    # hand from injecting an impulsive acceleration into the rigid tabletop.
    ET.SubElement(
        root,
        "option",
        {"timestep": "0.001", "integrator": "implicitfast", "solver": "Newton", "iterations": "100"},
    )
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "geom",
        {
            "condim": "4",
            "friction": "1.5 0.01 0.001",
            "solref": "0.04 1",
            "solimp": "0.85 0.95 0.005 0.5 2",
        },
    )
    ET.SubElement(root, "statistic", {"center": "0 0 0.7", "extent": "3.2"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {"ambient": "0.35 0.35 0.35", "diffuse": "0.65 0.65 0.65", "specular": "0.2 0.2 0.2"})
    ET.SubElement(visual, "global", {"azimuth": "135", "elevation": "-25"})

    actuator = root.find("actuator")
    if actuator is not None:
        root.remove(actuator)
    actuator = ET.SubElement(root, "actuator")
    actuator.append(ET.Comment("Position servos; ctrl is a joint-angle target, never a torque command."))
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            continue
        if joint_name.startswith("joint_"):
            arm_index = int(joint_name[len("joint_") :]) - 1
            if not 0 <= arm_index < len(ARM_POSITION_GAINS):
                continue
            kp, kv = ARM_POSITION_GAINS[arm_index]
            force_lower, force_upper = -ARM_FORCE_LIMITS[arm_index], ARM_FORCE_LIMITS[arm_index]
            actuator_name = joint_name
        elif joint_name.startswith("r_f_joint"):
            kp, kv = DEXHAND_POSITION_GAIN
            force_lower, force_upper = -DEXHAND_FORCE_LIMIT, DEXHAND_FORCE_LIMIT
            actuator_name = f"act_{joint_name}"
        else:
            continue

        lower, upper = model.jnt_range[joint_id]
        ET.SubElement(
            actuator,
            "position",
            {
                "name": actuator_name,
                "joint": joint_name,
                "kp": f"{kp:g}",
                "kv": f"{kv:g}",
                "ctrlrange": f"{lower:g} {upper:g}",
                "forcerange": f"{force_lower:g} {force_upper:g}",
                "forcelimited": "true",
            },
        )

    # Give the physical tabletop collision geom a stable name.  Put it in a
    # collision group that still supports objects but ignores the DexHand, so
    # tabletop penetration never blocks grasp-data collection.
    table_center = np.array([WORK_TABLE_CENTER_X, 0.0, WORK_TABLE_HEIGHT - WORK_TABLE_THICKNESS / 2])
    table_half_size = np.array([0.5, 0.4, WORK_TABLE_THICKNESS / 2])
    for geom in root.findall(".//geom"):
        if geom.get("type") != "box" or geom.get("contype", "1") == "0":
            continue
        position = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
        size = np.fromstring(geom.get("size", ""), sep=" ")
        if position.shape == (3,) and size.shape == (3,) and np.allclose(position, table_center) and np.allclose(size, table_half_size):
            geom.set("name", "work_table_collision")
            geom.set("contype", "2")
            geom.set("conaffinity", "1")
            break

    # Hand collision geoms use type 4.  With affinity 1 they still collide with
    # default type-1 task objects, while the type-2 tabletop is ignored.  Do
    # not change the RM65 link_6 geom itself; only its fused hand-base mesh and
    # the finger descendants belong to this collision group.
    for geom in hand_body.findall("geom"):
        if geom.get("mesh") == "right_hand_base" and geom.get("contype", "1") != "0":
            geom.set("contype", "4")
            geom.set("conaffinity", "1")
    for finger_body in hand_body.findall(".//body"):
        for geom in finger_body.findall("geom"):
            if geom.get("contype", "1") != "0":
                geom.set("contype", "4")
                geom.set("conaffinity", "1")

    def look_at_xyaxes(
        pos: tuple[float, float, float],
        target: tuple[float, float, float],
        up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> str:
        forward = np.array(target, dtype=np.float64) - np.array(pos, dtype=np.float64)
        forward /= np.linalg.norm(forward)
        up_axis = np.array(up, dtype=np.float64)
        up_axis /= np.linalg.norm(up_axis)
        x_axis = np.cross(forward, up_axis)
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
                    "pos": " ".join(f"{value:.3f}" for value in TABLE_CAMERA_POS),
                    "xyaxes": look_at_xyaxes(TABLE_CAMERA_POS, TABLE_CAMERA_TARGET),
                    "fovy": str(TABLE_CAMERA_FOVY_DEG),
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
                        "pos": " ".join(f"{value:.3f}" for value in HAND_BACK_CAMERA_POS),
                        "xyaxes": look_at_xyaxes(
                            HAND_BACK_CAMERA_POS,
                            HAND_BACK_CAMERA_TARGET,
                            HAND_BACK_CAMERA_UP,
                        ),
                        "fovy": str(HAND_BACK_CAMERA_FOVY_DEG),
                    },
                ),
            )

    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    for key in list(keyframe.findall("key")):
        if key.get("name") == "home":
            keyframe.remove(key)
    configured_home = np.concatenate([ARM_HOME_QPOS, DEXHAND_HOME_QPOS])
    if model.nq < len(configured_home):
        raise ValueError(f"Expected at least {len(configured_home)} qpos values, got {model.nq}.")
    # Free-jointed scene objects add seven qpos values each.  Preserve their
    # authored table poses while replacing only the robot's home configuration.
    home_qpos = model.qpos0.copy()
    home_qpos[: len(configured_home)] = configured_home
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": "home",
            "qpos": " ".join(f"{v:.6f}" for v in home_qpos),
            # The temporary URDF model has no actuators yet; the 26 position
            # servos are appended above while exporting MJCF.  Therefore use
            # the complete robot home vector instead of the temporary model.nu.
            "ctrl": " ".join(f"{v:.6f}" for v in configured_home),
        },
    )

    indent_xml(tree)
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
    indent_xml(arm_tree)
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
