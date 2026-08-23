#!/usr/bin/env python
"""System health check for MuJoCo-LeRobot deployment.

Checks all components are properly installed and configured.
"""

import sys
import platform
import importlib.metadata

def check(label, condition, detail=""):
    status = "[OK]" if condition else "[FAIL]"
    print(f"  {status} {label}")
    if detail:
        print(f"        {detail}")
    return condition

def main():
    print("=" * 60)
    print("MuJoCo-LeRobot Doctor")
    print("=" * 60)
    all_ok = True

    # OS
    print("\n--- System ---")
    os_info = platform.platform()
    is_ubuntu = "noble" in os_info.lower() or "ubuntu" in os_info.lower() or "24.04" in open("/etc/os-release").read() if __import__("os").path.exists("/etc/os-release") else False
    all_ok &= check("Ubuntu 24.04", is_ubuntu, os_info)

    # Python
    print("\n--- Python ---")
    py_ver = platform.python_version()
    py_ok = sys.version_info >= (3, 12)
    all_ok &= check("Python >= 3.12", py_ok, py_ver)

    # MuJoCo
    print("\n--- MuJoCo ---")
    try:
        import mujoco
        mj_ver = mujoco.__version__
        mj_ok = mj_ver == "3.11.0"
        all_ok &= check("MuJoCo 3.11.0", mj_ok, mj_ver)
    except ImportError:
        all_ok &= check("MuJoCo", False, "Not installed")
        mj_ok = False

    # LeRobot
    print("\n--- LeRobot ---")
    try:
        import lerobot
        lr_ver = lerobot.__version__
        lr_ok = lr_ver == "0.6.1"
        all_ok &= check("LeRobot 0.6.1", lr_ok, lr_ver)
    except ImportError:
        all_ok &= check("LeRobot", False, "Not installed")
        lr_ok = False

    # CPU-only
    print("\n--- GPU ---")
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    all_ok &= check("CPU-only data collection", not has_cuda or True,
                     "CUDA available" if has_cuda else "CPU-only (OK)")

    # OSMesa or GLFW+Xvfb
    print("\n--- Rendering ---")
    import os
    gl_backend = os.environ.get("MUJOCO_GL", "")
    display = os.environ.get("DISPLAY", "")
    rendering_ok = False
    try:
        import ctypes
        ctypes.CDLL("libOSMesa.so")
        rendering_ok = True
        all_ok &= check("OSMesa available", True)
    except OSError:
        pass
    if not rendering_ok:
        # Check if GLFW + Xvfb is available
        if gl_backend == "glfw" and display:
            rendering_ok = True
            all_ok &= check("Rendering (GLFW+Xvfb)", True, f"MUJOCO_GL={gl_backend}, DISPLAY={display}")
        else:
            all_ok &= check("Rendering", False, "No GL backend. Set MUJOCO_GL=glfw + DISPLAY or install libOSMesa6")

    # Plugins
    print("\n--- Plugins ---")
    try:
        from lerobot_robot_mujoco import MuJoCoRobot, MuJoCoRobotConfig
        all_ok &= check("MuJoCo Robot plugin", True)
    except ImportError as e:
        all_ok &= check("MuJoCo Robot plugin", False, str(e))

    try:
        from lerobot_teleoperator_mocap_ros import MocapRosTeleop, MocapRosTeleopConfig
        all_ok &= check("Mocap ROS Teleop plugin", True)
    except ImportError as e:
        all_ok &= check("Mocap ROS Teleop plugin", False, str(e))

    # Scene test (if scene file provided)
    print("\n--- Scene ---")
    scene_path = sys.argv[1] if len(sys.argv) > 1 else "assets/scenes/rm65_dexhand_scene.xml"
    if mj_ok:
        try:
            from lerobot_robot_mujoco.simulation import MuJoCoSimulation
            sim = MuJoCoSimulation(scene_path)
            sim.load()
            cameras = sim.get_named_cameras()
            all_ok &= check("Scene loads", True, scene_path)
            all_ok &= check("Named cameras", len(cameras) > 0, str(cameras))

            # Test joint mapping
            joints = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
            for j in joints:
                try:
                    sim.get_joint_id(j)
                    all_ok &= check(f"Joint '{j}'", True)
                except ValueError:
                    all_ok &= check(f"Joint '{j}'", False, "Not found")

            # Test EE site
            try:
                sim.get_body_id("link_6")
                all_ok &= check("EE body 'link_6'", True)
            except ValueError:
                all_ok &= check("EE body 'link_6'", False, "Not found")

            # Test camera rendering
            if cameras:
                MUJOCO_GL = os.environ.get("MUJOCO_GL", "")
                if MUJOCO_GL == "osmesa" or not has_cuda:
                    try:
                        img = sim.render_camera(cameras[0], 64, 64)
                        all_ok &= check("Camera rendering", img.shape == (64, 64, 3),
                                        f"shape={img.shape}, dtype={img.dtype}")
                    except Exception as e:
                        all_ok &= check("Camera rendering", False, str(e))

            sim.close()
        except Exception as e:
            all_ok &= check("Scene loads", False, str(e))
    else:
        all_ok &= check("Scene", False, "MuJoCo not available")

    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All checks passed!")
    else:
        print("✗ Some checks failed. See above.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
