"""MuJoCo simulation wrapper.

Provides a clean abstraction over MuJoCo's MjModel/MjData for
loading, stepping, state access, and camera rendering.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

# LeRobot record imports this plugin before users usually get feedback from
# MuJoCo. Default to EGL so offscreen cameras do not silently render black via X11.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np

logger = logging.getLogger(__name__)


class MuJoCoSimulation:
    """MuJoCo simulation environment.

    Wraps MjModel/MjData with helper methods for:
    - Scene loading and validation
    - Physics stepping
    - Joint state read/write
    - End-effector pose via forward kinematics
    - Named camera rendering
    """

    def __init__(self, scene_path: str, physics_dt: float = 0.002):
        self.scene_path = Path(scene_path)
        self.physics_dt = physics_dt
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._renderer: mujoco.Renderer | None = None
        self._viewer: mujoco.viewer.Handle | None = None
        self._named_cameras: list[str] = []

    def load(self) -> None:
        """Load the MuJoCo scene from XML."""
        if not self.scene_path.exists():
            raise FileNotFoundError(f"Scene file not found: {self.scene_path}")

        logger.info("Loading MuJoCo scene: %s", self.scene_path)
        self._model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self._data = mujoco.MjData(self._model)

        # Validate timestep
        if abs(self._model.opt.timestep - self.physics_dt) > 1e-6:
            logger.warning(
                "Scene timestep (%.4f) != configured physics_dt (%.4f). Using scene value.",
                self._model.opt.timestep,
                self.physics_dt,
            )

        # Enumerate named cameras
        self._named_cameras = []
        for i in range(self._model.ncam):
            cam_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if cam_name:
                self._named_cameras.append(cam_name)
        logger.info("Found %d named cameras: %s", len(self._named_cameras), self._named_cameras)

    def reset(self) -> None:
        """Reset simulation to initial state."""
        if self._model is None or self._data is None:
            raise RuntimeError("Simulation not loaded. Call load() first.")
        home_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id >= 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, home_id)
        else:
            mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

    def step(self, n_sub_steps: int = 1) -> None:
        """Step physics n_sub_steps times."""
        if self._model is None or self._data is None:
            raise RuntimeError("Simulation not loaded.")
        for _ in range(n_sub_steps):
            mujoco.mj_step(self._model, self._data)
        self.sync_viewer()

    def forward(self) -> None:
        """Run forward dynamics (kinematics + collision)."""
        if self._model is None or self._data is None:
            raise RuntimeError("Simulation not loaded.")
        mujoco.mj_forward(self._model, self._data)

    # ── Joint Access ────────────────────────────────────────────

    def get_joint_id(self, name: str) -> int:
        """Get joint ID by name."""
        if self._model is None:
            raise RuntimeError("Simulation not loaded.")
        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found in model.")
        return jid

    def get_joint_qpos_addr(self, name: str) -> int:
        """Get qpos address for a named joint."""
        jid = self.get_joint_id(name)
        return int(self._model.jnt_qposadr[jid])

    def get_joint_qvel_addr(self, name: str) -> int:
        """Get qvel address for a named joint."""
        jid = self.get_joint_id(name)
        return int(self._model.jnt_dofadr[jid])

    def get_joint_limits(self, name: str) -> tuple[float, float]:
        """Get (lower, upper) position limits for a joint."""
        jid = self.get_joint_id(name)
        return float(self._model.jnt_range[jid, 0]), float(self._model.jnt_range[jid, 1])

    def get_joint_position(self, name: str) -> float:
        """Get current joint position."""
        if self._data is None:
            raise RuntimeError("Simulation not loaded.")
        addr = self.get_joint_qpos_addr(name)
        return float(self._data.qpos[addr])

    def get_joint_velocity(self, name: str) -> float:
        """Get current joint velocity."""
        if self._data is None:
            raise RuntimeError("Simulation not loaded.")
        addr = self.get_joint_qvel_addr(name)
        return float(self._data.qvel[addr])

    def get_joint_positions(self, names: list[str]) -> np.ndarray:
        """Get positions for multiple joints."""
        return np.array([self.get_joint_position(n) for n in names])

    def get_joint_velocities(self, names: list[str]) -> np.ndarray:
        """Get velocities for multiple joints."""
        return np.array([self.get_joint_velocity(n) for n in names])

    def set_joint_positions(self, names: list[str], targets: np.ndarray) -> None:
        """Set target positions for actuated joints (via ctrl)."""
        if self._model is None or self._data is None:
            raise RuntimeError("Simulation not loaded.")
        for name, target in zip(names, targets):
            jid = self.get_joint_id(name)
            # Find the actuator that controls this joint
            for act_id in range(self._model.nu):
                if self._model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
                    if int(self._model.actuator_trnid[act_id, 0]) == jid:
                        self._data.ctrl[act_id] = target
                        break

    def set_joint_qpos(self, names: list[str], targets: np.ndarray) -> None:
        """Directly set joint positions in qpos (for IK solving)."""
        if self._model is None or self._data is None:
            raise RuntimeError("Simulation not loaded.")
        for name, target in zip(names, targets):
            addr = self.get_joint_qpos_addr(name)
            self._data.qpos[addr] = target

    def clip_to_joint_limits(self, names: list[str], targets: np.ndarray, margin: float = 0.01) -> np.ndarray:
        """Clip targets to joint limits with safety margin."""
        clipped = targets.copy()
        for i, name in enumerate(names):
            lo, hi = self.get_joint_limits(name)
            if lo < hi:  # finite limits
                clipped[i] = np.clip(clipped[i], lo + margin, hi - margin)
        return clipped

    # ── End Effector ────────────────────────────────────────────

    def get_site_id(self, name: str) -> int:
        """Get site ID by name."""
        if self._model is None:
            raise RuntimeError("Simulation not loaded.")
        sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"Site '{name}' not found in model.")
        return sid

    def get_body_id(self, name: str) -> int:
        """Get body ID by name."""
        if self._model is None:
            raise RuntimeError("Simulation not loaded.")
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in model.")
        return bid

    def get_ee_pose(self, site_name: str = "ee_site") -> np.ndarray:
        """Get end-effector pose as [x, y, z, qx, qy, qz, qw]."""
        if self._data is None:
            raise RuntimeError("Simulation not loaded.")
        sid = self.get_site_id(site_name)
        pos = self._data.site_xpos[sid].copy()
        rot_mat = self._data.site_xmat[sid].reshape(3, 3).copy()
        # Convert rotation matrix to quaternion
        quat = np.zeros(4, dtype=np.float64)
        mat_flat = rot_mat.flatten().astype(np.float64)
        mujoco.mju_mat2Quat(quat, mat_flat)
        return np.concatenate([pos, quat])

    def get_body_pose(self, body_name: str) -> np.ndarray:
        """Get body pose as [x, y, z, qx, qy, qz, qw]."""
        if self._data is None:
            raise RuntimeError("Simulation not loaded.")
        bid = self.get_body_id(body_name)
        pos = self._data.xpos[bid].copy()
        rot_mat = self._data.xmat[bid].reshape(3, 3).copy()
        quat = np.zeros(4, dtype=np.float64)
        mat_flat = rot_mat.flatten().astype(np.float64)
        mujoco.mju_mat2Quat(quat, mat_flat)
        return np.concatenate([pos, quat])

    def get_site_jacobian(self, site_name: str = "ee_site", joint_names: list[str] | None = None) -> np.ndarray:
        """Compute 6×N Jacobian for the given site w.r.t. specified joints.

        Returns:
            Jacobian array of shape (6, N) where N = len(joint_names).
            First 3 rows = translational, last 3 rows = rotational.
        """
        if self._model is None or self._data is None:
            raise RuntimeError("Simulation not loaded.")

        sid = self.get_site_id(site_name)
        n_dof = self._model.nv

        # Full Jacobian (6 × n_dof)
        jacp = np.zeros((3, n_dof))
        jacr = np.zeros((3, n_dof))
        mujoco.mj_jacSite(self._model, self._data, jacp, jacr, sid)
        full_jac = np.vstack([jacp, jacr])  # (6, n_dof)

        if joint_names is None:
            return full_jac

        # Select columns for specified joints
        col_indices = []
        for name in joint_names:
            jid = self.get_joint_id(name)
            dof_addr = int(self._model.jnt_dofadr[jid])
            col_indices.append(dof_addr)

        return full_jac[:, col_indices]

    # ── Camera Rendering ────────────────────────────────────────

    def get_named_cameras(self) -> list[str]:
        """Return list of named cameras in the scene."""
        return self._named_cameras.copy()

    def validate_camera(self, name: str) -> None:
        """Raise if camera name is not in the scene."""
        if name not in self._named_cameras:
            raise ValueError(
                f"Camera '{name}' not found in scene. "
                f"Available cameras: {self._named_cameras}"
            )

    def render_camera(self, camera_name: str, width: int = 480, height: int = 480) -> np.ndarray:
        """Render a named camera to RGB array.

        Args:
            camera_name: Name of the camera in the scene.
            width: Image width.
            height: Image height.

        Returns:
            RGB image as uint8 ndarray of shape (H, W, 3).
        """
        if self._model is None or self._data is None:
            raise RuntimeError("Simulation not loaded.")

        self.validate_camera(camera_name)

        if self._renderer is None or self._renderer.height != height or self._renderer.width != width:
            self._renderer = mujoco.Renderer(self._model, height=height, width=width)

        self._renderer.update_scene(self._data, camera=camera_name)
        image = self._renderer.render()
        return image.copy()

    def render_cameras(
        self, camera_configs: dict[str, tuple[int, int]]
    ) -> dict[str, np.ndarray]:
        """Render multiple cameras.

        Args:
            camera_configs: Dict of camera_name -> (width, height).

        Returns:
            Dict of camera_name -> RGB ndarray.
        """
        images = {}
        for cam_name, (w, h) in camera_configs.items():
            images[cam_name] = self.render_camera(cam_name, width=w, height=h)
        return images

    # ── On-screen Rendering ────────────────────────────────────

    def launch_viewer(
        self,
        show_left_ui: bool = True,
        show_right_ui: bool = True,
        key_callback: Callable[[int], None] | None = None,
    ) -> None:
        """Open an interactive MuJoCo viewer window for this simulation."""
        if self._model is None or self._data is None:
            raise RuntimeError("Simulation not loaded.")
        if self._viewer is not None and self._viewer.is_running():
            return

        self._viewer = mujoco.viewer.launch_passive(
            self._model,
            self._data,
            key_callback=key_callback,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
        )
        if hasattr(self._viewer, "cam"):
            self._viewer.cam.lookat[:] = np.array([-0.15, 0.0, 0.8])
            self._viewer.cam.distance = 2.4
            self._viewer.cam.azimuth = 135.0
            self._viewer.cam.elevation = -25.0
        self.sync_viewer()

    def sync_viewer(self) -> bool:
        """Refresh the viewer if it is open.

        Returns:
            True while the viewer exists and is still running.
        """
        if self._viewer is None:
            return False
        if not self._viewer.is_running():
            self.close_viewer()
            return False
        self._viewer.sync()
        return True

    def close_viewer(self) -> None:
        """Close the interactive viewer window if it is open."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ── Simulation Time ─────────────────────────────────────────

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        if self._data is None:
            raise RuntimeError("Simulation not loaded.")
        return float(self._data.time)

    @property
    def model(self) -> mujoco.MjModel:
        if self._model is None:
            raise RuntimeError("Simulation not loaded.")
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        if self._data is None:
            raise RuntimeError("Simulation not loaded.")
        return self._data

    # ── Cleanup ─────────────────────────────────────────────────

    def close(self) -> None:
        """Release resources."""
        self.close_viewer()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._data = None
        self._model = None
