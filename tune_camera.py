#!/usr/bin/env python3
"""
交互式相机安装位置调整工具

功能：
- 加载任意 MuJoCo 场景 XML，自动发现其中所有 <camera>（含 <include> 引用的文件）
- 在 MuJoCo viewer 中查看全局场景（可鼠标旋转）
- matplotlib 窗口实时显示当前所选相机的画面 + 6 个滑块（pos x/y/z, rot x/y/z）
- 键盘微调 + 鼠标拖动滑块，实时修改相机的 pos / quat
- 按 S 键双击确认后，直接写回相机定义所在的 XML 文件（写前自动备份 .bak）

用法：
    uv run python tune_camera.py
    uv run python tune_camera.py --scene model/xxx/scene.xml
    uv run python tune_camera.py --scene scene.xml --camera top_camera
    uv run python tune_camera.py --scene scene.xml --pos-range 0.3 \
        --home-q 0,-1.57,1.57,1.57,-1.57,0

说明：
    - 每个相机的 pos 滑块范围以启动时的初始值为中心（±pos-range），
      rot 滑块范围固定为 ±rot-range（度）
    - 相机看向自身 -Z 轴（画面上的 R/G/B 轴线即相机 X/Y/Z 轴，实时投影）
    - 3D 窗口实时显示世界坐标系 + 当前相机坐标系（viewer user_scn 注入）
    - model.cam_pos 存的是 XML 中的父 body 局部坐标，写回 XML 的值即滑块数值
    - 场景冻结在初始 qpos（不跑 mj_step），预览稳定不漂
    - 调参期间请勿在其他窗口打字（pynput 是全局键盘监听），Esc 退出
"""

import argparse
import os
import re
import shutil
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".matplotlib-cache"))

import cv2
import glfw
import matplotlib
matplotlib.rcParams["keymap.save"] = []
matplotlib.rcParams["keymap.quit"] = []

if os.environ.get("MPLBACKEND") is None:
    for backend in ("QtAgg", "TkAgg"):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import mujoco
import mujoco.viewer  # 子模块需显式导入
import numpy as np

def euler2quat(roll, pitch, yaw):
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_euler2Quat(quat, np.array([roll, pitch, yaw], dtype=np.float64), "XYZ")
    return quat


def quat2euler(quat):
    quat = np.asarray(quat, dtype=np.float64)
    mat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat)
    r = mat.reshape(3, 3)
    sy = np.hypot(r[0, 0], r[1, 0])
    if sy > 1e-9:
        roll = np.arctan2(r[2, 1], r[2, 2])
        pitch = np.arctan2(-r[2, 0], sy)
        yaw = np.arctan2(r[1, 0], r[0, 0])
    else:
        roll = np.arctan2(-r[1, 2], r[1, 1])
        pitch = np.arctan2(-r[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


DEFAULT_SCENE = "assets/scenes/rm65_dexhand_scene.xml"
DEFAULT_STATE = "assets/scenes/current_state.npz"
DEFAULT_POS_RANGE = 0.5      # pos 滑块半宽（m）
DEFAULT_ROT_RANGE = 180.0    # rot 滑块半宽（度）
POS_STEP = 0.001             # pos 键盘基础步进 1 mm
ROT_STEP_DEG = 1.0           # rot 键盘基础步进 1 度
SAVE_CONFIRM_SECS = 3.0      # S 双击确认窗口（秒）
IMG_W, IMG_H = 640, 480      # 离屏渲染尺寸

# 滑块行定义: (标签, 类型, 是否角度)  —— 范围运行时按相机动态确定
ROWS = [
    ("X",  "pos", False),
    ("Y",  "pos", False),
    ("Z",  "pos", False),
    ("RX", "rot", True),
    ("RY", "rot", True),
    ("RZ", "rot", True),
]

MAX_KEY_CAMS = 9             # 数字键 1..9 最多直接切换的相机数


def wrap_pi(angle):
    """把角度 wrap 到 [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def collect_included_files(root_path):
    """递归收集场景文件及其所有 <include> 引用的文件（去重，按引用顺序）"""
    result = []
    seen = set()

    def walk(path):
        path = os.path.abspath(os.path.normpath(path))
        if path in seen or not os.path.isfile(path):
            return
        seen.add(path)
        result.append(path)
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            return
        base = os.path.dirname(path)
        for m in re.finditer(r'<include\s+file="([^"]+)"', text):
            walk(os.path.join(base, m.group(1)))

    walk(root_path)
    return result


def find_camera_files(scene_path, cam_names):
    """
    在场景及其 include 文件中定位每个相机标签所在的文件。
    返回 {cam_name: xml_path 或 None（未找到，保存时禁用该相机）}
    """
    files = collect_included_files(scene_path)
    result = {}
    for name in cam_names:
        pat = re.compile(r'<camera\b[^>]*?\bname="%s"' % re.escape(name), re.DOTALL)
        found = None
        for path in files:
            with open(path, encoding="utf-8") as f:
                if pat.search(f.read()):
                    found = path
                    break
        result[name] = found
    return result


class KeyboardController:
    """pynput 后台线程键盘监听"""

    def __init__(self, num_cams):
        self._listener = None
        self._running = False
        self.key_states = {}
        self.num_cams = num_cams

    def start(self):
        try:
            from pynput import keyboard
        except Exception as exc:
            print(f"[Warn] pynput 不可用，键盘快捷键关闭；仍可用滑块和相机选择微调: {exc}")
            self.key_states = {}
            self._running = True
            return

        self.key_states = {
            # pos 步进
            "a": False, "d": False,   # X -/+
            "q": False, "e": False,   # Y -/+
            "r": False, "f": False,   # Z -/+
            # rot 步进
            "t": False, "g": False,   # roll  -/+
            "y": False, "h": False,   # pitch -/+
            "u": False, "j": False,   # yaw   -/+
            # 修饰键
            keyboard.Key.shift_l: False,
            keyboard.Key.ctrl_l: False,
            # 功能
            "s": False,               # 保存（双击确认）
            "n": False,               # 取消保存
            keyboard.Key.tab: False,  # 循环切换相机
            keyboard.Key.home: False, # 复位当前相机
        }
        # 数字键 1..N 直接切换相机（N 取实际相机数与 9 的较小值）
        for i in range(min(self.num_cams, MAX_KEY_CAMS)):
            self.key_states[str(i + 1)] = False

        def on_press(key):
            try:
                if key in self.key_states:
                    self.key_states[key] = True
                if hasattr(key, "char") and key.char:
                    char = key.char.lower()
                    if char in self.key_states:
                        self.key_states[char] = True
                if key == keyboard.Key.esc:
                    self._running = False
            except Exception:
                pass

        def on_release(key):
            try:
                if key in self.key_states:
                    self.key_states[key] = False
                if hasattr(key, "char") and key.char:
                    char = key.char.lower()
                    if char in self.key_states:
                        self.key_states[char] = False
            except Exception:
                pass

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        self._running = True

    def stop(self):
        self._running = False
        if self._listener:
            self._listener.stop()


class CameraTuner:
    def __init__(self, scene_path, start_cam=None, pos_range=DEFAULT_POS_RANGE,
                 rot_range=DEFAULT_ROT_RANGE, home_q=None, state_path=None, show_axes=True):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.scene_path = scene_path
        self.state_path = state_path
        self.pos_range = pos_range
        self.rot_range = rot_range
        self.show_axes = show_axes

        # 发现所有相机
        self.cam_names = []
        self.cam_ids = {}
        for i in range(self.model.ncam):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if not name:
                continue
            self.cam_names.append(name)
            self.cam_ids[name] = i
        if not self.cam_names:
            raise RuntimeError(f"场景 {scene_path} 中没有任何 <camera>")

        # 定位每个相机标签所在的 XML 文件（用于保存写回）
        self.cam_files = find_camera_files(scene_path, self.cam_names)

        if state_path is not None and os.path.isfile(state_path):
            self._apply_state_file(state_path)
        elif home_q is not None:
            self._apply_home_q(home_q)

        # 每个相机的状态（单一数据源）:
        #   pos(3,) 父 body 局部坐标, euler(3,) rad
        #   pos_lo/pos_hi(3,) 滑块范围（以初始 pos 为中心 ±pos_range）
        self.state = {}
        self.orig_state = {}
        for name in self.cam_names:
            cid = self.cam_ids[name]
            pos0 = self.model.cam_pos[cid].copy()
            self.state[name] = {
                "pos": pos0.copy(),
                "euler": np.asarray(quat2euler(self.model.cam_quat[cid]), dtype=float),
                "pos_lo": pos0 - pos_range,
                "pos_hi": pos0 + pos_range,
            }
            self.orig_state[name] = {
                "pos": pos0.copy(),
                "euler": self.state[name]["euler"].copy(),
            }

        # 起始相机：默认优先选挂在非 world body 上的相机（通常是"安装位置"的调参目标）
        if start_cam is not None:
            if start_cam not in self.cam_ids:
                raise RuntimeError(
                    f"相机 {start_cam} 不在场景中，可选: {', '.join(self.cam_names)}")
            self.cur_cam = start_cam
        else:
            self.cur_cam = next(
                (n for n in self.cam_names if self.model.cam_bodyid[self.cam_ids[n]] != 0),
                self.cam_names[0])

        self.pending_save = None      # 第一次按 S 的时间戳
        self.last_print_t = 0.0

        # GUI + 离屏渲染
        self.handle = None
        self.glfw_window = None
        self.scene = None
        self.context = None
        self.cam_view = None
        self._init_rendering()

        # 键盘
        self.keyboard = KeyboardController(len(self.cam_names))
        self.prev_key = {}

        # matplotlib 图
        self.fig = None
        self.im = None
        self.sliders = []
        self.info_text = None
        self.status_text = None
        self.radio = None
        self.save_button = None
        self.reset_button = None
        self._updating_radio = False
        self._frame_count = 0

    def _apply_home_q(self, home_q):
        """把 home_q（list[float]）应用到前 len(home_q) 个关节（模型顺序）"""
        n = min(len(home_q), self.model.njnt)
        for k in range(n):
            jid = k
            self.data.qpos[self.model.jnt_qposadr[jid]] = float(home_q[k])
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def _apply_state_file(self, state_path):
        state = np.load(state_path, allow_pickle=False)
        qpos = state["qpos"]
        qvel = state["qvel"] if "qvel" in state else None
        n = min(len(qpos), self.model.nq)
        self.data.qpos[:n] = qpos[:n]
        if qvel is not None:
            nv = min(len(qvel), self.model.nv)
            self.data.qvel[:nv] = qvel[:nv]
        else:
            self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        print(f"[状态] 从 {state_path} 开始调整")

    def _parent_body_name(self, name):
        """相机父 body 名（world 返回 'world'）"""
        cid = self.cam_ids[name]
        bid = self.model.cam_bodyid[cid]
        if bid == 0:
            return "world"
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"

    def _init_rendering(self):
        """先开 viewer GUI，再开离屏 GLFW context（顺序同 collect_keyboard_data.py）"""
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = 1.5
        self.handle.cam.azimuth = 140
        self.handle.cam.elevation = -30

        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.glfw_window = glfw.create_window(IMG_W, IMG_H, "Offscreen", None, None)
        glfw.make_context_current(self.glfw_window)

        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

        self.cam_view = mujoco.MjvCamera()
        self.cam_view.type = mujoco.mjtCamera.mjCAMERA_FIXED

        # viewer 用户几何场景: 每帧往里写 axes 胶囊, 3D 窗口中实时显示坐标系
        self.user_scn = getattr(self.handle, "user_scn", None)
        if self.show_axes and self.user_scn is None:
            print("[Warn] 当前 viewer 不支持 user_scn, 3D 窗口 axes 关闭 (画面 2D 叠加不受影响)")

    def _apply_state(self):
        """把当前相机 state 写入 model 并 forward"""
        cid = self.cam_ids[self.cur_cam]
        st = self.state[self.cur_cam]
        self.model.cam_pos[cid] = st["pos"]
        self.model.cam_quat[cid] = euler2quat(*st["euler"])
        mujoco.mj_forward(self.model, self.data)

    def _capture_image(self):
        """渲染当前相机画面 (H, W, 3) uint8 RGB"""
        cid = self.cam_ids[self.cur_cam]
        self.cam_view.fixedcamid = cid
        viewport = mujoco.MjrRect(0, 0, IMG_W, IMG_H)
        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(),
                               mujoco.MjvPerturb(), self.cam_view,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        rgb = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, self.context)
        # ascontiguousarray: flipud 返回非连续视图, cv2 叠加需要连续内存
        return np.ascontiguousarray(np.flipud(rgb))

    def _overlay_axes(self, img):
        """把当前相机自身坐标系的 X/Y/Z 轴 (R/G/B) 实时投影到画面上。

        针孔投影使用相机的真实 fovy（model.cam_fovy, 度）。
        相机看向 -Z，故 +Z 轴指向镜头后方，会被裁掉
        """
        cid = self.cam_ids[self.cur_cam]
        h, w = img.shape[:2]
        cpos = self.data.cam_xpos[cid].copy()
        cxmat = self.data.cam_xmat[cid].reshape(3, 3)
        fwd = -cxmat[:, 2]
        right, up = cxmat[:, 0], cxmat[:, 1]
        fovy = self.model.cam_fovy[cid]
        fy = 0.5 * h / np.tan(0.5 * np.deg2rad(fovy))
        fx = fy * w / h

        def proj(p):
            rel = p - cpos
            z = float(rel @ fwd)
            if z < 1e-3:
                return None
            return int(w / 2 + fx * float(rel @ right) / z), \
                   int(h / 2 - fy * float(rel @ up) / z)

        scale = 0.1
        for label, d, color in (("X", right, (255, 70, 70)),
                                ("Y", up, (90, 255, 90)),
                                ("Z", cxmat[:, 2], (100, 150, 255))):
            s = proj(cpos + fwd * 0.02)
            e = proj(cpos + fwd * 0.02 + d * scale)
            if s is not None and e is not None:
                cv2.line(img, s, e, color, 3, cv2.LINE_AA)
                cv2.putText(img, label, (e[0] + 4, e[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.putText(img, "R=X  G=Y  B=Z (camera frame)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

    def _basis_mat(self, z_dir):
        """构造 3x3 旋转矩阵 (row-major), 使胶囊体局部 z 轴指向 z_dir"""
        z = z_dir / np.linalg.norm(z_dir)
        ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(ref, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return np.column_stack([x, y, z])

    def _add_capsule_line(self, scene, start, end, radius, rgba):
        idx = scene.ngeom
        if idx >= scene.maxgeom:
            return
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length < 1e-9:
            return
        mid = (start + end) / 2.0
        mujoco.mjv_initGeom(
            scene.geoms[idx],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([radius, length / 2.0, 0.0]).reshape(3, 1),
            mid.reshape(3, 1),
            self._basis_mat(vec).reshape(9, 1),
            np.array(rgba, dtype=np.float32).reshape(4, 1),
        )
        scene.ngeom += 1

    def _add_box_marker(self, scene, pos, R, size, rgba):
        idx = scene.ngeom
        if idx >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(
            scene.geoms[idx],
            mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray(size, dtype=np.float64).reshape(3, 1),
            np.asarray(pos, dtype=np.float64).reshape(3, 1),
            np.asarray(R, dtype=np.float64).reshape(9, 1),
            np.array(rgba, dtype=np.float32).reshape(4, 1),
        )
        scene.ngeom += 1

    def _add_axis_frame(self, scene, origin, R, length, radius):
        """在 origin 处沿 R 的三列 (X/Y/Z) 添加三个胶囊轴 (R/G/B)"""
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        for i in range(3):
            d = R[:, i]
            self._add_capsule_line(scene, origin, origin + d * length, radius, list(colors[i]) + [1.0])

    def _add_camera_marker(self, scene, cpos, R, active):
        body_color = [1.0, 0.85, 0.12, 1.0] if active else [0.45, 0.48, 0.52, 0.75]
        line_color = [1.0, 0.85, 0.12, 1.0] if active else [0.65, 0.68, 0.72, 0.65]
        self._add_box_marker(scene, cpos, R, [0.035, 0.024, 0.018], body_color)

        forward = -R[:, 2]
        right = R[:, 0]
        up = R[:, 1]
        center = cpos + forward * 0.14
        half_w = 0.065
        half_h = 0.045
        corners = [
            center + right * half_w + up * half_h,
            center - right * half_w + up * half_h,
            center - right * half_w - up * half_h,
            center + right * half_w - up * half_h,
        ]
        for corner in corners:
            self._add_capsule_line(scene, cpos, corner, 0.002, line_color)
        for a, b in zip(corners, corners[1:] + corners[:1]):
            self._add_capsule_line(scene, a, b, 0.002, line_color)

    def _update_3d_axes(self):
        """更新 viewer user 场景中的 3D 坐标系: 世界坐标系 + 当前相机坐标系。

        须在 handle.lock() 内调用 (viewer 渲染线程可能正在读该场景)。
        """
        scn = self.user_scn
        if scn is None:
            return
        scn.ngeom = 0
        # 世界坐标系 (arm base 处)
        self._add_axis_frame(scn, np.zeros(3), np.eye(3),
                             length=0.10, radius=0.004)
        for name in self.cam_names:
            cid = self.cam_ids[name]
            cpos = self.data.cam_xpos[cid].copy()
            R = self.data.cam_xmat[cid].reshape(3, 3)
            active = name == self.cur_cam
            self._add_camera_marker(scn, cpos, R, active)
            if active:
                self._add_axis_frame(scn, cpos, R, length=0.08, radius=0.003)

    # ---------- 键盘 ----------

    def _handle_keys(self):
        ks = self.keyboard.key_states
        if not ks:
            return
        # 边缘检测（单击事件）
        edges = set()
        for k, v in ks.items():
            if v and not self.prev_key.get(k, False):
                edges.add(k)
        self.prev_key = dict(ks)

        from pynput import keyboard as kb

        # 单击：切换相机 / 复位 / 保存 / 取消
        switched = False
        for i in range(min(len(self.cam_names), MAX_KEY_CAMS)):
            if str(i + 1) in edges:
                self._switch_cam(i)
                switched = True
                break
        if not switched:
            if kb.Key.tab in edges:
                self._switch_cam((self.cam_names.index(self.cur_cam) + 1)
                                 % len(self.cam_names))
            elif kb.Key.home in edges:
                self._reset_cam()
            elif "s" in edges:
                self._press_save()
            elif "n" in edges:
                if self.pending_save is not None:
                    self.pending_save = None
                    print("[保存] 已取消")

        # 按住连续步进（Shift 微调 / Ctrl 粗调）
        mod = 1.0
        if ks.get(kb.Key.shift_l):
            mod = 0.1
        elif ks.get(kb.Key.ctrl_l):
            mod = 10.0
        self._step_pos("a", -1.0, mod)
        self._step_pos("d", +1.0, mod)
        self._step_pos("q", -1.0, mod)
        self._step_pos("e", +1.0, mod)
        self._step_pos("r", -1.0, mod)
        self._step_pos("f", +1.0, mod)
        self._step_rot("t", 0, -1.0, mod)
        self._step_rot("g", 0, +1.0, mod)
        self._step_rot("y", 1, -1.0, mod)
        self._step_rot("h", 1, +1.0, mod)
        self._step_rot("u", 2, -1.0, mod)
        self._step_rot("j", 2, +1.0, mod)

    def _step_pos(self, key, sign, mod):
        if not self.keyboard.key_states.get(key):
            return
        axis = {"a": 0, "d": 0, "q": 1, "e": 1, "r": 2, "f": 2}[key]
        st = self.state[self.cur_cam]
        st["pos"][axis] = np.clip(st["pos"][axis] + sign * POS_STEP * mod,
                                  st["pos_lo"][axis], st["pos_hi"][axis])
        self._on_changed()

    def _step_rot(self, key, axis, sign, mod):
        if not self.keyboard.key_states.get(key):
            return
        st = self.state[self.cur_cam]
        st["euler"][axis] = wrap_pi(st["euler"][axis] + sign * np.deg2rad(ROT_STEP_DEG) * mod)
        self._on_changed()

    def _switch_cam(self, idx):
        self.cur_cam = self.cam_names[idx]
        self.pending_save = None
        if self.radio is not None and not self._updating_radio:
            self._updating_radio = True
            self.radio.set_active(idx)
            self._updating_radio = False
        st = self.state[self.cur_cam]
        print(f"[切换] 当前相机: {self.cur_cam} (parent: {self._parent_body_name(self.cur_cam)})")
        print(f"  pos  = {np.array2string(st['pos'], precision=4)}")
        print(f"  euler= {np.array2string(np.degrees(st['euler']), precision=2)} (deg)")

    def _on_radio_camera(self, label):
        if self._updating_radio:
            return
        if label in self.cam_names:
            self._switch_cam(self.cam_names.index(label))
            self._refresh_slider_limits()
            self._sync_sliders_to_state()

    def _on_figure_key(self, event):
        if event.key in {"s", "S"}:
            self.save()
        elif event.key in {"home", "Home"}:
            self._reset_cam()
        elif event.key in {"tab", "Tab"}:
            self._switch_cam((self.cam_names.index(self.cur_cam) + 1) % len(self.cam_names))
            self._refresh_slider_limits()
            self._sync_sliders_to_state()

    def _reset_cam(self):
        name = self.cur_cam
        self.state[name]["pos"] = self.orig_state[name]["pos"].copy()
        self.state[name]["euler"] = self.orig_state[name]["euler"].copy()
        self.pending_save = None
        print(f"[复位] {name} 已恢复到启动时的 XML 原始值")
        self._on_changed()

    def _on_changed(self):
        now = time.time()
        if now - self.last_print_t > 0.1:
            self._print_state()
            self.last_print_t = now

    def _print_state(self):
        st = self.state[self.cur_cam]
        print(f"[{self.cur_cam}] pos={np.array2string(st['pos'], precision=4)} "
              f"quat={np.array2string(euler2quat(*st['euler']), precision=4)}")

    def _slider_limits(self, i):
        """第 i 行滑块的 (lo, hi)：pos 按当前相机动态，rot 固定"""
        st = self.state[self.cur_cam]
        if i < 3:
            return float(st["pos_lo"][i]), float(st["pos_hi"][i])
        return -self.rot_range, self.rot_range

    def _init_figure(self):
        """创建 matplotlib 窗口：相机选择 + 相机画面 + 6 个滑块"""
        plt.ion()
        self.fig = plt.figure(figsize=(9, 8))
        self.fig.canvas.manager.set_window_title("Camera Tuner")
        self.fig.canvas.mpl_connect("key_press_event", self._on_figure_key)

        ax_img = self.fig.add_axes([0.05, 0.40, 0.9, 0.49])
        ax_img.imshow(np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8))
        ax_img.axis("off")
        self.im = ax_img.images[0]

        # 数值信息（画面左上方）
        self.info_text = self.fig.text(0.05, 0.91, "", fontsize=10, family="monospace")
        self.fig.text(0.05, 0.865,
            "camera looks at local -Y  |  Keys: A/D Q/E R/F pos, T/G Y/H U/J rot, "
            "Shift=x0.1, Ctrl=x10, 1..N cam, S save, Home reset, Esc quit",
            fontsize=8, color="dimgray")

        ax_radio = self.fig.add_axes([0.05, 0.15, 0.16, 0.17])
        self.radio = mwidgets.RadioButtons(ax_radio, self.cam_names, active=self.cam_names.index(self.cur_cam))
        self.radio.on_clicked(self._on_radio_camera)
        ax_radio.set_title("Camera", fontsize=9)

        ax_save = self.fig.add_axes([0.05, 0.095, 0.16, 0.035])
        self.save_button = mwidgets.Button(ax_save, "Save XML")
        self.save_button.on_clicked(lambda event: self.save())

        ax_reset = self.fig.add_axes([0.05, 0.050, 0.16, 0.035])
        self.reset_button = mwidgets.Button(ax_reset, "Reset")
        self.reset_button.on_clicked(lambda event: self._reset_cam())

        # 6 个滑块（pos 绿色、rot 橙色）；初值取当前相机状态，避免首帧触发 on_changed
        st = self.state[self.cur_cam]
        self.sliders = []
        for i, (label, kind, is_deg) in enumerate(ROWS):
            lo, hi = self._slider_limits(i)
            val0 = np.degrees(st["euler"][i - 3]) if is_deg else st["pos"][i]
            ax = self.fig.add_axes([0.28, 0.30 - i * 0.045, 0.60, 0.030])
            s = mwidgets.Slider(
                ax, label, lo, hi, valinit=float(np.clip(val0, lo, hi)),
                valfmt="%.4f" if not is_deg else "%.1f",
                color="green" if kind == "pos" else "orange",
            )
            s.on_changed(lambda v, i=i: self._on_slider(i, v))
            self.sliders.append(s)

        # 底部状态行
        self.status_text = self.fig.text(0.05, 0.035, "", fontsize=10)

    def _refresh_slider_limits(self):
        """切换相机后更新 pos 滑块范围（pos 范围随相机初始值变化）"""
        for i in range(3):
            lo, hi = self._slider_limits(i)
            self.sliders[i].valmin = lo
            self.sliders[i].valmax = hi

    def _sync_sliders_to_state(self):
        st = self.state[self.cur_cam]
        for i, (_, _, is_deg) in enumerate(ROWS):
            val = np.degrees(st["euler"][i - 3]) if is_deg else st["pos"][i]
            lo, hi = self._slider_limits(i)
            self.sliders[i].set_val(float(np.clip(val, lo, hi)))

    def _on_slider(self, row, value):
        """滑块拖动回调：更新 state（单一数据源，主循环每帧应用到 model）"""
        _, _, is_deg = ROWS[row]
        lo, hi = self._slider_limits(row)
        st = self.state[self.cur_cam]
        if is_deg:
            axis = row - 3
            st["euler"][axis] = np.deg2rad(float(np.clip(value, lo, hi)))
        else:
            axis = row
            st["pos"][axis] = float(np.clip(value, lo, hi))
        self._on_changed()

    def _update_figure(self, img):
        """刷新画面、滑块位置与数值文本（每 2 帧刷新一次，避免拖慢主循环）"""
        self._frame_count += 1
        if self._frame_count % 2 != 0:
            return

        self.im.set_data(img)

        st = self.state[self.cur_cam]
        for i, (_, _, is_deg) in enumerate(ROWS):
            val = np.degrees(st["euler"][i - 3]) if is_deg else st["pos"][i]
            lo, hi = self._slider_limits(i)
            v = float(np.clip(val, lo, hi))
            # matplotlib 3.11 的 set_val 无条件触发 on_changed，
            # 值未变时跳过，避免主循环每帧触发回调
            if abs(self.sliders[i].val - v) > 1e-9:
                self.sliders[i].set_val(v)

        quat = euler2quat(*st["euler"])
        parent = self._parent_body_name(self.cur_cam)
        self.info_text.set_text(
            f"{self.cur_cam} [parent: {parent}]\n"
            f"pos  = {st['pos'][0]:+.4f}  {st['pos'][1]:+.4f}  {st['pos'][2]:+.4f}\n"
            f"quat = {quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}"
        )
        if self.pending_save is not None:
            self.status_text.set_text(">>> Press S again to CONFIRM SAVE (N to cancel) <<<")
            self.status_text.set_color("red")
        else:
            self.status_text.set_text("")
            self.status_text.set_color("red")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _press_save(self):
        xml_path = self.cam_files.get(self.cur_cam)
        if xml_path is None:
            print(f"[保存] 未找到 {self.cur_cam} 的定义文件，无法保存")
            return
        if self.pending_save is None:
            # 第一次：显示将写入的内容
            self.pending_save = time.time()
            st = self.state[self.cur_cam]
            quat = euler2quat(*st["euler"])
            print("=" * 50)
            print(f"[保存] 将写入 {xml_path}:")
            print(f"  <camera name=\"{self.cur_cam}\"")
            print(f"    pos=\"{st['pos'][0]:.4f} {st['pos'][1]:.4f} {st['pos'][2]:.4f}\"")
            print(f"    quat=\"{quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}\"/>")
            print(f"[保存] {SAVE_CONFIRM_SECS:.0f} 秒内再按 S 确认写入，按 N 取消")
        else:
            if time.time() - self.pending_save <= SAVE_CONFIRM_SECS:
                self.save()
            else:
                self.pending_save = None
                print("[保存] 超时，请重新按 S")

    def save(self):
        name = self.cur_cam
        xml_path = self.cam_files.get(name)
        if xml_path is None:
            print(f"[保存] 未找到 {name} 的定义文件，无法保存")
            return
        st = self.state[name]
        quat = euler2quat(*st["euler"])
        quat = quat / np.linalg.norm(quat)
        try:
            self._write_camera_to_xml(xml_path, name, st["pos"], quat)
            self.pending_save = None
            print(f"[保存] 已写入 {xml_path}（备份在 {xml_path}.bak），校验通过")
        except Exception as e:
            print(f"[保存] 失败: {e}")
            print(f"[保存] 原文件仍在（备份: {xml_path}.bak）")

    @staticmethod
    def _write_camera_to_xml(xml_path, cam_name, pos, quat):
        """
        只替换 camera 标签那一行内的 pos/quat 属性，文件其余字节不动。
        前提：camera 标签为单行自闭合。
        """
        with open(xml_path, encoding="utf-8") as f:
            text = f.read()
        pat = re.compile(r'<camera\s+name="%s"[^>]*/>' % re.escape(cam_name))
        m = pat.search(text)
        if not m:
            raise RuntimeError(
                f"在 {xml_path} 中未找到单行 <camera name=\"{cam_name}\" ... />，放弃保存")
        tag = m.group(0)
        pos_attr = f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"'
        quat_attr = f'quat="{quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}"'
        if re.search(r'\bpos="[^"]*"', tag):
            tag = re.sub(r'\bpos="[^"]*"', pos_attr, tag)
        else:
            tag = tag.replace("/>", f" {pos_attr}/>")
        tag = re.sub(r'\s+(quat|euler|xyaxes)="[^"]*"', "", tag)
        tag = tag.replace("/>", f" {quat_attr}/>")
        shutil.copy2(xml_path, xml_path + ".bak")
        new_text = text[:m.start()] + tag + text[m.end():]
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(new_text)
        # 写后验证：能重新加载
        mujoco.MjModel.from_xml_path(xml_path)

    def run(self):
        print("=" * 60)
        print("相机安装位置调整工具")
        print("=" * 60)
        print(f"场景: {self.scene_path}")
        if self.state_path and os.path.isfile(self.state_path):
            print(f"初始状态: {self.state_path}")
        elif self.state_path:
            print(f"初始状态: 未找到 {self.state_path}，使用 XML/home-q 默认姿态")
        for i, name in enumerate(self.cam_names):
            src = self.cam_files.get(name) or "(未找到，不可保存)"
            key = str(i + 1) if i < MAX_KEY_CAMS else "-"
            print(f"  [{key}] {name}  (parent: {self._parent_body_name(name)}, 定义于: {src})")
        print("-" * 60)
        print("  A/D  Q/E  R/F     : pos X/Y/Z 步进 (1mm)")
        print("  T/G  Y/H  U/J     : rot X/Y/Z 步进 (1度)")
        print("  Shift/Ctrl 按住    : 步进 x0.1 / x10")
        print("  1..N / Tab        : 切换相机")
        print("  鼠标拖动窗口滑块   : 精细调整 pos / rot")
        print("  S (双击)          : 保存并写回 XML（写前备份 .bak）")
        print("  N                 : 取消保存")
        print("  Home              : 复位当前相机到原始值")
        print("  Esc               : 退出")
        if self.show_axes:
            if self.user_scn is not None:
                print("  3D 窗口坐标系    : 开 (世界坐标系 + 当前相机坐标系, 实时)")
                print("  画面 axes 叠加   : 开 (R=X G=Y B=Z, 每帧投影到相机画面)")
            else:
                print("  画面 axes 叠加   : 开 (3D 窗口 user_scn 不可用, 仅画面叠加)")
        else:
            print("  坐标系显示       : 关 (--no-axes)")
        print("\n注意: 调参期间请勿在其他窗口打字（全局键盘监听）")
        print(f"注意: pos 是相机父 body 的局部坐标（world 即世界坐标）")
        print("=" * 60 + "\n")

        self.keyboard.start()
        self._init_figure()
        self._switch_cam(self.cam_names.index(self.cur_cam))  # 打印初始状态
        try:
            while self.keyboard._running and self.handle.is_running():
                self._handle_keys()
                self._apply_state()
                if self.show_axes:
                    with self.handle.lock():
                        self._update_3d_axes()
                img = self._capture_image()
                if self.show_axes:
                    self._overlay_axes(img)
                self._update_figure(img)
                self.handle.sync()
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n中断")
        finally:
            self.keyboard.stop()
            plt.close(self.fig)
            if self.handle:
                self.handle.close()
            if self.glfw_window:
                glfw.destroy_window(self.glfw_window)
                glfw.terminate()
            print("已退出")


def parse_args():
    ap = argparse.ArgumentParser(description="交互式 MuJoCo 相机安装位置调整工具")
    ap.add_argument("--scene", default=DEFAULT_SCENE,
                    help=f"场景 XML 路径（默认 {DEFAULT_SCENE}）")
    ap.add_argument("--camera", default=None,
                    help="启动时选中的相机名（默认优先选挂在非 world body 上的相机）")
    ap.add_argument("--pos-range", type=float, default=DEFAULT_POS_RANGE,
                    help=f"pos 滑块半宽（m，默认 {DEFAULT_POS_RANGE}）")
    ap.add_argument("--rot-range", type=float, default=DEFAULT_ROT_RANGE,
                    help=f"rot 滑块半宽（度，默认 {DEFAULT_ROT_RANGE}）")
    ap.add_argument("--home-q", default=None,
                    help="逗号分隔的关节初始值，应用到模型前 N 个关节"
                         "（如 0,-1.57,1.57,1.57,-1.57,0）")
    ap.add_argument("--state", default=DEFAULT_STATE,
                    help=f"从 teleop 保存的 qpos/qvel 状态开始调整（默认 {DEFAULT_STATE}，不存在则忽略）")
    ap.add_argument("--no-state", action="store_true",
                    help="不读取 --state，直接使用 XML/home-q 默认姿态")
    ap.add_argument("--no-axes", action="store_true",
                    help="不在相机画面上叠加相机自身坐标系 axes (R=X G=Y B=Z)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    home_q = None
    if args.home_q:
        home_q = [float(x) for x in args.home_q.split(",") if x.strip()]
    CameraTuner(
        scene_path=args.scene,
        start_cam=args.camera,
        pos_range=args.pos_range,
        rot_range=args.rot_range,
        home_q=home_q,
        state_path=None if args.no_state else args.state,
        show_axes=not args.no_axes,
    ).run()
