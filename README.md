# KeyCollect MuJoCo 遥操作场景

基于 MuJoCo 和 LeRobot 的机器人仿真遥操作项目。当前默认场景是：一张 60cm 高的小桌子上安装 RM65 机械臂和右手机械手，前方紧贴一张 50cm 高的工作桌，桌面上放两把螺丝刀；场景内包含两个可微调摄像头。

## 系统要求

| 组件 | 版本/说明 |
|---|---|
| OS | Ubuntu 24.04 / WSL Ubuntu 24.04 |
| Python | 3.12 |
| MuJoCo | 3.11.0 |
| LeRobot | 0.6.1 |
| 显示环境 | 需要可用桌面环境 |

## 安装

WSL 首次安装 Ubuntu 24.04：

```bash
wsl --install -d Ubuntu-24.04
wsl --set-default Ubuntu-24.04
```

安装系统依赖：

```bash
sudo apt update
sudo apt install -y \
  python3.12-venv \
  python3-pip \
  libgl1-mesa-dev \
  libglfw3 \
  libglfw3-dev \
  git
```

创建虚拟环境并安装 Python 包：

```bash
cd /path/to/KeyCollect
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install matplotlib mujoco==3.11.0 lerobot==0.6.1 pynput opencv-python-headless
python -m pip install -e ./lerobot_robot_mujoco
python -m pip install -e ./lerobot_teleoperator_keyboard_mouse
```

`tune_camera.py` 需要 matplotlib 的 GUI 后端。如果没有 Tk/Qt，需要额外安装一个，例如：

```bash
.venv/bin/python -m pip install PyQt6
```

## 生成当前场景

运行：

```bash
.venv/bin/python scripts/build_hardware_scene.py
```

默认读取：

```text
assets/arm/RM65-6F.urdf
assets/hand/dexhand021_right_simplified.urdf
```

并生成：

```text
assets/scenes/rm65_dexhand_scene.urdf
assets/scenes/rm65_dexhand_scene.xml
```

推荐运行和查看 `.xml` 文件。它包含 MuJoCo 原生相机、灯光和默认 `home` 姿态。

当前场景布局：

| 物体 | 说明 |
|---|---|
| 机械臂小桌 | 上表面高度 `0.60m` |
| 螺丝刀工作桌 | 上表面高度 `0.50m` |
| 两张桌子间距 | 约 `0.03m` |
| 螺丝刀 | 固定在工作桌桌面上 |
| 机械狗 | 当前不加载，已由小桌子替代 |
| 默认姿态 | `home` keyframe，机械手朝向螺丝刀 |

## 查看场景

```bash
export MUJOCO_GL=glfw
.venv/bin/python scripts/viewer.py assets/scenes/rm65_dexhand_scene.xml
```

`export MUJOCO_GL=glfw` 表示让 MuJoCo 使用 GLFW 打开图形窗口。只对当前终端生效；如果不想每次输入，可以加到 `~/.bashrc`。

## 遥操作

```bash
export MUJOCO_GL=glfw
.venv/bin/python scripts/teleop.py
```

默认加载：

```text
assets/scenes/rm65_dexhand_scene.xml
```

按键映射：

| 输入 | 动作 |
|---|---|
| `W/S` | 末端前后 |
| `A/D` | 末端左右 |
| `Q/E` | 末端上下 |
| `Z/X` | 手腕 roll |
| `R/F` | 手部开合 |
| `1..6` | 手势预设 |
| `U/J` | 拇指微调 |
| `I/K` | 食指微调 |
| `O/L` | 中指微调 |
| `P/;` | 无名指和小指微调 |

如果 `pynput` 可用，需要按住 `Space` 才输出动作。如果 `pynput` 不可用，脚本会使用 MuJoCo 主窗口按键 fallback：把焦点点到 MuJoCo viewer 后直接按键，不需要 `Space`。

退出遥操作时会自动保存当前姿态到：

```text
assets/scenes/current_state.npz
```

`tune_camera.py` 默认会从这个状态开始调整摄像头，所以可以先把机械臂遥操作到目标姿态，再接着调相机。

遥操作速度可以运行时调整：

```bash
.venv/bin/python scripts/teleop.py \
  --control-fps 30 \
  --translation-step 0.02 \
  --rotation-step 0.08 \
  --hand-step 0.05 \
  --max-joint-step 0.10
```

## 摄像头

当前场景有两个摄像头：

| 摄像头 | 位置 |
|---|---|
| `table_camera` | 放在机械臂所在小桌子上、机械臂前方，正对螺丝刀 |
| `wrist_overhead_camera` | 挂在机械手末端 `link_6` 上 |

微调摄像头：

```bash
.venv/bin/python tune_camera.py
```

默认会读取 `assets/scenes/current_state.npz` 作为机械臂/手的当前状态；如果想忽略当前状态、直接从 XML 默认姿态开始：

```bash
.venv/bin/python tune_camera.py --no-state
```

也可以指定启动时调哪个摄像头：

```bash
.venv/bin/python tune_camera.py --camera table_camera
.venv/bin/python tune_camera.py --camera wrist_overhead_camera
```

`tune_camera.py` 会在 MuJoCo 场景里显示摄像头小模型和视锥，微调页面里可以选择当前摄像头，并用滑块调整 `X/Y/Z/RX/RY/RZ`。调好后点击 `Save XML` 保存到：

```text
assets/scenes/rm65_dexhand_scene.xml
```

保存前会自动备份：

```text
assets/scenes/rm65_dexhand_scene.xml.bak
```

## 验证场景

```bash
.venv/bin/python - <<'PY'
import mujoco
m = mujoco.MjModel.from_xml_path("assets/scenes/rm65_dexhand_scene.xml")
print("joints", m.njnt, "geoms", m.ngeom, "cameras", m.ncam)
PY
```

正常情况下会有 RM65 和右手关节、两个摄像头和完整几何体。

## 更换硬件

机械臂和手的资源目录：

```text
assets/arm/
assets/hand/
```

替换硬件后运行：

```bash
.venv/bin/python scripts/build_hardware_scene.py \
  --arm assets/arm/your_arm.urdf \
  --hand assets/hand/your_hand.urdf \
  --arm-mount-link link_6 \
  --hand-root-link right_hand_base \
  --hand-mount-xyz 0 0 -0.08 \
  --hand-mount-rpy 0 0 0 \
  --output assets/scenes/your_scene.xml
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--arm` | 机械臂 URDF 路径 |
| `--hand` | 手或夹爪 URDF 路径 |
| `--arm-mount-link` | 手挂到机械臂哪个 link 上 |
| `--hand-root-link` | 手模型根 link |
| `--hand-mount-xyz` | 手相对机械臂末端的位置偏移，单位米 |
| `--hand-mount-rpy` | 手相对机械臂末端的姿态，单位弧度 |
| `--output` | 输出场景路径，推荐 `.xml` |

## 工程结构

```text
KeyCollect/
├── assets/
│   ├── arm/            # RM65 URDF + mesh
│   ├── hand/           # dexhand URDF + mesh
│   ├── dog/            # 狗模型资源，当前默认场景不加载
│   └── scenes/         # 当前生成和运行的场景
├── config/             # YAML 配置文件
├── lerobot_robot_mujoco/
├── lerobot_teleoperator_keyboard_mouse/
├── processors/
├── scripts/
└── tune_camera.py
```

## 今天的主要更新

- 生成脚本现在输出最终 MJCF：`assets/scenes/rm65_dexhand_scene.xml`
- 用 60cm 小桌子替代机械狗作为机械臂底座
- 螺丝刀工作桌高度调整为 50cm，并尽量靠近机械臂
- 修复 RM65/灵巧手 mesh 被错误缩小导致看不到机械臂的问题
- 增加 `home` 默认姿态，让机械手默认朝向螺丝刀
- 添加两个摄像头：桌面相机和手腕相机
- `tune_camera.py` 支持场景内显示摄像头模型、选择摄像头、滑块微调和保存 XML
- `teleop.py` 默认加载当前 XML 场景，补充按键提示和无 `pynput` fallback
