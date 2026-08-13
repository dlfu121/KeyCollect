# MuJoCo 3.11.0 x LeRobot 0.6.1 仿真数据采集系统

基于 MuJoCo 物理引擎和 LeRobot 框架的机器人仿真遥操作与数据采集系统。

## 系统要求

| 组件 | 版本 |
|---|---|
| OS | Ubuntu 24.04 |
| Python | 3.12 |
| MuJoCo | 3.11.0 |
| LeRobot | 0.6.1 |
| 显示器 | 需要可用桌面环境 |

## 一、安装系统依赖

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

> `libglfw3` 和 `libglfw3-dev` 用于 MuJoCo 窗口渲染。

## 二、创建 Python 虚拟环境

```bash
cd /path/to/KeyCollect
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

## 三、安装 Python 包

```bash
python -m pip install --upgrade pip
python -m pip install mujoco==3.11.0
python -m pip install lerobot==0.6.1
python -m pip install pynput
python -m pip install -e ./lerobot_robot_mujoco
python -m pip install -e ./lerobot_teleoperator_keyboard_mouse
```

## 四、启动屏幕渲染

```bash
export MUJOCO_GL=glfw
python3 scripts/viewer.py
```

默认遥操作使用 `assets/scene/rm65_dexhand_scene.urdf`。单独查看场景时可以指定这个文件：

```bash
python3 scripts/viewer.py assets/scene/rm65_dexhand_scene.urdf
```

也可以指定自己的场景文件：

```bash
python3 scripts/viewer.py assets/scene/your_scene.xml
```

如果你已经在桌面环境里运行，一般不需要再额外设置 `DISPLAY`。

## 五、键盘和鼠标遥操作

直接启动：

```bash
export MUJOCO_GL=glfw
python3 scripts/teleop.py
```

按键映射：

| 输入 | 动作 |
|---|---|
| `W/S` | 前后 |
| `A/D` | 左右 |
| `Q/E` | 上下 |
| `Z/X` | 手腕旋转 |
| `R/F` | 夹爪开合 |
| `Space` | 按住才会输出动作 |
| `Esc` | 退出遥操作 |

当前默认硬件是：

```text
机械臂：assets/arm/RM65-6F.urdf
灵巧手：assets/hand/dexhand021_right_simplified.urdf
组合场景：assets/scene/rm65_dexhand_scene.urdf
```

## 六、更换硬件

当前项目支持把机械臂和末端手爪分别放在 `assets/arm/` 和 `assets/hand/`，再生成一个 MuJoCo 可加载的组合场景。

### 6.1 当前目录约定

```text
assets/
├── arm/
│   ├── RM65-6F.urdf
│   ├── base_link.STL
│   ├── link_1.STL
│   └── ...
├── hand/
│   ├── dexhand021_right_simplified.urdf
│   ├── right_hand_base.STL
│   ├── r_f_link1_1.STL
│   └── ...
└── scene/
    ├── demo_scene.xml
    └── rm65_dexhand_scene.urdf
```

### 6.2 重新生成组合硬件场景

如果替换了机械臂或手，运行：

```bash
python3 scripts/build_hardware_scene.py
```

默认会读取：

```text
assets/arm/RM65-6F.urdf
assets/hand/dexhand021_right_simplified.urdf
```

并生成：

```text
assets/scene/rm65_dexhand_scene.urdf
```

也可以手动指定文件：

```bash
python3 scripts/build_hardware_scene.py \
  --arm assets/arm/your_arm.urdf \
  --hand assets/hand/your_hand.urdf \
  --arm-mount-link link_6 \
  --hand-root-link right_hand_base \
  --hand-mount-xyz 0 0 -0.08 \
  --hand-mount-rpy 0 0 0 \
  --output assets/scene/your_hardware_scene.urdf
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--arm` | 机械臂 URDF 路径 |
| `--hand` | 手或夹爪 URDF 路径 |
| `--arm-mount-link` | 手要挂到机械臂哪个 link 上 |
| `--hand-root-link` | 手模型的根 link |
| `--hand-mount-xyz` | 手相对机械臂末端的安装偏移，单位米 |
| `--hand-mount-rpy` | 手相对机械臂末端的安装姿态，单位弧度 |
| `--output` | 生成的组合场景路径 |

如果手和机械臂没有贴紧，优先调 `--hand-mount-xyz`。例如手离机械臂太远，可以继续减小 z 偏移：

```bash
python3 scripts/build_hardware_scene.py --hand-mount-xyz 0 0 -0.12
```

### 6.3 更新遥操作配置

生成新场景后，检查并更新 `config/robot.yaml`：

```yaml
scene_path: assets/scene/rm65_dexhand_scene.urdf

arm_joint_names:
  - joint_1
  - joint_2
  - joint_3
  - joint_4
  - joint_5
  - joint_6

gripper_joint_names:
  - r_f_joint1_1
  - r_f_joint1_2
  - r_f_joint2_1
  - r_f_joint2_2
```

`scripts/teleop.py` 当前默认也使用这套 RM65 + 右手命名。如果换了新的硬件，需要同步改脚本里的：

```text
arm_joints
gripper_joints
--ee-body
```

### 6.4 验证新硬件是否可加载

```bash
python3 - <<'PY'
import mujoco
m = mujoco.MjModel.from_xml_path("assets/scene/rm65_dexhand_scene.urdf")
print("joints", m.njnt, "geoms", m.ngeom, "cameras", m.ncam)
PY
```

如果能正常输出关节和几何数量，就说明 MuJoCo 可以加载该硬件场景。

## 七、给机器人/场景同事的交付规范

请优先交付 **MuJoCo MJCF/XML 场景文件**，也可以先交付 URDF + mesh。URDF 作为来源文件时，本项目会通过 `scripts/build_hardware_scene.py` 生成组合场景。

### 推荐交付内容

将 URDF 和 mesh 文件放入：

```text
assets/arm/
├── your_arm.urdf
├── base_link.STL
└── ...

assets/hand/
├── your_hand.urdf
├── hand_base.STL
└── ...
```

也可以交付已经拼好的 MuJoCo XML：

```text
assets/scene/
└── your_scene.xml
```

### 文件必须包含

1. 机器人本体和关节

机械臂关节需要有稳定、明确的名字，例如：

```xml
<joint name="joint_1" .../>
<joint name="joint_2" .../>
<joint name="joint_3" .../>
```

2. 末端安装 link

请说明手或夹爪应该挂到机械臂哪个 link 上，例如：

```text
arm_mount_link: link_6
```

3. 手或夹爪根 link

请说明手模型的根 link，例如：

```text
hand_root_link: right_hand_base
```

4. 手指或夹爪关节

如果是灵巧手，请提供建议控制的关节列表和手势目标值：

```text
open:
  r_f_joint1_1: 0.0
  r_f_joint2_1: 0.0

grasp:
  r_f_joint1_1: 1.0
  r_f_joint2_1: 0.8
```

5. mesh 路径

mesh 文件必须放在 URDF 附近，且路径能被脚本改写成相对路径。不要只提供本机绝对路径。

### 需要一并说明的信息

请随文件一起提供以下信息：

```text
1. 机械臂 URDF 文件路径
2. 手或夹爪 URDF 文件路径
3. 机械臂关节名列表，按控制顺序排列
4. 手或夹爪关节名列表
5. 末端安装 link 名称
6. 手或夹爪根 link 名称
7. 每个关节的运动范围
8. mesh 文件是否全部齐全
9. 推荐初始姿态
```

### 最小验收标准

交付前请确认：

- URDF 可以被 XML 解析器正常读取
- 所有 mesh 文件都能在 `assets/arm/` 或 `assets/hand/` 下找到
- 所有关节名没有重复
- 机械臂末端安装 link 确实存在
- 手或夹爪根 link 确实存在
- 关节范围使用弧度
- 长度单位使用米

## 旧 demo 场景

项目仍保留 `assets/scene/demo_scene.xml`，用于测试最小 6 轴机械臂 + 简单夹爪。

```bash
python3 scripts/viewer.py assets/scene/demo_scene.xml
```

## 工程结构

```text
KeyCollect/
├── assets/
│   ├── arm/            # 机械臂 URDF + mesh
│   ├── hand/           # 手或夹爪 URDF + mesh
│   └── scene/          # 生成后的 MuJoCo/URDF 场景
├── config/             # YAML 配置文件
├── lerobot_robot_mujoco/    # MuJoCo Robot Plugin
├── lerobot_teleoperator_keyboard_mouse/  # 键鼠 Teleop Plugin
├── processors/         # IK、EE 控制、动作映射
└── scripts/            # 诊断、生成和屏幕渲染脚本
```

## 屏幕渲染

项目支持两种方式：

| 方式 | 用途 |
|---|---|
| 屏幕渲染 | 打开 MuJoCo viewer 窗口观察仿真 |
| 双相机窗口 | `scripts/teleop.py` 中的 `camera_panel` |
| 离屏渲染 | 采集相机图像，用于记录和训练 |

代码里也可以直接启动 viewer：

```python
from lerobot_robot_mujoco.simulation import MuJoCoSimulation

sim = MuJoCoSimulation("assets/scene/rm65_dexhand_scene.urdf")
sim.load()
sim.reset()
sim.launch_viewer()

while sim.sync_viewer():
    sim.step()
```

## 任务

- `open_cabinet_door`: 开柜门
- `pick_screwdriver`: 拾取螺丝刀

## 备注

如果你在无桌面环境里跑，才需要额外考虑 Xvfb 或 OSMesa。
