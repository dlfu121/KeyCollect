# 机器人/场景交付规范

优先交付 **MuJoCo MJCF/XML 场景文件**，而不是只交付 URDF。URDF 可以作为机器人来源文件提供，但最终用于本项目运行的文件应是 MuJoCo 可直接加载的 `.xml`。

## 推荐交付内容

```text
assets/
├── scene/
│   └── your_scene.xml          # 最终可运行的 MuJoCo 场景
└── robot/
    ├── your_robot.urdf         # 可选：机器人原始 URDF
    └── meshes/                 # STL/OBJ/DAE 等 mesh 文件
        ├── link1.stl
        ├── link2.stl
        └── ...
```

## 场景文件必须包含

1. 机器人本体和关节

机器人关节需要有稳定、明确的名字，例如：

```xml
<joint name="joint1" .../>
<joint name="joint2" .../>
<joint name="joint3" .../>
```

这些名字要提供给控制程序，用来填写 `config/robot.yaml`。

2. 末端执行器 site

请在夹爪或工具中心位置放一个 site：

```xml
<site name="ee_site" pos="0 0 0" size="0.005"/>
```

如果不用 `ee_site` 这个名字，请同时说明实际名称。

3. 观察相机

提供两个固定相机，要严格按照装配体的位置布置，推荐命名：

```xml
<camera name="camera_front" .../>
<camera name="camera_side" .../>
```

可选再提供：

```xml
<camera name="camera_top" .../>
```

4. 夹爪关节

如果机器人有夹爪，请提供夹爪关节名，例如：

```xml
<joint name="finger_left_joint" .../>
<joint name="finger_right_joint" .../>
```

5. actuator

需要给可控关节配置 actuator，否则程序无法控制机器人运动：

```xml
<actuator>
  <motor name="act_joint1" joint="joint1" gear="50"/>
  <motor name="act_joint2" joint="joint2" gear="50"/>
</actuator>
```

如果使用 `<position>` actuator，也请说明控制范围、单位和对应关节。

## 坐标和单位要求

| 项目 | 要求 |
|---|---|
| 长度单位 | 米 |
| 角度单位 | 弧度 |
| 重力方向 | 推荐 `0 0 -9.81` |
| 机器人基座 | 推荐放在世界坐标附近 |
| 末端 site | 放在真实工具中心或夹爪中心 |
| 相机 | 能清楚看到机器人、操作物体和任务区域 |

## 需要一并说明的信息

请随文件一起提供以下信息：

```text
1. 场景 XML 文件路径
2. 机械臂关节名列表，按控制顺序排列
3. 夹爪关节名列表
4. 末端执行器 site 名称
5. 相机名称列表
6. 每个关节的运动范围
7. mesh 文件是否全部使用相对路径
8. 是否需要特殊 asset 路径或材质贴图
```

## 最小验收标准

交付前请确认：

- `mujoco.MjModel.from_xml_path("your_scene.xml")` 可以成功加载
- 所有 mesh 路径都能被找到
- 场景里至少有一个灯光、地面或工作台
- `ee_site` 能跟随机器人末端运动
- `camera_front` 和 `camera_side` 能看到操作区域
- 所有关节名和相机名没有重复
- 可控关节都有 actuator

## URDF 交付说明

如果只能先交付 URDF，请同时提供：

```text
your_robot.urdf
meshes/
关节名说明
末端 link 名称
夹爪关节说明
推荐安装姿态和基座位置
```

URDF 转成 MJCF 后，还需要补充 MuJoCo 场景信息，例如相机、灯光、地面、任务物体、actuator、`ee_site` 等。本项目最终运行时仍以 MuJoCo XML 为准。

# 工程结构

```text
KeyCollect/
├── assets/
│   └── scene/          # MuJoCo 场景 XML
├── config/             # YAML 配置文件
├── lerobot_robot_mujoco/    # MuJoCo Robot Plugin
├── lerobot_teleoperator_keyboard_mouse/  # 键鼠 Teleop Plugin
├── processors/         # IK、EE 控制、动作映射
└── scripts/            # 诊断和屏幕渲染脚本
```

# 屏幕渲染

项目支持两种方式：

| 方式 | 用途 |
|---|---|
| 屏幕渲染 | 打开 MuJoCo viewer 窗口观察仿真 |
| 离屏渲染 | 采集相机图像，用于记录和训练 |

代码里也可以直接启动 viewer：

```python
from lerobot_robot_mujoco.simulation import MuJoCoSimulation

sim = MuJoCoSimulation("assets/scene/demo_scene.xml")
sim.load()
sim.reset()
sim.launch_viewer()

while sim.sync_viewer():
    sim.step()
```

# 任务

- `open_cabinet_door`: 开柜门
- `pick_screwdriver`: 拾取螺丝刀

# 备注

如果你在无桌面环境里跑，才需要额外考虑 Xvfb 或 OSMesa。
