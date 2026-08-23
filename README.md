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

## 零、在wsl安装Ubantu24.04

在Powershell中输入

```bash
wsl --install -d Ubuntu-24.04
```

设置用户名和密码后会直接进入Ubantu24.04系统，退出wsl，并且重新进入。

在Powershell中继续输入

```bash
wsl --set-default Ubuntu-24.04
```

设置Ubantu24.04为默认版本，之后再输入

```bash
wsl
```

进入Ubantu24.04

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

## 二、进入 KeyCollect Python 环境

```bash
cd /home/ee304/dongziyue/KeyCollect
conda activate keycollect
python --version
```

## 三、安装 Python 包

```bash
python -m pip install --upgrade pip
python -m pip install matplotlib
python -m pip install mujoco==3.11.0
python -m pip install lerobot==0.6.1
python -m pip install 'lerobot[dataset]==0.6.1'
python -m pip install lerobot[viz]
python -m pip install -e ./lerobot_robot_mujoco
python -m pip install -e './lerobot_teleoperator_mocap_ros[rosbridge]'
```

## 四、启动屏幕渲染

```bash
conda activate keycollect
export MUJOCO_GL=glfw
python3 scripts/viewer.py
```

默认遥操作使用带 26 个位置伺服器的 `assets/scenes/rm65_dexhand_scene.xml`。URDF 是场景生成的中间产物，不应直接用于遥操作。单独查看场景时可以指定 MJCF 文件：

```bash
python3 scripts/viewer.py assets/scenes/rm65_dexhand_scene.xml
```

也可以指定自己的场景文件：

```bash
python3 scripts/viewer.py assets/scene/your_scene.xml
```

如果你已经在桌面环境里运行，一般不需要再额外设置 `DISPLAY`。

## 五、动捕手套遥操作

直接启动：

```bash
conda activate keycollect
export MUJOCO_GL=glfw
python3 scripts/teleop.py
```

该入口只订阅 `/right_wrist_pose` 和 `/right_joint_poses`，不会注册键盘或鼠标控制。
启动时保持右手自然中立，收到的第一帧会作为自动零点。

当前默认硬件是：

```text
机械臂：assets/arm/RM65-6F.urdf
灵巧手：assets/hand/dexhand021_right_simplified.urdf
组合场景：assets/scenes/rm65_dexhand_scene.xml
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
assets/scenes/rm65_dexhand_scene.xml
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
scene_path: assets/scenes/rm65_dexhand_scene.xml

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
m = mujoco.MjModel.from_xml_path("assets/scenes/rm65_dexhand_scene.xml")
print("joints", m.njnt, "geoms", m.ngeom, "cameras", m.ncam)
PY
```

如果能正常输出关节和几何数量，就说明 MuJoCo 可以加载该硬件场景。
### 七、摄像头组件
修改摄像头位置：`python tune_camera.py`
### 八、数据采集命令
```bash
conda activate keycollect

MUJOCO_GL=egl

lerobot-record \
  --robot.type=mujoco \
  --robot.id=rm65_dexhand \
  --robot.calibration_dir=.cache/lerobot/calibration/robots/mujoco \
  --robot.scene_path=assets/scenes/rm65_dexhand_scene.xml \
  --robot.arm_joint_names='["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]' \
  --robot.gripper_joint_names='["r_f_joint1_1","r_f_joint1_2","r_f_joint1_3","r_f_joint1_4","r_f_joint2_1","r_f_joint2_2","r_f_joint2_3","r_f_joint2_4","r_f_joint3_1","r_f_joint3_2","r_f_joint3_3","r_f_joint3_4","r_f_joint4_1","r_f_joint4_2","r_f_joint4_3","r_f_joint4_4","r_f_joint5_1","r_f_joint5_2","r_f_joint5_3","r_f_joint5_4"]' \
  --robot.ee_site_name=link_6 \
  --robot.cameras='{"table_camera":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist_overhead_camera":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30}}' \
  --teleop.type=mocap_ros \
  --teleop.transport=auto \
  --teleop.position_scale=0.01 \
  --teleop.stale_timeout_s=0.25 \
  --dataset.repo_id=dlfu121/Industrial \
  --dataset.single_task="这里填上具体任务命令" \
  --dataset.root=data/rm65_dexhand_test \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false
```
采集后，上传数据：
```bash
hf auth login
```
可以用我的accesstoken，会发在群里，然后上传远程仓库：
```bash
hf upload dlfu121/Industrial1 . --repo-type=dataset
```

## 九、动捕手套 ROS1 遥操作

`first_version/mocap_joint_publisher` 保持不变。KeyCollect 新增的
`mocap_ros` LeRobot 遥操作插件直接消费它发布的：

- `/right_wrist_pose` (`geometry_msgs/PoseStamped`)
- `/right_joint_poses` (`std_msgs/Float32MultiArray`, 57 维)

映射沿用旧 `mujoco_push_t-main` 的第一帧自动归零、位置 `z/x/y`
轴重排、`0.01` 位置缩放、右手姿态相似变换，以及 57 维手套到
DexHand 20 关节的索引和 PIP-DIP 耦合。消息超过 0.25 秒未更新时，
插件自动输出零增量。

### 9.1 使用 rosbridge（当前 Python 3.12 环境）

ROS Noetic 终端启动动捕节点和 rosbridge：

```bash
source /opt/ros/noetic/setup.bash
source first_version/mocap_joint_publisher/devel/setup.bash
roslaunch rosbridge_server rosbridge_websocket.launch
```

KeyCollect 终端使用独立的 Python 3.12 环境：

```bash
conda activate keycollect
python -m pip install -e './lerobot_teleoperator_mocap_ros[rosbridge]'
```

启动原动捕节点后，先确认话题正常：

```bash
rostopic hz /right_wrist_pose
rostopic echo -n 1 /right_joint_poses
```

### 9.2 transport 选择

ROS Noetic 与 KeyCollect Python 3.12 分属不同环境，默认 `transport=auto`
会在 `rospy` 不可用时连接本机 `127.0.0.1:9090` 的 rosbridge，
不需要修改 `mocap_joint_publisher`。也可以显式指定：

```bash
python scripts/teleop.py --transport=rosbridge
```

### 9.3 先做不录制的遥操作检查

下面命令中的机器人参数与第八节一致，仅将遥操作器换成动捕手套：

```bash
MUJOCO_GL=glfw lerobot-teleoperate \
  --robot.type=mujoco \
  --robot.id=rm65_dexhand \
  --robot.scene_path=assets/scenes/rm65_dexhand_scene.xml \
  --robot.arm_joint_names='["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]' \
  --robot.gripper_joint_names='["r_f_joint1_1","r_f_joint1_2","r_f_joint1_3","r_f_joint1_4","r_f_joint2_1","r_f_joint2_2","r_f_joint2_3","r_f_joint2_4","r_f_joint3_1","r_f_joint3_2","r_f_joint3_3","r_f_joint3_4","r_f_joint4_1","r_f_joint4_2","r_f_joint4_3","r_f_joint4_4","r_f_joint5_1","r_f_joint5_2","r_f_joint5_3","r_f_joint5_4"]' \
  --robot.ee_site_name=link_6 \
  --teleop.type=mocap_ros \
  --teleop.transport=auto \
  --teleop.position_scale=0.01 \
  --teleop.stale_timeout_s=0.25 \
  --fps=30
```

启动时保持手腕和手指处于自然中立姿态，第一帧会成为零点。确认轴向、
幅度和手指方向正确后即可用第八节命令采集；
逐个 DexHand 关节的控制增量会写入数据集 action，而不仅是记录一个总开合量。

常用参数位于 `config/mocap_teleop.yaml`：

- `position_scale`：手腕平移比例，默认沿用旧系统的 `0.01`
- `orientation_scale`：手腕旋转比例
- `finger_scale`：手指弯曲比例
- `max_*_delta_*`：单控制周期最大变化量
- `transport`：`auto`、`rospy` 或 `rosbridge`
