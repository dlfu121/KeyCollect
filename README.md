# KeyCollect created by ZiyueDong，YixiangZhang 8/24/26

基于 MuJoCo 3.11 和 LeRobot 0.6 的 RM65-6F + DexHand 仿真、动捕遥操作和数据采集工程。

当前只保留动捕手套控制链路：机械臂使用 `rm65/rm65_ik.py` 的解析逆运动学，键盘/鼠标插件和旧 DLS 控制已移除。

## 1. 安装

要求 Ubuntu 24.04、Python 3.12、可用桌面环境（无窗口录制可使用 EGL）。

```bash
cd /home/ee304/dongziyue/KeyCollect
conda activate keycollect
sudo apt install -y libgl1-mesa-dev libglfw3 libglfw3-dev
python -m pip install -U pip
python -m pip install mujoco==3.11.0 'lerobot[dataset,viz]==0.6.1'
python -m pip install -e ./lerobot_robot_mujoco
python -m pip install -e './lerobot_teleoperator_mocap_ros[rosbridge]'
```

## 2. 准备工作

启动前需要另开三个终端，此项目终端只运行KeyCollect（可直接从四开始，前面三个配置基本不会动）（前三个配置在文末找）

### 4. 四终端启动流程（可直接从此处开始，前面的配置基本不会动）

四个终端必须按照下面的顺序启动。启动后的终端必须保持运行，不要关闭。

#### 终端 1：roscore

加载 ROS Noetic：

```bash
source /opt/ros/noetic/setup.bash
```

启动 ROS Master：

```bash
roscore
```

正常输出包括：

```text
started core service [/rosout]
ROS_MASTER_URI=http://localhost:11311/
```

保持终端 1 运行。

#### 终端 2：mocap_joint_publisher

进入动捕 Catkin 工作空间：

```bash
cd "$HOME/dongziyue/mocap_joint_publisher"
```

加载 ROS Noetic：

```bash
source /opt/ros/noetic/setup.bash
```

加载动捕工作空间：

```bash
source "$HOME/dongziyue/mocap_joint_publisher/devel/setup.bash"
```

可选检查包是否可见：

```bash
rospack find mocapapi
```

正常输出：

```text
/home/ee304/dongziyue/mocap_joint_publisher/src/mocapapi
```

启动动捕节点：

```bash
rosrun mocapapi mocap_joint_publisher
```

注意：ROS 包名是 `mocapapi`，节点名才是 `mocap_joint_publisher`。不要使用 `roslaunch mocap_joint_publisher mocap_joint_publisher.launch`。

正常现象：

- 节点持续运行，没有 traceback 或自动退出。
- Axis Studio 开始广播后，节点持续接收数据。
- ROS 中出现 `/right_wrist_pose` 和 `/right_joint_poses`。
- 两个话题的频率约为 50 Hz。

保持终端 2 运行。

#### 终端 3：ROSBridge

加载 ROS Noetic：

```bash
source /opt/ros/noetic/setup.bash
```

启动 ROSBridge WebSocket：

```bash
roslaunch rosbridge_server rosbridge_websocket.launch
```

正常输出会说明 WebSocket 已在 `9090` 端口启动或监听。

保持终端 3 运行。

#### 终端 4：验证数据并启动 KeyCollect（本终端）

先加载 ROS：

```bash
source /opt/ros/noetic/setup.bash
```

确认两个动捕话题存在：

```bash
rostopic list | grep -E '/right_wrist_pose|/right_joint_poses'
```

正常输出：

```text
/right_joint_poses
/right_wrist_pose
```

检查手腕频率：(也可以不检查，基本没问题)

```bash
rostopic hz /right_wrist_pose
```

正常值约为 50 Hz。看到稳定频率后按 `Ctrl+C` 结束检查。

检查手指频率：

```bash
rostopic hz /right_joint_poses
```

正常值约为 50 Hz。看到稳定频率后按 `Ctrl+C` 结束检查。

检查一帧手指数据：

```bash
rostopic echo -n 1 /right_joint_poses
```

正常情况下会输出一条 `Float32MultiArray` 消息，其 `data` 包含 57 个数值。

确认 ROSBridge 正在监听 9090：

```bash
ss -ltn | grep ':9090'
```

进入 KeyCollect：

```bash
cd "$HOME/dongziyue/KeyCollect"
```

初始化 Conda shell：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

激活 KeyCollect 环境：

```bash
conda activate keycollect
```

确认环境：

```bash
echo "$CONDA_DEFAULT_ENV"
```

正常输出：

```text
keycollect
```

可选检查场景执行器数量：

```bash
python -c "import mujoco; m=mujoco.MjModel.from_xml_path('assets/scenes/rm65_dexhand_scene.xml'); print('actuators =', m.nu)"
```

正常输出：

```text
actuators = 26
```
> ok到这里就开完了所有的终端，可以开始调试啦:)


## 3. 常用命令

```bash
# 查看场景
export MUJOCO_GL=glfw
python scripts/viewer.py assets/scenes/rm65_dexhand_scene.xml

# 检查依赖和场景
python scripts/doctor.py

# 启动动捕遥操作（默认订阅 ROS 话题）
python scripts/teleop.py --transport auto
```

动捕节点应发布 `/right_wrist_pose`（`geometry_msgs/PoseStamped`）和 `/right_joint_poses`（`std_msgs/Float32MultiArray`，57 维）。


启动遥操作时保持右手自然中立，第一帧会自动作为零点；消息超过超时时间会输出安全零增量。


## 4. 配置

- `config/robot.yaml`：场景路径、机械臂和 DexHand 关节名、末端 body。
- `config/simulation.yaml`：MuJoCo 时间步长、渲染和安全参数。
- `config/mocap_teleop.yaml`：ROS 话题、传输方式、位置/姿态/手指缩放、超时。

默认模型：

```text
assets/arm/RM65-6F.urdf
assets/hand/dexhand021_right_simplified.urdf
assets/scenes/rm65_dexhand_scene.xml
```

替换模型后生成组合场景：

```bash
python scripts/build_hardware_scene.py \
  --arm assets/arm/your_arm.urdf \
  --hand assets/hand/your_hand.urdf \
  --output assets/scenes/your_scene.xml
```

然后同步修改 `config/robot.yaml` 中的关节名和场景路径。`scripts/teleop.py` 固定使用解析 RM65 IK，不再提供 `--ik-solver` 参数。


## 5. 数据采集

先确认无录制遥操作正常，再运行项目的 MuJoCo 录制入口。该入口复用
`lerobot-record` 的数据集和视频编码逻辑，并增加逐 episode 的快捷键与自动 reset：

- `q`、`n` 或右方向键：结束并保存当前 episode，随后 reset 场景；
- `r` 或左方向键：丢弃当前 episode，reset 后重新录制；
- `Esc`：保存当前 episode、reset 场景并结束整个录制任务；
- 达到 `episode_time_s` 时也会自动保存并 reset；
- 每次 reset 都恢复 XML 的 `home` keyframe，然后重新随机排布红、蓝螺丝刀。

```bash
export MUJOCO_GL=egl
python scripts/record_mujoco.py \
  --robot.type=mujoco \
  --robot.id=rm65_dexhand \
  --robot.scene_path=assets/scenes/rm65_dexhand_scene.xml \
  --robot.ee_site_name=link_6 \
  --robot.arm_joint_names='["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]' \
  --robot.gripper_joint_names='["r_f_joint1_1","r_f_joint1_2","r_f_joint1_3","r_f_joint1_4","r_f_joint2_1","r_f_joint2_2","r_f_joint2_3","r_f_joint2_4","r_f_joint3_1","r_f_joint3_2","r_f_joint3_3","r_f_joint3_4","r_f_joint4_1","r_f_joint4_2","r_f_joint4_3","r_f_joint4_4","r_f_joint5_1","r_f_joint5_2","r_f_joint5_3","r_f_joint5_4"]' \
  --robot.cameras='{"table_camera":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist_overhead_camera":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30}}' \
  --teleop.type=mocap_ros \
  --teleop.transport=auto \
  --dataset.repo_id=local/rm65_dexhand \
  --dataset.single_task="Use the RM65 DexHand to grasp a screwdriver" \
  --dataset.root=data/rm65_dexhand \
  --dataset.fps=30 \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=0 \
  --dataset.push_to_hub=false
```

建议每次采集使用新的 `data/<dataset_name>/` 目录。上传 Hugging Face 前先执行 `hf auth login`。


## 6. 工程结构

```text
assets/                         # 模型、网格和场景
config/                         # 运行配置
lerobot_robot_mujoco/           # MuJoCo 仿真和运动学辅助
lerobot_teleoperator_mocap_ros/ # 动捕遥操作插件及测试
rm65/                           # RM65 当前解析 FK/IK
processors/                     # 末端控制和数据处理
scripts/                        # viewer、teleop、场景生成和诊断
data/                           # 数据集输出和 smoke-test 数据
```


## 7. 测试（第2部分的前三项）

本文档用于 Linux 主机关机或重启后，重新启动 Axis Studio → ROS1 → ROSBridge → KeyCollect MuJoCo 动捕遥操作系统。

### 1. 固定环境

- Linux 用户：`ee304`
- KeyCollect：`$HOME/dongziyue/KeyCollect`
- Conda 环境：`keycollect`
- ROS：ROS1 Noetic
- 动捕 Catkin 工作空间：`$HOME/dongziyue/mocap_joint_publisher`
- ROS 包名：`mocapapi`
- 动捕节点名：`mocap_joint_publisher`
- Windows 动捕上位机：`192.170.10.10`
- Linux 有线网卡：`enp3s0`
- Linux 有线地址：`192.170.10.8`
- Axis Studio UDP 目标：`192.170.10.8:7077`
- ROSBridge WebSocket：`127.0.0.1:9090`
- 手腕话题：`/right_wrist_pose`
- 手指话题：`/right_joint_poses`，57 维
- 正常发布频率：约 50 Hz
- MuJoCo 场景：`assets/scenes/rm65_dexhand_scene.xml`
- 场景执行器数量：26
- 稳定参数：`kp=100`、`kv=10`

> 启动 KeyCollect 的瞬间，手腕和所有手指必须保持自然中立姿态。动捕遥操作器会把收到的第一帧作为零点。

### 2. Axis Studio 设置

在 Windows 上打开 Axis Studio，确认广播参数：

- 协议：UDP
- 格式：二进制
- 帧头：新帧头
- 旋转顺序：YXZ
- 位移：开启
正常情况下会很快出现来自 `192.170.10.10`、发往 `192.170.10.8.7077` 的 UDP 数据包。确认后进入四终端启动流程。


```bash
python -m pytest -q lerobot_teleoperator_mocap_ros/tests rm65/test_rm65_ik.py
```

测试覆盖 RM65 运动学、重定向、场景接口和插件发现，不会自动启动真实 ROS 节点。
