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
python -m pip install mujoco==3.11.0 'lerobot[dataset,viz,training]==0.6.1'
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

机器人、动捕、相机和数据集参数统一保存在 `config/record_mujoco.yaml`。
默认的新环境包含三路视觉 observation：

```text
observation.images.table_camera             RGB，身前固定相机
observation.images.wrist_overhead_camera    RGB，随 link_6 运动
observation.images.table_camera_depth       float32 米制深度，形状 (480, 640, 1)
```

身前 RGB 和深度使用同一个 MuJoCo 相机，因此像素严格对齐。身前相机与固定的机械臂
底座都位于 world 坐标系，二者相对位姿不变。

### 5.1 采集新的 RGB + Depth 数据

先确认无录制遥操作正常，再启动采集：

```bash
cd "$HOME/dongziyue/KeyCollect"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate keycollect

python scripts/record_next.py --profile depth
```

`depth` 是默认 profile，因此也可以省略 `--profile depth`。启动器会设置
`MUJOCO_GL=glfw`，并在 `/media/ee304/FDL` 下选择第一个未占用目录：

```text
/media/ee304/FDL/rm65_dexhand_depth_run_001
/media/ee304/FDL/rm65_dexhand_depth_run_002
/media/ee304/FDL/rm65_dexhand_depth_run_003
...
```

已存在的目录不会被覆盖。运行前可以预览编号和实际命令：

```bash
python scripts/record_next.py --profile depth --dry-run
```

录制中，`q` / `n` / `Right` 保存当前 episode 并进入下一条，`r` / `Left`
放弃当前 episode 并立即重录，`Esc` 保存当前 episode 后结束整个采集会话。

临时覆盖 episode 数量或时长：

```bash
python scripts/record_next.py --profile depth \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=45
```

深度帧以浮点米为输入，由 LeRobot 的 depth encoder 单独编码；它不会被当作普通
RGB 视频。不要向旧的 RGB-only 数据集 resume 写入，因为新旧 feature schema 不同。

### 5.2 合并新的深度数据

只合并 `rm65_dexhand_depth_run_*`：

```bash
python scripts/merge_datasets.py \
  --profile depth \
  --data_root /media/ee304/FDL
```

默认输出：

```text
目录：data/rm65_dexhand_depth_merged
repo：local/rm65_dexhand_depth_merged
```

合并脚本按 profile 筛选，因此不会把旧 RGB 数据误混进深度数据。需要跳过某些采集
批次时，例如跳过 2 和 5：

```bash
python scripts/merge_datasets.py --profile depth --exclude 2 5
```

### 5.3 检查合并后的深度特征

```bash
python -c "from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata; m=LeRobotDatasetMetadata('local/rm65_dexhand_depth_merged', root='data/rm65_dexhand_depth_merged'); print(m.features['observation.images.table_camera_depth']); print('depth_keys:', m.depth_keys)"
```

必须看到：

```text
shape: [480, 640, 1]
is_depth_map: true
depth_keys: ['observation.images.table_camera_depth']
```

### 5.4 训练 RGB + Depth ACT

标准 ResNet18 只接受三通道输入，而数据集保留的是单通道米制深度。必须通过项目的
`train_act_depth.py` 启动训练；它只在深度进入共享 ResNet 前把 `C=1` 复制成
`C=3`，RGB 图像保持不变。该入口会强制 LeRobot 解码深度时使用米，确保采集、
训练归一化统计和在线推理始终使用同一单位。

```bash
python scripts/train_act_depth.py \
  --policy.type=act \
  --policy.device=cuda \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/rm65_dexhand_depth_merged \
  --dataset.root=/home/ee304/dongziyue/KeyCollect/data/rm65_dexhand_depth_merged \
  --dataset.depth_output_unit=m \
  --dataset.video_backend=pyav \
  --output_dir=outputs/train/act_rm65_dexhand_depth \
  --batch_size=16 \
  --steps=40000 \
  --save_freq=20000 \
  --env_eval_freq=0
```

训练生成的 checkpoint 仍是标准 LeRobot ACT 格式，但其配置会声明三路视觉输入，
其中 `table_camera_depth` 的形状为 `[1, 480, 640]`。

### 5.5 推理 RGB + Depth ACT

推理脚本会读取 checkpoint 声明的视觉特征，并自动安装与训练相同的单通道适配器：

```bash
MUJOCO_GL=glfw python scripts/infer_mujoco.py \
  --checkpoint outputs/train/act_rm65_dexhand_depth \
  --dataset-root /home/ee304/dongziyue/KeyCollect/data/rm65_dexhand_depth_merged \
  --device cuda \
  --random-seed 42
```

正常日志应包含：

```text
Policy ready: act on cuda
Controlling ['table_camera', 'wrist_overhead_camera', 'table_camera_depth'] cameras at 24 Hz
```

`--device cuda` 会在 CUDA 不可用时直接报错，避免无意中退回 CPU；需要自动选择时可改为
`--device auto`。

### 5.6 旧 RGB 流程

旧采集、合并、训练和推理仍可使用。旧 profile 会关闭深度并继续使用
`rm65_dexhand_run_*` 前缀：

```bash
# 旧的两路 RGB 采集
python scripts/record_next.py --profile rgb

# 只合并旧 RGB 数据
python scripts/merge_datasets.py --profile rgb

# 旧 ACT 训练不需要深度适配器
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/rm65_dexhand_merged \
  --dataset.root=/home/ee304/dongziyue/KeyCollect/data/rm65_dexhand_merged \
  --dataset.video_backend=pyav \
  --output_dir=outputs/train/act_rm65_dexhand_rgb \
  --batch_size=16 \
  --steps=40000 \
  --save_freq=20000 \
  --env_eval_freq=0

# 旧 checkpoint 仍由同一个推理脚本加载，只会请求原来的两路 RGB
MUJOCO_GL=glfw python scripts/infer_mujoco.py \
  --checkpoint outputs/train/act_rm65_dexhand_rgb \
  --dataset-root /home/ee304/dongziyue/KeyCollect/data/rm65_dexhand_merged \
  --device cuda \
  --random-seed 42
```


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
