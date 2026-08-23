# RM65 MATLAB 封闭解析逆运动学验证

本目录是一个自包含的 MATLAB 验证工程，不依赖 Robotics System Toolbox。机械臂按照睿尔曼网页公布的改进 D-H 参数建立，逆解为肩部、肘部和腕部有限分支的几何封闭解析解，不使用数值迭代。

## 文件说明

- `rm65_parameters.m`：各型号 MDH 参数、offset 和关节限位。
- `rm65_mdh_transform.m`：单节改进 D-H 变换。
- `rm65_fk.m`：正运动学和各关节坐标系。
- `rm65_ik_all.m`：枚举限位内的全部解析逆解。
- `rm65_ik_nearest.m`：选择最接近当前关节角的解析解。
- `rm65_select_continuous.m`：带分支滞回、速度和奇异惩罚的连续选择器。
- `rm65_pose_error.m`：正运动学回代误差。
- `rm65_validate_animation.m`：30 Hz 路径验证和动态可视化。
- `rm65_batch_test.m`：随机位姿批量验证。
- `run_rm65_validation.m`：默认动画的一键运行入口。

## 运行动态验证

可以直接打开并运行 `run_rm65_validation.m`。也可以在 MATLAB 中执行：

```matlab
cd('F:\zahuo\rm65\matlab_validation');
result = rm65_validate_animation("RM65-B", 12, true);
```

参数依次为：机器人型号、动画时长（秒）、是否按墙钟时间实时播放。程序默认使用 30 Hz，即 12 秒对应 361 个路径点。设为 `false` 可以取消帧间等待，快速跑完整条路径：

```matlab
result = rm65_validate_animation("RM65-B", 12, false);
```

窗口左侧显示机械臂、完整目标路径、已执行路径以及目标/实际法兰坐标系。右侧实时显示：

1. 六个关节角；
2. 六个关节角速度；
3. 位置和姿态回代误差。

全部数据保存在返回的 `result` 中：

```matlab
result.solvedJointDeg
result.jointVelocityDegPerSecond
result.positionError
result.orientationError
result.solutionCount
result.branch
```

## 批量验证

```matlab
report = rm65_batch_test("RM65-B", 500);
```

批量程序在所有关节限位内随机产生 500 组关节角，由正运动学生成目标位姿，再调用解析逆解器并执行正运动学回代。

## 调用自己的路径点

每个目标点应为 4×4 法兰齐次变换矩阵，位置单位为米。连续轨迹应把上一点关节角作为下一点的 `seed`：

```matlab
robot = rm65_parameters("RM65-B");
seed = deg2rad([0, 0, 0, 0, 0, 0]);

for index = 1:size(targetPoses, 3)
    target = targetPoses(:, :, index);
    solution = rm65_ik_nearest(target, robot, seed);
    jointsDeg(index, :) = solution.qDeg;
    seed = solution.q;
end
```

如果输入的是 TCP 位姿而不是机器人法兰位姿，应先去除工具变换：

```matlab
targetFlange = targetTcp / flangeToTcp;
```

## 型号和尺寸

默认提供 RM65-B、RM65-B-V、RM65-6FB、RM65-6FB-V 和 RM65-6F。网页抓取结果没有完整显示型号表中的 `d6` 数值，因此当前按照官方工作半径与 `256 + 210 + d6` 的几何关系设置为：B=144 mm、6FB=161 mm、6F=172.5 mm。真机验证前请以具体机械臂铭牌和厂家原始参数表复核 `d6`。
