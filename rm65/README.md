# RM65 六自由度逆运动学求解器

依据睿尔曼官方 RM65 改进 D-H 参数实现，内部单位为米和弧度。求解器采用几何封闭解析法，不使用雅可比矩阵、随机初值或数值迭代，适用于 RM65-B、RM65-6FB 和 RM65-6F。

## 使用

```python
import numpy as np
from rm65_ik import RM65Kinematics, pose_matrix, rotation_from_rpy

robot = RM65Kinematics("RM65-B")
target = pose_matrix(
    position=[0.30, 0.10, 0.35],                 # 米
    rotation=rotation_from_rpy([180, 0, 30], degrees=True),
)
solutions = robot.inverse_all(target, seed=[0, 0, 0, 0, 0, 0], seed_degrees=True)

for solution in solutions:
    print(solution.joints_deg, solution.shoulder, solution.elbow, solution.wrist)

# 如果只需要一组，则返回最接近 seed 的解析解。
nearest = robot.inverse(target, seed=[0, 0, 0, 0, 0, 0], seed_degrees=True)
```

运行自检：

```powershell
python -m unittest -v
python rm65_ik.py
```

`inverse_all()` 枚举关节限位内的肩部、肘部和腕部解析分支。`inverse()` 根据 `seed` 返回最近的一组，实际连续轨迹中应传入机械臂当前关节角。腕部奇异时存在无穷多解，程序使用 `seed` 中的 J4 固定一组代表解。目标矩阵描述的是法兰坐标系；若使用工具中心点（TCP），应先右乘工具变换的逆矩阵得到法兰目标。

连续轨迹推荐使用带分支滞回的有状态选择器。它会惩罚肩部、肘部和腕部分支跳变、速度超限、速度突变、逼近关节限位以及腕部奇异附近 J4/J6 的大幅补偿运动：

```python
from rm65_ik import RM65IKContinuitySelector

selector = RM65IKContinuitySelector(robot, sample_period=1 / 30)

# 第一帧使用机械臂当前角度确定起始解析支路。
solution = selector.solve(first_target, initial_seed=current_joints_deg, seed_degrees=True)

# 后续帧自动使用内部保存的上一帧解、速度和分支标签。
for target in remaining_targets:
    solution = selector.solve(target)
    command_joints_deg = solution.joints_deg
```

每条新轨迹开始前可以调用 `selector.reset(current_joints, degrees=True)`。最近一次选择的总代价和分项可通过 `selector.last_cost`、`selector.last_cost_terms` 查看。

## 参数来源与约定

- MDH 行参数：`a=[0,0,256,0,0,0] mm`、`alpha=[0,90,0,90,-90,90]°`、`d=[240.5,0,0,210,0,d6] mm`。
- 建模角度 = 机械关节角度 + offset，`offset=[0,90,90,0,0,0]°`。
- 限位：J1 ±178°、J2 ±130°、J3 ±135°、J4 ±178°、J5 ±128°、J6 ±360°。
- `d6` 由官方工作半径与前两段长度对应：B=144 mm、6FB=161 mm、6F=172.5 mm；也可用 `RM65Kinematics(model="custom", d6=...)` 显式覆盖。

部署到真机前，请用示教器读取若干已知关节角/末端位姿进行坐标系和型号复核，并在低速模式下验证。

## MATLAB 验证工程

MATLAB 封闭解析解、30 Hz 机械臂动画、六关节角/角速度监视和批量验证位于 `matlab_validation` 文件夹。直接运行 `matlab_validation/run_rm65_validation.m`，详细说明见 `matlab_validation/README_MATLAB.md`。
