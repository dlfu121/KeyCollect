# RealMan RM65-6F 完整运动学与动力学参数  
## Codex / MuJoCo 植入规范

> 适用目标：已有 MuJoCo 机械臂几何模型，需要补齐运动学、惯性参数、IK 与一致性校验。  
> 型号：**RealMan RM65-6F**  
> 单位约定：长度 `m`，角度 `rad`，质量 `kg`，惯量 `kg·m²`

---

## 1. 重要型号说明

本文专门针对 **RM65-6F**。

不要使用 RM65-B 的末端尺寸：

- RM65-B：`d6 = 0.144 m`
- **RM65-6F：`d6 = 0.1725 m`**

因此 RM65-6F 的零位法兰高度、工作半径与 RM65-B 不同。

---

# 2. 基础规格

| 参数 | RM65-6F |
|---|---:|
| 自由度 | 6 |
| 构型 | 6R 串联机械臂 |
| 额定负载 | 5 kg |
| 自重 | 约 7.3 kg |
| 工作半径 | 约 638.5 mm |
| 重复定位精度 | ±0.05 mm |
| TCP 最大线速度 | ≤ 1.8 m/s |
| 六维力传感器量程 | 200 N / 7 N·m |
| 六维力精度 | ±0.5% FS |

---

# 3. 关节限位与速度

| Joint | 下限 deg | 上限 deg | 下限 rad | 上限 rad | 最大速度 deg/s | 最大速度 rad/s |
|---|---:|---:|---:|---:|---:|---:|
| J1 | -178 | 178 | -3.106686069 | 3.106686069 | 180 | 3.141592654 |
| J2 | -130 | 130 | -2.268928028 | 2.268928028 | 180 | 3.141592654 |
| J3 | -135 | 135 | -2.356194490 | 2.356194490 | 225 | 3.926990817 |
| J4 | -178 | 178 | -3.106686069 | 3.106686069 | 225 | 3.926990817 |
| J5 | -128 | 128 | -2.234021443 | 2.234021443 | 225 | 3.926990817 |
| J6 | -360 | 360 | -6.283185307 | 6.283185307 | 225 | 3.926990817 |

旧版官方 URDF 中部分关节速度统一写成 `3.14 rad/s`。  
工程实现建议按当前官方本体规格使用上表速度。

---

# 4. Modified D-H 运动学模型

RM65 使用 **Modified Denavit-Hartenberg（Craig MDH）**。

单节变换定义为：

```text
T_(i-1,i)
=
Rx(alpha_(i-1))
*
Tx(a_(i-1))
*
Rz(theta_i)
*
Tz(d_i)
```

其中：

```text
theta_i = q_i + offset_i
```

## 4.1 RM65-6F MDH 参数

| i | a(i-1) [m] | alpha(i-1) [rad] | d(i) [m] | offset [rad] |
|---|---:|---:|---:|---:|
| 1 | 0 | 0 | 0.2405 | 0 |
| 2 | 0 | +π/2 | 0 | +π/2 |
| 3 | 0.256 | 0 | 0 | +π/2 |
| 4 | 0 | +π/2 | 0.210 | 0 |
| 5 | 0 | -π/2 | 0 | 0 |
| 6 | 0 | +π/2 | 0.1725 | 0 |

最关键的参数：

```python
d1 = 0.2405
a2 = 0.256
d4 = 0.210
d6 = 0.1725
```

以及：

```python
offset2 = +pi/2
offset3 = +pi/2
```

---

# 5. Codex 可直接使用的运动学常量

```python
import numpy as np

RM65_6F_DOF = 6

RM65_6F_A = np.array([
    0.0,
    0.0,
    0.256,
    0.0,
    0.0,
    0.0,
], dtype=float)

RM65_6F_ALPHA = np.array([
    0.0,
    np.pi / 2,
    0.0,
    np.pi / 2,
    -np.pi / 2,
    np.pi / 2,
], dtype=float)

RM65_6F_D = np.array([
    0.2405,
    0.0,
    0.0,
    0.210,
    0.0,
    0.1725,
], dtype=float)

RM65_6F_OFFSET = np.array([
    0.0,
    np.pi / 2,
    np.pi / 2,
    0.0,
    0.0,
    0.0,
], dtype=float)

RM65_6F_Q_MIN = np.deg2rad([
    -178.0,
    -130.0,
    -135.0,
    -178.0,
    -128.0,
    -360.0,
])

RM65_6F_Q_MAX = np.deg2rad([
    178.0,
    130.0,
    135.0,
    178.0,
    128.0,
    360.0,
])

RM65_6F_QD_MAX = np.deg2rad([
    180.0,
    180.0,
    225.0,
    225.0,
    225.0,
    225.0,
])
```

---

# 6. 官方 URDF 关节坐标链

用于校准现有 MJCF。

| Joint | Parent | Child | origin xyz [m] | origin rpy [rad] | axis |
|---|---|---|---|---|---|
| joint_1 | base_link | link_1 | `0 0 0.2405` | `0 0 0` | `0 0 1` |
| joint_2 | link_1 | link_2 | `0 0 0` | `1.5708 -1.5708 0` | `0 0 1` |
| joint_3 | link_2 | link_3 | `0.256 0 0` | `0 0 1.5708` | `0 0 1` |
| joint_4 | link_3 | link_4 | `0 -0.210 0` | `1.5708 0 0` | `0 0 1` |
| joint_5 | link_4 | link_5 | `0 0 0` | `-1.5708 0 0` | `0 0 1` |
| joint_6 | link_5 | link_6 | `0 -0.1725 0` | `1.5708 0 0` | `0 0 1` |

注意：URDF 中所有关节轴都定义为各自局部坐标系 `z` 轴。  
现有 MuJoCo 模型不需要机械地照抄 `axis="0 0 1"`，只要实际世界坐标中的转轴方向一致即可。

---

# 7. 正运动学 FK

## 7.1 MDH 单节变换

```python
def rm65_6f_mdh_transform(a, alpha, d, theta):
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ct = np.cos(theta)
    st = np.sin(theta)

    return np.array([
        [ct,       -st,       0.0,      a],
        [st*ca,  ct*ca,       -sa, -d*sa],
        [st*sa,  ct*sa,        ca,  d*ca],
        [0.0,       0.0,      0.0,    1.0],
    ], dtype=float)
```

## 7.2 FK

```python
def rm65_6f_fk(q):
    q = np.asarray(q, dtype=float)

    if q.shape != (6,):
        raise ValueError("RM65-6F q must have shape (6,)")

    theta = q + RM65_6F_OFFSET

    T = np.eye(4)

    for i in range(6):
        T = T @ rm65_6f_mdh_transform(
            RM65_6F_A[i],
            RM65_6F_ALPHA[i],
            RM65_6F_D[i],
            theta[i],
        )

    return T
```

---

# 8. FK 零位自检

输入：

```python
q = np.zeros(6)
```

理论上：

```text
x ≈ 0
y ≈ 0
z ≈ 0.879 m
```

因为：

```text
0.2405 + 0.256 + 0.210 + 0.1725 = 0.879
```

理论零位法兰变换约为：

```python
T_base_flange = np.array([
    [-1.0,  0.0, 0.0, 0.0],
    [ 0.0, -1.0, 0.0, 0.0],
    [ 0.0,  0.0, 1.0, 0.879],
    [ 0.0,  0.0, 0.0, 1.0],
])
```

如果 MuJoCo 中 `qpos = 0` 后 flange 不是这个位姿，优先检查：

1. MJCF joint axis
2. 关节正方向
3. body 预旋转
4. MuJoCo 零位与 RealMan 零位是否一致

不要直接修改官方 MDH 参数。

---

# 9. MuJoCo 关节映射层

强烈建议不要直接假设：

```text
q_mujoco == q_rm65
```

应该保留：

```python
q_rm65 = joint_sign * q_mujoco + joint_zero_offset
```

初始配置：

```python
RM65_6F_MUJOCO_SIGN = np.array([
    1, 1, 1, 1, 1, 1
], dtype=float)

RM65_6F_MUJOCO_ZERO = np.array([
    0, 0, 0, 0, 0, 0
], dtype=float)
```

接口：

```python
def mujoco_to_rm65_q(q_mj):
    q_mj = np.asarray(q_mj, dtype=float)

    return (
        RM65_6F_MUJOCO_SIGN * q_mj
        + RM65_6F_MUJOCO_ZERO
    )
```

逐轴校准方法：

```text
q = [0,0,0,0,0,0]
J1 +10°
J2 +10°
J3 +10°
J4 +10°
J5 +10°
J6 +10°
```

分别比较：

```text
MDH FK
vs
MuJoCo flange/site pose
```

若方向相反：

```python
joint_sign[i] = -1
```

若存在固定零位角差：

```python
joint_zero_offset[i] = delta
```

不要修改 `RM65_6F_OFFSET` 来补偿 MJCF 建模差异。

---

# 10. TCP 与工具坐标系

RM65 本体 FK 只计算到：

```text
base
 ->
J1
 ->
...
 ->
J6
 ->
flange
```

工具单独定义：

```text
flange
 ->
gripper
 ->
TCP
```

使用：

```python
T_base_tcp = T_base_flange @ T_flange_tcp
```

推荐接口：

```python
def rm65_6f_fk_tcp(q, T_flange_tcp=None):
    T_base_flange = rm65_6f_fk(q)

    if T_flange_tcp is None:
        return T_base_flange

    return T_base_flange @ T_flange_tcp
```

不要把夹爪长度直接修改进 `d6`。

---

# 11. 动力学参数说明

官方动力学表提供：

- Link 质量
- 质心位置
- 惯量参数

当前官方 CAD 表中的惯量以 Link 坐标系原点描述，而 MuJoCo `<inertial>` 需要的是：

- COM 位置
- COM 处惯量

因此必须使用平行移轴定理：

```text
I_COM
=
I_origin
-
m * ( ||r||² E - r r^T )
```

单位转换：

```text
COM:
mm -> m
乘 1e-3

Inertia:
kg·mm² -> kg·m²
乘 1e-6
```

---

# 12. 官方 CAD 原始动力学数据

> 下表惯量为官方原始 CAD 数据，用于追溯。  
> 不建议直接原样写进 MuJoCo `<inertial>`。

| Link | mass kg | COM x mm | COM y mm | COM z mm | Lxx | Lxy | Lxz | Lyy | Lyz | Lzz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| link_1 | 1.510 | 0.491 | 7.803 | -10.744 | 2928.466 | -32.630 | -5.816 | 2506.350 | 47.925 | 1756.017 |
| link_2 | 1.653 | 183.722 | 0.103 | -1.665 | 1711.553 | -38.271 | 2314.910 | 70514.722 | 6.507 | 70036.186 |
| link_3 | 0.726 | 0.029 | -90.105 | 4.039 | 7259.884 | 2.994 | -0.314 | 371.872 | 44.451 | 7228.758 |
| link_4 | 0.671 | 0.007 | -9.486 | -8.041 | 794.014 | -0.821 | -0.655 | 596.235 | -34.785 | 486.228 |
| link_5 | 0.647 | 0.032 | -83.769 | 2.326 | 5375.604 | 2.665 | -0.304 | 285.265 | 14.235 | 5359.769 |
| link_6 | 0.248 | -0.426 | 0.237 | -27.223 | 308.844 | -3.781 | -1.468 | 304.616 | 0.888 | 122.620 |

惯量单位：

```text
kg·mm²
```

---

# 13. 推荐给 MuJoCo 使用的 COM 动力学参数

以下数据已经换算为：

```text
mass      kg
COM       m
inertia   kg·m²
```

并已经通过平行移轴定理转换到 COM。

| Body | mass kg | COM x m | COM y m | COM z m | Ixx | Iyy | Izz | Ixy | Ixz | Iyz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_link | 1.862000 | -0.000499870 | -0.000052709 | 0.060019000 | 0.0017232 | 0.0017051 | 0.00090158 | -0.0000031058 | 0.000037924 | -0.0000013691 |
| link_1 | 1.510000 | 0.000491000 | 0.007803000 | -0.010744000 | 0.002662574 | 0.002331624 | 0.001663499 | -0.000038414 | 0.000002144 | -0.000078728 |
| link_2 | 1.653000 | 0.183722000 | 0.000103000 | -0.001665000 | 0.001706951 | 0.014714618 | 0.014227953 | -0.000007004 | 0.001809240 | 0.000006223 |
| link_3 | 0.726000 | 0.000029000 | -0.090105000 | 0.004039000 | 0.001356100 | 0.000360029 | 0.001333686 | 0.000001097 | -0.000000399 | 0.000308554 |
| link_4 | 0.671000 | 0.000007000 | -0.009486000 | -0.008041000 | 0.000690278 | 0.000552846 | 0.000425862 | -0.000000866 | -0.000000693 | -0.000085974 |
| link_5 | 0.647000 | 0.000032000 | -0.083769000 | 0.002326000 | 0.000830942 | 0.000281764 | 0.000816973 | 0.000000930 | -0.000000352 | -0.000111817 |
| link_6 | 0.248000 | -0.000426000 | 0.000237000 | -0.027223000 | 0.000125056 | 0.000120739 | 0.000122560 | -0.000003806 | 0.000001407 | -0.000000712 |

> 建议 Codex 在真正替换 MJCF 之前，再对这些数值做一次自动正定性检查。

---

# 14. MuJoCo inertial 写法

MuJoCo `fullinertia` 顺序：

```text
Ixx Iyy Izz Ixy Ixz Iyz
```

示例：

```xml
<inertial
    pos="0.000491000 0.007803000 -0.010744000"
    mass="1.510000"
    fullinertia="
        0.002662574
        0.002331624
        0.001663499
       -0.000038414
        0.000002144
       -0.000078728" />
```

RM65-6F `link_6`：

```xml
<inertial
    pos="-0.000426000 0.000237000 -0.027223000"
    mass="0.248000"
    fullinertia="
        0.000125056
        0.000120739
        0.000122560
       -0.000003806
        0.000001407
       -0.000000712" />
```

---

# 15. 动力学版本差异

当前官方 CAD 参数页与较旧 `rm_models` RM65-6F URDF 在少量惯性参数上存在差异。

尤其：

```text
link_6
```

旧 URDF 质量约：

```text
0.14434 kg
```

当前 RM65-6F CAD 数据：

```text
0.248 kg
```

建议：

> 不要把旧 URDF 的 link_6 mass、COM 与当前 CAD link_6 inertia 混搭。

本文推荐使用同一套当前 RM65-6F CAD 数据进行：

```text
mass
COM
inertia
```

一致建模。

这样整臂总质量约：

```text
7.317 kg
```

与官方六维力版本约 `7.3 kg` 自重基本一致。

---

# 16. 动力学模型

MuJoCo 会自动根据：

- body mass
- COM
- inertia
- joint definition
- gravity
- contact

构造机械臂动力学：

```text
M(q) qdd
+
C(q,qd) qd
+
g(q)
+
tau_friction

=

tau_actuator
+
J(q)^T F_external
```

不需要 Codex 手工重新推导整套：

```text
M(q)
C(q,qd)
g(q)
```

---

# 17. 官方未明确给出的动力学参数

以下参数不要让 Codex 猜：

```text
joint armature
motor rotor inertia
viscous damping
Coulomb friction
motor electrical parameters
gear efficiency
backlash
```

建议全部配置化，例如：

```python
JOINT_DAMPING = [...]
JOINT_FRICTION = [...]
JOINT_ARMATURE = [...]
```

如果暂无实机辨识数据：

- 保留现有 MuJoCo 数值
- 或设为较小稳定值
- 明确标注为“仿真调参值”
- 不写成“官方参数”

---

# 18. Jacobian 与 IK

第一版推荐使用数值 Jacobian + Damped Least Squares。

公式：

```text
dq
=
J^T
*
(J J^T + lambda² I)^(-1)
*
error
```

示意：

```python
dq = J.T @ np.linalg.solve(
    J @ J.T + damping**2 * np.eye(J.shape[0]),
    error
)

q_next = q + step_size * dq

q_next = np.clip(
    q_next,
    RM65_6F_Q_MIN,
    RM65_6F_Q_MAX
)
```

---

# 19. IK 必须支持

```text
position-only IK
position + orientation IK
joint limits
velocity limits
dq step limit
damping
maximum iterations
convergence flag
residual error
Jacobian condition number
```

推荐输出：

```python
IKResult(
    q=...,
    converged=...,
    iterations=...,
    position_error=...,
    orientation_error=...,
    condition_number=...
)
```

---

# 20. 奇异状态保护

至少注意：

## 腕部奇异

```text
q5 ≈ 0
```

## 肘部伸直奇异

```text
q3 ≈ 0
```

全零：

```python
q = [0, 0, 0, 0, 0, 0]
```

适合做 FK 自检，但不推荐作为长期工作姿态。

---

# 21. MuJoCo 控制链建议

运动学模块只负责：

```text
Target Cartesian Pose
        ↓
RM65 IK
        ↓
q_target
```

已有控制器继续负责：

```text
q_target
    ↓
trajectory generation
    ↓
position / velocity / torque controller
    ↓
MuJoCo actuator
```

不要让 IK 直接写 actuator torque。

---

# 22. 六维力传感器

RM65-6F 包含六维力传感器。

如果 MuJoCo 中需要模拟：

```text
Fx Fy Fz
Mx My Mz
```

建议建立独立传感器 site：

```text
link_6
  ↓
force_sensor_site
  ↓
flange
```

必须明确：

```text
sensor frame
flange frame
TCP frame
```

三者的固定变换关系。

不要默认六维力传感器坐标系和 TCP 坐标系完全相同。

---

# 23. Codex 推荐文件结构

```text
rm65_6f_params.py
    官方参数
    关节限位
    MDH
    dynamics

rm65_6f_kinematics.py
    MDH transform
    FK
    TCP FK
    Jacobian
    IK

rm65_6f_mujoco_adapter.py
    qpos index
    joint sign
    zero offset
    flange site access

rm65_6f_validation.py
    zero FK
    single-joint test
    random FK comparison
    FK-IK roundtrip
```

---

# 24. 自动测试

必须加入：

## 24.1 零位 FK

```python
def test_fk_zero():
    q = np.zeros(6)

    T = rm65_6f_fk(q)

    np.testing.assert_allclose(
        T[:3, 3],
        [0.0, 0.0, 0.879],
        atol=1e-6
    )
```

## 24.2 关节限位

```text
test_joint_limits()
```

## 24.3 单关节方向

```text
test_joint1_positive()
...
test_joint6_positive()
```

## 24.4 MuJoCo 对齐

随机生成：

```text
100 组合法 q
```

比较：

```text
MDH T_base_flange
vs
MuJoCo flange site pose
```

## 24.5 FK-IK roundtrip

```text
q_random
    ↓
FK
    ↓
target pose
    ↓
IK
    ↓
q_solution
    ↓
FK
```

最终比较末端位姿误差。

## 24.6 动力学惯量检查

每个惯量矩阵：

```python
np.all(np.linalg.eigvalsh(I) > 0)
```

必须通过。

---

# 25. 可以直接交给 Codex 的最终执行指令

```text
任务：
在现有 MuJoCo 工程中植入 RealMan RM65-6F 的完整运动学和动力学参数。

不要重建当前机器人 mesh、环境、移动底盘、SLAM 或其他模块。

机械臂型号：
RealMan RM65-6F

====================
一、运动学
====================

自由度：
6

Craig Modified-DH：

T_(i-1,i)
=
Rx(alpha_(i-1))
Tx(a_(i-1))
Rz(theta_i)
Tz(d_i)

theta_i = q_rm65[i] + offset[i]

a =
[
0,
0,
0.256,
0,
0,
0
]

alpha =
[
0,
pi/2,
0,
pi/2,
-pi/2,
pi/2
]

d =
[
0.2405,
0,
0,
0.210,
0,
0.1725
]

offset =
[
0,
pi/2,
pi/2,
0,
0,
0
]

注意：
RM65-6F 的 d6 是 0.1725 m，
不得使用 RM65-B 的 0.144 m。

关节限位 deg：

q_min =
[-178,-130,-135,-178,-128,-360]

q_max =
[178,130,135,178,128,360]

最大速度 deg/s：

qd_max =
[180,180,225,225,225,225]

内部全部转换为 rad。

====================
二、MuJoCo 关节映射
====================

不要假设：

q_mujoco == q_rm65

必须增加：

q_rm65
=
joint_sign * q_mujoco
+
joint_zero_offset

默认：

joint_sign =
[1,1,1,1,1,1]

joint_zero_offset =
[0,0,0,0,0,0]

通过逐轴 +10 deg 与 MDH FK 对比自动校准。

不要修改官方 MDH offset 来补偿 MuJoCo 零位。

====================
三、FK 零位测试
====================

q =
[0,0,0,0,0,0]

必须得到：

position approximately =
[0,0,0.879] m

rotation approximately =

[-1  0  0
  0 -1  0
  0  0  1]

====================
四、动力学
====================

使用本文“推荐给 MuJoCo 使用的 COM 动力学参数”。

单位：

mass = kg
COM = m
inertia = kg*m^2

不要把官方原始 CAD L 惯量直接写入 MuJoCo。

MuJoCo fullinertia 顺序：

Ixx Iyy Izz Ixy Ixz Iyz

不要自行杜撰：

armature
motor rotor inertia
joint damping
Coulomb friction
gear efficiency

已有值保留并配置化。

====================
五、TCP
====================

RM65 本体 FK 只算到 flange。

T_base_tcp
=
T_base_flange
@
T_flange_tcp

T_flange_tcp 必须为配置项。

不要把夹爪长度加进 d6。

====================
六、IK
====================

实现 Damped Least Squares IK。

必须支持：

position-only
position+orientation
joint limits
velocity limits
dq limit
damping
convergence result
iteration count
position residual
orientation residual
Jacobian condition number

IK 只输出 q_target。

不得直接修改 MuJoCo actuator torque。

====================
七、自动测试
====================

必须实现：

test_fk_zero()
test_joint_limits()
test_each_joint_positive()
test_fk_vs_mujoco_random()
test_fk_ik_roundtrip()
test_tcp_transform()
test_inertia_positive_definite()

随机 FK vs MuJoCo 至少测试 100 组合法关节角。

====================
八、建议文件
====================

rm65_6f_params.py
rm65_6f_kinematics.py
rm65_6f_mujoco_adapter.py
rm65_6f_validation.py

不要改动当前 SLAM、导航、底盘与环境逻辑。
```

---

# 26. 参考来源

RealMan 官方 RM65 本体参数 / D-H：

https://develop.realman-robotics.com/en/robot/robotParameter/RM65OntologyParameters/

RealMan 官方模型仓库：

https://github.com/RealManRobot/rm_models

RM65-6F 官方 URDF：

https://github.com/RealManRobot/rm_models/blob/main/RM65/urdf/RM65-6F/urdf/RM65-6F.urdf

RealMan 官方 RM_API2：

https://github.com/RealManRobot/RM_API2

---

# 27. 最终实现原则

Codex 应遵守：

```text
官方几何参数
        ↓
MDH FK
        ↓
MuJoCo joint mapping
        ↓
MuJoCo flange pose
        ↓
自动一致性验证
```

只有：

```text
MDH FK
≈
MuJoCo 实际 flange pose
```

在全零、逐轴运动以及大量随机姿态下都一致，才能认为 RM65-6F 的运动学植入完成。

动力学方面则要求：

```text
mass
COM
inertia
```

必须来自同一版本的数据集，禁止混搭旧 URDF 与新 CAD 的 link_6 参数。
