#!/usr/bin/env python3
"""生成一个“实验室内部”风格的 MuJoCo 场景。

场景包含：
  - 四面墙围出的一间实验室，多张实验台/工作台、置物架、方凳、台面器皿等细节；
  - 主工作台上放一红一蓝两把螺丝刀（freejoint，可拾取/推动）；
  - 一名静态工人（人体模型）站在走廊一端，面朝主工作台；
  - 若干“规则运动”的物体：横穿走廊的往复料车、旋转扫掠的挡杆、悬摆的吊摆、
    传送带上往复的包裹——全部由弹簧/重力驱动，阻尼为 0，无需任何控制器即可
    在查看器中自行做无衰减往复运动，供机械狗做动态避障。

用法:
    python3 build_scene.py            # 生成 scene.xml 并校验编译
    python3 build_scene.py --view     # 生成后打开交互式查看器
    python3 build_scene.py --seed 0   # 固定随机种子（仅影响台面小道具摆放）
"""

import argparse
import random

# ---------------------------------------------------------------------------
# 房间与布局常量（单位：米）
# ---------------------------------------------------------------------------
ROOM_X = (-12.0, 3.0)      # 房间在 x 方向的范围
ROOM_Y = (-3.0, 3.0)       # 房间在 y 方向的范围
WALL_H = 2.8               # 墙高
WALL_T = 0.10              # 墙厚（半厚在几何体里用 WALL_T）

TABLE_TOP_Z = 0.74         # 主/靠墙实验台桌面高度
WORKER_X = -10.0           # 工人站立位置的 x（面朝 +x，朝向主工作台）
WORKER_LEG_R = 0.085       # 工人腿半径
LANE_HALF_W = 1.2          # 中央走廊半宽：动态障碍物的活动区

# 从 model/camera 迁移的高台（原为标定桌），摆在东北墙角，不占走廊
CORNER_TABLE_TOP_Z = 1.00          # 桌面高度
CORNER_TABLE_HALF = (0.45, 0.35, 0.02)   # 桌面板半尺寸(x,y,z)
CORNER_TABLE_LEG_HALF = 0.03       # 桌腿半边长
CORNER_TABLE_XY = (2.25, 2.35)     # 东北角：东墙内面 x=2.9、北墙内面 y=2.9

# 棋盘格标定板（平铺在高台桌面上），几何与纹理同 model/camera：
# 纹理 assets/calib_board.png 是 9x7 格 + 白边 → 8x6 内角点，行列不等避免旋转对称歧义。
BOARD_HALF = (0.15, 0.15, 0.004)   # 板半尺寸(x,y,z)
BOARD_TEX = "../calib_board.png"


def bench(name, x, y, yaw=0.0, hx=0.70, hy=0.45, top_z=TABLE_TOP_Z, mat="wood"):
    """一张矩形实验台（桌面 + 四条腿），静态。中心在 (x, y)，可绕 z 转 yaw。"""
    leg_z = (top_z - 0.04) / 2.0          # 腿中心高度（桌面下沿到地面的一半）
    leg_h = (top_z - 0.04) / 2.0          # 腿半高
    lx, ly = hx - 0.05, hy - 0.05         # 腿相对桌面内缩
    return f'''    <body name="{name}" pos="{x} {y} 0" euler="0 0 {yaw}">
      <geom name="{name}_top" type="box" pos="0 0 {round(top_z-0.02,3)}" size="{hx} {hy} 0.02" material="{mat}"/>
      <geom name="{name}_l1" type="box" pos=" {lx}  {ly} {round(leg_z,3)}" size="0.03 0.03 {round(leg_h,3)}" material="metal"/>
      <geom name="{name}_l2" type="box" pos=" {lx} -{ly} {round(leg_z,3)}" size="0.03 0.03 {round(leg_h,3)}" material="metal"/>
      <geom name="{name}_l3" type="box" pos="-{lx}  {ly} {round(leg_z,3)}" size="0.03 0.03 {round(leg_h,3)}" material="metal"/>
      <geom name="{name}_l4" type="box" pos="-{lx} -{ly} {round(leg_z,3)}" size="0.03 0.03 {round(leg_h,3)}" material="metal"/>
    </body>'''


def shelf(name, x, y, yaw=0.0, hx=0.45, hy=0.25, n=3, h=1.6):
    """一个多层置物架，静态。n 层隔板 + 四立柱。"""
    posts = ""
    for sx in (hx - 0.03, -(hx - 0.03)):
        for sy in (hy - 0.03, -(hy - 0.03)):
            posts += (f'      <geom type="box" pos="{round(sx,3)} {round(sy,3)} {round(h/2,3)}" '
                      f'size="0.02 0.02 {round(h/2,3)}" material="metal"/>\n')
    boards = ""
    for k in range(n):
        z = 0.05 + k * (h - 0.1) / (n - 1)
        boards += (f'      <geom type="box" pos="0 0 {round(z,3)}" '
                   f'size="{hx} {hy} 0.015" material="shelf"/>\n')
    return f'''    <body name="{name}" pos="{x} {y} 0" euler="0 0 {yaw}">
{posts}{boards}    </body>'''


def corner_table(name, x, y, top_z=CORNER_TABLE_TOP_Z, half=CORNER_TABLE_HALF,
                 leg_half=CORNER_TABLE_LEG_HALF, mat="table_mat", board=True):
    """从 model/camera 迁移过来的高台（桌面 1.0 m，比实验台高），摆在墙角。
    桌面板 + 4 条方腿；尺寸/材质与 camera 项目的标定桌一致。
    board=True 时在桌面上平铺一块棋盘格标定板（不参与碰撞，纯视觉基准）。"""
    thx, thy, thz = half
    leg_z = round((top_z - 2 * thz) / 2, 4)
    legs = ""
    for sx in (-1, 1):
        for sy in (-1, 1):
            legs += (f'      <geom name="{name}_leg_{sx}_{sy}" type="box" '
                     f'pos="{round(sx*(thx-leg_half-0.02),4)} '
                     f'{round(sy*(thy-leg_half-0.02),4)} {leg_z}" '
                     f'size="{leg_half} {leg_half} {leg_z}" material="{mat}"/>\n')
    board_xml = ""
    if board:
        bx, by, bz = BOARD_HALF
        board_xml = (f'      <geom name="calib_board" type="box" '
                     f'pos="0 0 {round(top_z+bz,4)}" size="{bx} {by} {bz}" '
                     f'material="board_mat" contype="0" conaffinity="0"/>\n')
    head = (f'    <!-- 东北墙角高台（桌面 {top_z:.1f} m，迁自 model/camera 的标定桌）'
            + ('+ 棋盘格标定板 -->' if board else '-->'))
    return f'''{head}
    <body name="{name}" pos="{x} {y} 0">
      <geom name="{name}_top" type="box" pos="0 0 {round(top_z-thz,4)}" size="{thx} {thy} {thz}" material="{mat}"/>
{legs}{board_xml}    </body>'''


def stool(name, x, y):
    """一张圆凳，静态。"""
    return f'''    <body name="{name}" pos="{x} {y} 0">
      <geom type="cylinder" pos="0 0 0.44" size="0.16 0.02" material="shelf"/>
      <geom type="cylinder" pos="0 0 0.22" size="0.025 0.22" material="metal"/>
    </body>'''


def prop(name, x, y, z, kind, rng):
    """台面小道具：烧杯（cylinder）/ 试剂瓶（box）/ 仪器（box），静态点缀。"""
    c = f"{round(rng.uniform(0.4,0.8),2)} {round(rng.uniform(0.5,0.9),2)} {round(rng.uniform(0.6,0.95),2)} 1"
    if kind == "beaker":
        return f'    <geom name="{name}" type="cylinder" pos="{x} {y} {round(z+0.06,3)}" size="0.035 0.06" rgba="{c}"/>'
    if kind == "bottle":
        return f'    <geom name="{name}" type="box" pos="{x} {y} {round(z+0.08,3)}" size="0.04 0.04 0.08" rgba="{c}"/>'
    return f'    <geom name="{name}" type="box" pos="{x} {y} {round(z+0.05,3)}" size="0.09 0.06 0.05" rgba="{c}"/>'


def crate(name, x, y, hx, hy, hz, color):
    """走廊里的静态货箱障碍物（长方体，贴地），供机械狗绕行。"""
    return f'''    <body name="{name}" pos="{x} {y} {round(hz,3)}">
      <geom type="box" size="{hx} {hy} {hz}" rgba="{color}"/>
    </body>'''


# ---------------------------------------------------------------------------
# 规则运动的物体：全部用弹簧/重力驱动，damping=0 → 无衰减往复运动，无需控制器
# ---------------------------------------------------------------------------
def rail_shuttle(name, x, y0, color, rail_ctr, rail_half,
                 mass=6.0, stiffness=22.0, cart_half=0.22):
    """沿 y 轴往复的料车：静态导轨 + 滑块。滑块用 slide 关节，弹簧平衡点设在导轨中心
    rail_ctr，起点在 y0（qpos=0），故 springref = rail_ctr - y0，无阻尼等幅往复于
    [y0, y0+2*springref]。悬空 3 mm 于轨上，避免摩擦损耗。周期 T ≈ 2π·sqrt(m/k)。"""
    springref = round(rail_ctr - y0, 3)
    cov_lo, cov_hi = round(rail_ctr - rail_half, 3), round(rail_ctr + rail_half, 3)
    far = round(y0 + 2 * springref, 3)
    return f'''    <!-- 导轨（静态） -->
    <body name="{name}_rail" pos="{x} {rail_ctr} 0.03">
      <geom type="box" size="0.06 {rail_half} 0.03" material="metal"/>
    </body>
    <!-- 料车（沿 y 往复，扫过整根导轨） -->
    <!-- 导轨中心 y={rail_ctr}、半长{rail_half}（覆盖 {cov_lo}~{cov_hi}），料车半宽{cart_half} -->
    <!-- 起点 y={y0}(qpos=0)，springref={springref} 使平衡点在中心{rail_ctr}，无阻尼等幅往复于 {y0}↔{far}，边缘正好扫到两端 -->
    <body name="{name}" pos="{x} {y0} 0.16">
      <joint name="{name}_j" type="slide" axis="0 1 0" damping="0"
             stiffness="{stiffness}" springref="{springref}"/>
      <geom type="box" size="{cart_half} {cart_half} 0.10" rgba="{color}" mass="{mass}"/>
      <geom type="box" pos="0 0 0.13" size="0.20 0.20 0.03" rgba="{color}" mass="0.3"/>
    </body>'''


def sweep_arm(name, x, y, span, color, stiffness=8.0):
    """绕竖直轴扫掠的挡杆：立柱 + 横杆。横杆用 hinge，弹簧平衡点在 +span/2，
    起始角 0，故在 [0, span] 间扫；body 预转 -span/2 使扫掠对称于 +x 方向。
    杆高约 0.35 m，正好扫机械狗身位。"""
    half = round(span / 2.0, 3)
    return f'''    <!-- 立柱（静态） -->
    <body name="{name}_post" pos="{x} {y} 0">
      <geom type="cylinder" pos="0 0 0.42" size="0.05 0.42" material="metal"/>
    </body>
    <!-- 扫掠横杆 -->
    <body name="{name}" pos="{x} {y} 0.35" euler="0 0 {-half}">
      <joint name="{name}_j" type="hinge" axis="0 0 1" damping="0"
             stiffness="{stiffness}" springref="{half}"/>
      <geom type="capsule" fromto="0 0 0 0.85 0 0" size="0.035" rgba="{color}" mass="1.5"/>
      <geom type="box" pos="0.85 0 0" size="0.04 0.09 0.09" rgba="{color}" mass="0.4"/>
    </body>'''


def pendulum(name, x, y, color, length=0.9, tilt=0.6):
    """悬摆吊摆：门架 + 吊杆 + 摆锤，起始倾角 tilt（rad），靠重力无阻尼摆动。
    摆锤最低点约 (WALL_H 顶架 - length)。"""
    return f'''    <!-- 门架横梁（静态，挂点在 z=2.2） -->
    <body name="{name}_gantry" pos="{x} {y} 0">
      <geom type="box" pos="0 0 2.2" size="0.05 0.6 0.05" material="metal"/>
      <geom type="cylinder" pos="0 -0.55 1.1" size="0.04 1.1" material="metal"/>
      <geom type="cylinder" pos="0  0.55 1.1" size="0.04 1.1" material="metal"/>
    </body>
    <!-- 摆体（绕 x 轴在 y-z 平面内摆动） -->
    <body name="{name}" pos="{x} {y} 2.2" euler="{tilt} 0 0">
      <joint name="{name}_j" type="hinge" axis="1 0 0" damping="0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -{length}" size="0.02" rgba="0.6 0.6 0.62 1" mass="0.2"/>
      <geom type="sphere" pos="0 0 -{length}" size="0.12" rgba="{color}" mass="4.0"/>
    </body>'''


def conveyor(name, x0, y, amp, color, stiffness=30.0):
    """靠墙的传送带 + 往复包裹（环境细节），沿 x 往复。"""
    belt_len = amp + 0.6
    return f'''    <!-- 传送带（静态） -->
    <body name="{name}_belt" pos="{round(x0+amp,3)} {y} 0.35">
      <geom type="box" size="{round(belt_len,3)} 0.35 0.05" material="belt"/>
      <geom type="box" pos="{round(belt_len-0.05,3)} 0 -0.18" size="0.05 0.33 0.18" material="metal"/>
      <geom type="box" pos="-{round(belt_len-0.05,3)} 0 -0.18" size="0.05 0.33 0.18" material="metal"/>
    </body>
    <!-- 包裹（沿 x 往复） -->
    <body name="{name}_pkg" pos="{x0} {y} 0.52">
      <joint name="{name}_j" type="slide" axis="1 0 0" damping="0"
             stiffness="{stiffness}" springref="{amp}"/>
      <geom type="box" size="0.16 0.16 0.12" rgba="{color}" mass="2.0"/>
    </body>'''


def cabinet(name, x, y, yaw=0.0, hx=0.8, hy=0.3, h=1.9, n_shelf=3, shelf_step=0.45):
    """靠墙的柜子（静态）：外壳 + n 层内部挡板 + 双开门（微开，朝 +y 打开）。
    挡板以柜体中高为中心、按 shelf_step 等距分布。门是子 body 但无关节，纯静态造型。"""
    mid = round(h / 2, 3)
    boards = ""
    for k in range(n_shelf):
        z = mid + (k - (n_shelf - 1) / 2) * shelf_step
        boards += (f'      <geom type="box" pos="0 -0.02 {z:.2f}" '
                   f'size="{round(hx-0.02,2)} {round(hy-0.02,2)} 0.02" material="shelf"/>\n')
    return f'''    <!-- 柜子（静态，贴南墙，双开门 + 三层挡板） -->
    <!-- 柜宽 {2*hx}(半{hx}) 深 {2*hy}(半{hy}) 高 {h}；背面贴南墙内表面 y={round(y-hy,2)} -->
    <body name="{name}" pos="{x} {y} 0" euler="0 0 {yaw:g}">
      <!-- 柜体外壳 -->
      <geom type="box" pos="0 0 0.03"   size="{hx} {hy} 0.03" material="shelf"/>  <!-- 底板 -->
      <geom type="box" pos="0 0 {round(h-0.03,3)}"   size="{hx} {hy} 0.03" material="shelf"/>  <!-- 顶板 -->
      <geom type="box" pos="0 {round(-hy+0.01,3)} {mid}" size="{hx} 0.02 {mid}" material="shelf"/> <!-- 背板 -->
      <geom type="box" pos="{round(hx-0.01,3)} 0 {mid}"  size="0.02 {hy} {mid}" material="shelf"/> <!-- 右侧板 -->
      <geom type="box" pos="{round(-hx+0.01,3)} 0 {mid}" size="0.02 {hy} {mid}" material="shelf"/> <!-- 左侧板 -->
      <!-- 三层内部挡板 -->
{boards}      <!-- 双开柜门（微开，朝走廊方向 +y 打开） -->
      <body name="{name}_door_l" pos="{round(-hx+0.01,3)} {round(hy-0.01,3)} {mid}">
        <geom type="box" pos="{round(hx/2-0.02,3)} 0.06 0" size="{round(hx/2-0.02,3)} 0.015 {round(mid-0.05,3)}"
              euler="0 0 0.35" material="metal"/>
      </body>
      <body name="{name}_door_r" pos="{round(hx-0.01,3)} {round(hy-0.01,3)} {mid}">
        <geom type="box" pos="{round(-hx/2+0.02,3)} 0.06 0" size="{round(hx/2-0.02,3)} 0.015 {round(mid-0.05,3)}"
              euler="0 0 -0.35" material="metal"/>
      </body>
    </body>'''


def barrier(name, x, y, hx, hy, hz, color):
    """走廊里的宽挡板（静态障碍物）。原先此处是 sweep_1 扫掠杆，已换成静态挡板。"""
    return f'''    <!-- 宽障碍物（静态挡板） -->
    <body name="{name}" pos="{x} {y} {hz}">
      <geom type="box" size="{hx} {hy} {hz}" rgba="{color}"/>
    </body>'''


def screwdriver(name, color, pos, yaw):
    """一把螺丝刀：彩色手柄 + 金属刀杆，带 freejoint 可拾取/推动，躺在桌面上。"""
    return f'''    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" euler="0 0 {yaw}">
      <freejoint/>
      <geom name="{name}_handle" type="cylinder" fromto="-0.045 0 0 0.03 0 0"
            size="0.014" rgba="{color}" condim="6" friction="1 0.02 0.02"/>
      <geom name="{name}_shaft" type="cylinder" fromto="0.03 0 0 0.115 0 0"
            size="0.004" rgba="0.75 0.75 0.78 1" condim="6" friction="1 0.02 0.02"/>
    </body>'''


def walls():
    """四面墙（静态），围出实验室。"""
    x0, x1 = ROOM_X
    y0, y1 = ROOM_Y
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = (x1 - x0) / 2, (y1 - y0) / 2
    z = WALL_H / 2
    return f'''    <geom name="wall_S" type="box" pos="{cx} {y0} {z}" size="{hx} {WALL_T} {z}" material="wall"/>
    <geom name="wall_N" type="box" pos="{cx} {y1} {z}" size="{hx} {WALL_T} {z}" material="wall"/>
    <geom name="wall_W" type="box" pos="{x0} {cy} {z}" size="{WALL_T} {hy} {z}" material="wall"/>
    <geom name="wall_E" type="box" pos="{x1} {cy} {z}" size="{WALL_T} {hy} {z}" material="wall"/>'''


def build_xml(rng):
    sd_z = TABLE_TOP_Z + 0.016
    red = screwdriver("screwdriver_red", "0.85 0.15 0.15 1", (0.18, -0.18, sd_z), 0.4)
    blue = screwdriver("screwdriver_blue", "0.15 0.25 0.85 1", (-0.10, 0.20, sd_z), -1.1)

    # 静态家具 -------------------------------------------------------------
    furniture = [
        bench("bench_main", 0.0, 0.0),                       # 主工作台（放螺丝刀）
        bench("bench_S1", -8.5, -2.55, hx=0.8, hy=0.35),     # 南墙实验台
        shelf("shelf_W", -11.4, 1.4, hy=0.22),               # 西墙置物架
        corner_table("corner_table", *CORNER_TABLE_XY),      # 东北墙角高台（迁自 camera）
    ]
    # 台面小道具 -----------------------------------------------------------
    props = []
    kinds = ["beaker", "bottle", "instrument"]
    for j in range(rng.randint(2, 3)):
        px = round(-8.5 + rng.uniform(-0.65, 0.65), 3)
        py = round(-2.55 + rng.uniform(-0.23, 0.23), 3)
        props.append(prop(f"prop_{j}", px, py, TABLE_TOP_Z, rng.choice(kinds), rng))

    # 静态障碍物（走廊里，供机械狗绕行；避开扫掠杆约 1 m 半径的活动圆） --------
    statics = [
        crate("crate_1", -2.2, 0.7, 0.30, 0.25, 0.25, "0.76 0.60 0.42 1"),
        crate("crate_2", -6.5, 0.85, 0.28, 0.28, 0.22, "0.55 0.58 0.62 1"),  # 移开，避让机械狗(x≈-8.5)
        crate("crate_3", -4.7, 0.9, 0.24, 0.20, 0.30, "0.35 0.62 0.45 1"),
    ]

    # 规则运动的物体（当前仅一个：横穿走廊的往复料车） --------------------------
    # 注：x=-6.0 处原是 sweep_1 旋转扫掠杆，已换成同位置的静态 barrier_1（见 fixtures）。
    # 需要动态扫掠障碍时，把 sweep_arm("sweep_1", -6.0, 0.0, 1.6, ...) 加回本列表，
    # 并从 fixtures 去掉 barrier_1、在 <contact> 里补 sweep_1/sweep_1_post 的 exclude。
    movers = [
        rail_shuttle("shuttle_a", -3.5, -0.08, "0.90 0.45 0.15 1",
                     rail_ctr=1.25, rail_half=1.35, mass=6.0, stiffness=22.0),
    ]

    # 走廊/靠墙的静态构件（柜子、宽挡板） -------------------------------------
    fixtures = [
        cabinet("cabinet", -3.5, -2.6),
        barrier("barrier_1", -6.0, 0.0, 0.12, 1.2, 0.6, "0.85 0.75 0.15 1"),
    ]

    furniture_xml = "\n".join(furniture)
    props_xml = "\n".join(props)
    statics_xml = "\n".join(statics)
    movers_xml = "\n".join(movers)
    fixtures_xml = "\n".join(fixtures)

    head = f'''<mujoco model="lab_scene">
  <option timestep="0.004" integrator="implicitfast"/>
  <compiler angle="radian"/>

  <visual>
    <global offwidth="1920" offheight="1080" elevation="-22" azimuth="135"/>
    <rgba haze="0.14 0.16 0.20 1"/>
    <map force="0.1" zfar="40"/>
    <quality shadowsize="4096"/>
  </visual>

  <statistic center="-4.5 0 0.7" extent="13"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".25 .3 .38" rgb2="0 0 0" width="32" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1=".18 .20 .24" rgb2=".24 .27 .32"/>
    <material name="grid" texture="grid" texrepeat="8 8" texuniform="true" reflectance=".15"/>
    <material name="wood"   rgba="0.72 0.53 0.34 1" reflectance="0.05"/>
    <material name="metal"  rgba="0.55 0.57 0.60 1" reflectance="0.35"/>
    <material name="shelf"  rgba="0.80 0.82 0.85 1" reflectance="0.1"/>
    <material name="belt"   rgba="0.20 0.20 0.22 1" reflectance="0.05"/>
    <material name="wall"   rgba="0.86 0.87 0.88 1" reflectance="0.04"/>
    <material name="worker" rgba="0.90 0.55 0.15 1"/>
    <material name="skin"   rgba="0.85 0.68 0.55 1"/>
    <material name="table_mat" rgba="0.55 0.42 0.28 1" reflectance="0.05"/>
    <texture name="board_tex" type="2d" file="{BOARD_TEX}"/>
    <material name="board_mat" texture="board_tex" texrepeat="1 1" texuniform="false"/>
  </asset>

  <default>
    <geom condim="3" friction="0.9 0.05 0.005" solref="0.008 1"/>
  </default>

  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" condim="3"/>
    <light name="top1" pos="-3 0 3.5" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <light name="top2" pos="-8 0 3.5" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <light name="side" pos="-4 -3 3" dir="0.3 1 -1.2" diffuse="0.4 0.4 0.4"/>
    <camera name="overview" pos="-4.5 -11 10.5" xyaxes="1 0 0 0 0.68 0.73"/>
    <camera name="top" pos="-4.5 0 12" xyaxes="1 0 0 0 1 0"/>

{walls()}
'''

    body = f'''
    <!-- ================= 静态家具 ================= -->
{furniture_xml}

    <!-- ================= 台面小道具 ================= -->
{props_xml}

    <!-- ================= 静态障碍物（走廊里） ================= -->
{statics_xml}

    <!-- ================= 螺丝刀（主工作台上） ================= -->
{red}
{blue}

    <!-- ================= 工人（静态站姿，面朝主工作台） ================= -->
    <body name="worker" pos="{WORKER_X} 0 0">
      <geom name="w_leg_l" type="capsule" fromto="0  0.11 0.04 0  0.11 0.46" size="{WORKER_LEG_R}" material="worker"/>
      <geom name="w_leg_r" type="capsule" fromto="0 -0.11 0.04 0 -0.11 0.46" size="{WORKER_LEG_R}" material="worker"/>
      <geom name="w_torso" type="capsule" fromto="0 0 0.46 0 0 0.98" size="0.14" material="worker"/>
      <geom name="w_head"  type="sphere"  pos="0 0 1.15" size="0.11" material="skin"/>
      <geom name="w_arm_l" type="capsule" fromto="0 0.17 0.92 0.02 0.20 0.55" size="0.05" material="worker"/>
      <geom name="w_arm_r" type="capsule" fromto="0 -0.17 0.92 0.02 -0.20 0.55" size="0.05" material="worker"/>
      <geom name="w_hand_l" type="sphere" pos="0.02 0.20 0.50" size="0.045" material="skin"/>
      <geom name="w_hand_r" type="sphere" pos="0.02 -0.20 0.50" size="0.045" material="skin"/>
    </body>

    <!-- ================= 规则运动的物体（弹簧/重力驱动，无衰减往复） ================= -->
{movers_xml}
{fixtures_xml}
  </worldbody>

  <!-- 每个运动部件与其自身静态支架不参与碰撞（支架只是结构，不是障碍物） -->
  <contact>
    <exclude body1="shuttle_a" body2="shuttle_a_rail"/>
  </contact>
</mujoco>
'''
    return head + body


def main():
    ap = argparse.ArgumentParser(description="生成实验室内景 MuJoCo 场景")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（仅影响台面小道具摆放）")
    ap.add_argument("--out", default="scene.xml", help="输出 XML 文件路径")
    ap.add_argument("--view", action="store_true", help="生成后打开交互式查看器")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    xml = build_xml(rng)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"已写出 {args.out}（种子={args.seed}）")

    import mujoco
    model = mujoco.MjModel.from_xml_path(args.out)
    print(f"模型编译成功：{model.nbody} 个 body，{model.ngeom} 个 geom，{model.njnt} 个关节。")

    if args.view:
        import mujoco.viewer
        data = mujoco.MjData(model)
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()



