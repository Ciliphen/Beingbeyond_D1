#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
D1 全原语实机测试 — 验证 arm / hand / camera 三个原语的所有合约。

测试清单：
  d1_hand:
    1. info            — 查询关节元数据
    2. move_joint      — 设置关节角度（0-1 归一化）
    3. state_joint     — 读取关节角度
    4. state_speed     — 读取关节速度
    5. set_speed       — 设置速度上限
    6. set_torque      — 设置力矩上限

  d1_arm:
    7. info            — 查询关节元数据
    8. move_joint      — 设置绝对关节角度（弧度）

  d1_camera:
    9. info            — 查询相机参数
   10. snapshot        — 取一帧 RGB
   11. depth_snapshot  — 取一帧深度
   12. rgbd            — 取同步 RGB-D
   13. intrinsics      — 查询内参

Usage:
    conda activate bb_d1
    cd ~/Beingbeyond_D1
    python examples/test_all_primitives.py
"""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from arm.d1_arm_primitive import D1ArmPrimitive
from camera.d1_camera_primitive import D1CameraPrimitive
from hand.d1_hand_primitive import D1HandPrimitive

# ═══════════════════════════════════════════════════════════════════════════
# 配置 — 根据实际接线修改
# ═══════════════════════════════════════════════════════════════════════════
HAND_TYPE  = "right"          # "right" or "left"
HAND_CAN   = "can0"           # CAN 接口名
ARMV_DEV    = "/dev/ttyUSB0"   # 机械臂串口
URDF_PATH  = ""               # URDF 路径，默认空
CAM_WIDTH  = 640
CAM_HEIGHT = 480
CAM_FPS    = 30

PASSED = 0
FAILED = 0


def test(name: str, fn, *args, **kwargs):
    """Run a single test and report result."""
    global PASSED, FAILED
    try:
        result = fn(*args, **kwargs)
        PASSED += 1
        print(f"  ✅ {name}")
        return result
    except Exception as e:
        FAILED += 1
        print(f"  ❌ {name} — {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
def test_hand():
    print("─" * 50)
    print("【d1_hand 灵巧手】")
    print("─" * 50)

    with D1HandPrimitive(hand_type=HAND_TYPE, can_iface=HAND_CAN) as hand:
        # 1. info
        info = test("info", hand.info)
        if info:
            print(f"       model={info['model']}, joints={info['num_joints']}, "
                  f"names={info['joint_names']}")

        # 2. move_joint — 开到半闭合
        n = info["num_joints"] if info else 6
        test("move_joint", hand.move_joint, [0.5] * n)
        time.sleep(0.5)

        # 3. state_joint
        pos = test("state_joint", hand.state_joint)
        if pos:
            print(f"       positions={[round(p, 3) for p in pos]}")

        # 4. state_speed
        test("state_speed", hand.state_speed)

        # 5. set_speed
        test("set_speed", hand.set_speed, [0.8] * n)

        # 6. set_torque
        test("set_torque", hand.set_torque, [0.5] * n)

        # 复位
        test("move_joint (open)", hand.move_joint, [0.0] * n)
        time.sleep(0.5)

    print()


def test_arm():
    import math

    print("─" * 50)
    print("【d1_arm 机械臂】")
    print("─" * 50)

    arm = D1ArmPrimitive(dev=ARMV_DEV, urdf_path=URDF_PATH)
    try:
        # 7. info
        info = test("info", arm.info)
        if info:
            jnames = info["joint_names"]
            print(f"       model={info['model']}, joints={info['num_joints']}")
            print(f"       关节名: {jnames}")

        # 8. state_joint — 读取当前角度
        current = test("state_joint", arm.state_joint)
        if not current:
            print("       ⚠ 无法读取关节角度，跳过运动测试")
            return
        n = len(current)
        print(f"       当前臂关节角度 (deg): {[round(math.degrees(p), 2) for p in current]}")

        # 9. 归零 → 逐个关节 ±15° 运动测试
        #     对齐 2_控制头和机械臂.py 的流程：
        #     先全部归零（安全基准位），再逐关节 ±15° 摆动
        #     头部关节保持当前位置不动
        home_rad = [0.0] * n
        print(f"\n       [步骤1] 所有臂关节先归零 (deg): {[0.0]*n}")
        test("move_joint (全部归零)", arm.move_joint, home_rad)
        time.sleep(3.0)

        print(f"\n       [步骤2] 逐个关节 ±15° 测试（其他关节保持 0°）...")
        for idx in range(n):
            name = jnames[idx] if idx < len(jnames) else str(idx)
            print(f"\n       --- 关节 [{idx}] {name} ---")

            # +15°
            target = list(home_rad)
            target[idx] = math.radians(15.0)
            test(f"move_joint ({name} +15°)", arm.move_joint, target)
            time.sleep(2.0)

            # -15°
            target[idx] = math.radians(-15.0)
            test(f"move_joint ({name} -15°)", arm.move_joint, target)
            time.sleep(2.0)

            # 回到 0°
            test(f"move_joint ({name} 回零)", arm.move_joint, list(home_rad))
            time.sleep(1.0)

        print(f"\n       [完成] 所有臂关节已归零，测试结束")

        # 最终状态检查
        final = test("state_joint", arm.state_joint)
        if final:
            print(f"\n       最终臂关节角度 (deg): {[round(math.degrees(p), 2) for p in final]}")

    finally:
        arm.close()

    print()


def test_camera():
    print("─" * 50)
    print("【d1_camera 深度相机】")
    print("─" * 50)

    with D1CameraPrimitive(width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS) as cam:
        # 9. info
        info = test("info", cam.info)
        if info:
            print(f"       model={info['model']}, "
                  f"{info['width']}x{info['height']}@{info['fps']}fps")

        # 10. snapshot
        rgb = test("snapshot", cam.snapshot)
        if rgb is not None:
            import numpy as np
            print(f"       rgb shape={rgb.shape}, dtype={rgb.dtype}, "
                  f"range=[{rgb.min()}, {rgb.max()}]")

        # 11. depth_snapshot
        depth = test("depth_snapshot", cam.depth_snapshot)
        if depth is not None:
            print(f"       depth shape={depth.shape}, dtype={depth.dtype}, "
                  f"range=[{depth.min():.3f}, {depth.max():.3f}]m")

        # 12. rgbd
        result = test("rgbd", cam.rgbd)
        if result is not None:
            r, d = result
            print(f"       rgbd: rgb={r.shape}, depth={d.shape}")

        # 13. intrinsics
        K = test("intrinsics", cam.intrinsics)
        if K:
            print(f"       fx={K['fx']:.1f}, fy={K['fy']:.1f}, "
                  f"cx={K['cx']:.1f}, cy={K['cy']:.1f}")

    print()


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("  D1 全原语实机测试")
    print("=" * 50)
    print()

    test_hand()
    test_arm()
    test_camera()

    print("=" * 50)
    total = PASSED + FAILED
    print(f"  结果: {PASSED}/{total} 通过", end="")
    if FAILED > 0:
        print(f", {FAILED} 失败")
    else:
        print(" ✅")
    print("=" * 50)
