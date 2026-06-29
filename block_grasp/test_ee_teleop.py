#!/usr/bin/env python3
"""
Simple EE teleop — palm faces down, WASD slides parallel to table.

Based on examples_中文/6_键盘遥操.py patterns:
- Safe initial posture, workspace clamping, incremental IK
- Terminal raw-mode input (no OpenCV window needed)

W/S → X± (forward/back)   A/D → Y± (left/right)   Q/E → Z± (up/down)
R → reset to start   ESC → quit

Usage:
    conda activate bb_d1
    cd ~/Beingbeyond_D1
    python block_grasp/test_ee_teleop.py
"""
import math
import select
import sys
import termios
import time
import tty

import numpy as np
from scipy.spatial.transform import Rotation as R

from beingbeyond_d1_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig
from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
from beingbeyond_d1_sdk.head_arm import HeadArmRobot

STEP = 0.01   # 1cm
Z_STEP = 0.01
MAX_OFFSET = np.array([0.10, 0.10, 0.10])
IK_FAIL_THR = 0.10  # m — relaxed for small workspace


def _rot_x(a): c,s = math.cos(a), math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)
def _rot_y(a): c,s = math.cos(a), math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)
def _rot_z(a): c,s = math.cos(a), math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)
def _ortho(M): U,_,Vt = np.linalg.svd(M); return U @ Vt


def _getch(timeout=0.01):
    dr, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if dr else None


def _raw_mode():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return fd, old


def _restore(fd, old):
    if fd is not None:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    print("\033[91m⚠ 急停按钮请保持触手可及！\033[0m\n")

    urdf = get_default_urdf_path()
    kin = D1Kinematics(D1KinematicsConfig(urdf_path=urdf))
    robot = HeadArmRobot(urdf_path=urdf, dev="/dev/ttyUSB0", baudrate=1_000_000)

    # ── Safe initial posture ──────────────────────────────────────────
    print("[Init] Moving to safe posture ...")
    q_init_deg = [0, 0,  0, -60, 60,  0, 0, 0]
    q_init = np.radians(q_init_deg)
    robot.set_positions(q_init)
    robot.wait_until_reached(q_init, active_joint_indices=range(8))
    time.sleep(0.3)

    # ── Read initial EE pose ──────────────────────────────────────────
    q = np.asarray(robot.get_positions(), dtype=float)
    q_head, q_arm = kin.split_q(q)
    T0 = kin.ee_in_base(q_head, q_arm)
    p0 = T0[:3, 3].copy()
    R0 = T0[:3, :3].copy()

    p_des = p0.copy()
    R_des = R0.copy()
    print(f"       EE: ({p0[0]:.3f}, {p0[1]:.3f}, {p0[2]:.3f})")

    # ── Help ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  W/S X±  |  A/D Y±  |  Q/E Z±  |  R reset  |  ESC quit")
    print(f"  Step={STEP*100:.0f}cm  max=({MAX_OFFSET[0]*100:.0f},{MAX_OFFSET[1]*100:.0f},{MAX_OFFSET[2]*100:.0f})cm  IK_thr={IK_FAIL_THR*100:.0f}cm")
    print("=" * 55 + "\n")

    # ── Terminal raw mode ─────────────────────────────────────────────
    fd, old = _raw_mode()

    try:
        while True:
            ch = _getch(timeout=0.02)
            if ch is None:
                time.sleep(0.02)
                continue

            moved = False
            # ── Translation ───────────────────────────────────────────
            if ch == 'w':
                p_des[0] += STEP; moved = True
            elif ch == 's':
                p_des[0] -= STEP; moved = True
            elif ch == 'a':
                p_des[1] += STEP; moved = True
            elif ch == 'd':
                p_des[1] -= STEP; moved = True
            elif ch == 'q':
                p_des[2] += Z_STEP; moved = True
            elif ch == 'e':
                p_des[2] -= Z_STEP; moved = True
            # ── Reset ─────────────────────────────────────────────────
            elif ch == 'r':
                p_des = p0.copy()
                R_des = R0.copy()
                q_head, q_arm = kin.split_q(q_init)
                print("  ↺ reset to start")
            # ── Quit ──────────────────────────────────────────────────
            elif ch == '\x1b':  # ESC
                break

            if moved:
                # Clamp workspace
                offset = p_des - p0
                offset = np.clip(offset, -MAX_OFFSET, MAX_OFFSET)
                p_des = p0 + offset

                # Build target transform
                T_tgt = np.eye(4, dtype=float)
                T_tgt[:3, :3] = R_des
                T_tgt[:3, 3] = p_des

                # IK
                try:
                    q_hs, q_as, err, it = kin.ik_T_ee_with_arm_only(T_tgt, q_head, q_arm)
                    if np.isnan(err) or err > IK_FAIL_THR:
                        print(f"  ⚠ IK fail: err={err:.3f} (thr={IK_FAIL_THR})")
                    else:
                        robot.set_positions(np.concatenate([q_hs, q_as]))
                        q_head, q_arm = q_hs, q_as
                        print(f"  → ({p_des[0]:.3f}, {p_des[1]:.3f}, {p_des[2]:.3f})  err={err:.4f}")
                except Exception as e:
                    print(f"  ✗ IK: {e}")

    except KeyboardInterrupt:
        print("\n[Exit]")
    finally:
        _restore(fd, old)
        robot.close()


if __name__ == "__main__":
    main()
