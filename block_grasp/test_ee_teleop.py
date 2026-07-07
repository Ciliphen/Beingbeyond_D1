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
import math, os, sys
import select
import termios
import time
import tty

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from beingbeyond_d1_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig
from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
from beingbeyond_d1_sdk.head_arm import HeadArmRobot
from beingbeyond_d1_sdk.dex_hand import DexHand
from block_grasp.ik_scipy import scipy_ik
from block_grasp.config import GRAVITY_SAG_FACTOR, JOINT_JUMP_THR_DEG

STEP = 0.01   # 1cm
Z_STEP = 0.01
ORI_STEP = math.radians(5.0)  # 5°
MAX_OFFSET = np.array([0.30, 0.30, 0.15])
IK_FAIL_THR = 0.02  # m — Jacobian IK converges to ~0.0001, 2cm is generous
JOINT_JUMP_THR = math.radians(JOINT_JUMP_THR_DEG)  # near-singularity guard (from config)


def _rot_x(a): c,s = math.cos(a), math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)
def _rot_y(a): c,s = math.cos(a), math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)
def _rot_z(a): c,s = math.cos(a), math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)
def _ortho(M): U,_,Vt = np.linalg.svd(M); return U @ Vt

def _map_hand(t):
    """Map t∈[0,1] to 6D joint positions. 0=open, 1=power-grasp close."""
    A = [0.64, 0.8, 0.54, 0.58, 0.0, 0.0]   # closed
    B = [0.0,  0.8, 0.0,  0.0,  0.0, 0.0]   # open
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return [b + t * (a - b) for a, b in zip(A, B)]


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
    hand = DexHand(hand_type="right", can_iface="can0", baudrate=1_000_000)
    HAND_LEVELS = [0.0, 0.3, 0.5, 0.65, 0.8, 1.0]   # open → close levels
    HAND_NAMES  = ["open", "loose", "half", "firm", "tight", "max"]
    hand_level = 0   # index into HAND_LEVELS

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
    p_desired = p0.copy()   # user-commanded (uncompensated) EE position

    hand.set_joint_pos(_map_hand(0.0))  # open at start
    print(f"       EE: ({p0[0]:.3f}, {p0[1]:.3f}, {p0[2]:.3f})  hand=open")

    # ── Help ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  W/S X±  |  A/D Y±  |  Q/E Z±  |  R reset")
    print("  U/O roll±  I/K pitch±  J/L yaw±  (5°)")
    print("  SPACE/B hand close/open  |  H help  |  ESC quit")
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

            # ── Current EE orientation from FK (translation uses p_desired) ──
            T_cur = kin.ee_in_base(q_head, q_arm)
            R_cur = T_cur[:3, :3].copy()
            delta_pos = np.zeros(3)
            delta_R = np.eye(3)
            moved = False

            # ── Translation ───────────────────────────────────────────
            if ch == 'w':
                delta_pos[0] += STEP; moved = True
            elif ch == 's':
                delta_pos[0] -= STEP; moved = True
            elif ch == 'a':
                delta_pos[1] += STEP; moved = True
            elif ch == 'd':
                delta_pos[1] -= STEP; moved = True
            elif ch == 'q':
                delta_pos[2] += Z_STEP; moved = True
            elif ch == 'e':
                delta_pos[2] -= Z_STEP; moved = True
            # ── Hand (SPACE = next, B = prev) ─────────────────────────
            elif ch == ' ':
                hand_level = min(hand_level + 1, len(HAND_LEVELS) - 1)
                pos = HAND_LEVELS[hand_level]
                hand.set_joint_pos(_map_hand(pos))
                print(f"  🖐 {HAND_NAMES[hand_level]} ({pos:.2f})")
            elif ch == 'b':
                hand_level = max(hand_level - 1, 0)
                pos = HAND_LEVELS[hand_level]
                hand.set_joint_pos(_map_hand(pos))
                print(f"  🖐 {HAND_NAMES[hand_level]} ({pos:.2f})")
            # ── Orientation ──────────────────────────────────────────
            elif ch == 'u':    delta_R = _rot_x(+ORI_STEP); moved = True
            elif ch == 'o':    delta_R = _rot_x(-ORI_STEP); moved = True
            elif ch == 'i':    delta_R = _rot_y(-ORI_STEP); moved = True
            elif ch == 'k':    delta_R = _rot_y(+ORI_STEP); moved = True
            elif ch == 'j':    delta_R = _rot_z(+ORI_STEP); moved = True
            elif ch == 'l':    delta_R = _rot_z(-ORI_STEP); moved = True
            elif ch == 'h':
                print("\n  W/S X±  A/D Y±  Q/E Z±  U/O roll±  I/K pitch±  J/L yaw±")
                print("  SPACE hand+  B hand-  R reset\n")
            # ── Reset ─────────────────────────────────────────────────
            elif ch == 'r':
                q_head, q_arm = kin.split_q(q_init)
                robot.set_positions(q_init)
                robot.wait_until_reached(q_init, active_joint_indices=range(8))
                p_desired = p0.copy(); R_cur = R0.copy()
                print("  ↺ reset to start")
            # ── Quit ──────────────────────────────────────────────────
            elif ch == '\x1b':  # ESC
                break

            if moved:
                # Delta acts on the *desired* (uncompensated) position so the
                # gravity-sag term below is re-derived from scratch each step
                # and never accumulates.
                p_prev = p_desired.copy()   # rollback target if IK is rejected
                p_desired = p_desired + delta_pos
                R_new = _ortho(delta_R @ R_cur)

                # Clamp workspace relative to origin p0
                offset = np.clip(p_desired - p0, -MAX_OFFSET, MAX_OFFSET)
                p_desired = p0 + offset

                # Gravity-sag compensation: the arm droops under its own weight
                # the further it reaches, so raise the IK target Z to match.
                # Same cubic model as grasp_controller (dz ∝ r³).
                dist = math.hypot(p_desired[0], p_desired[1])
                z_sag = GRAVITY_SAG_FACTOR * dist ** 3
                p_tgt = p_desired.copy()
                p_tgt[2] += z_sag

                # Build target transform
                T_tgt = np.eye(4, dtype=float)
                T_tgt[:3, :3] = R_new
                T_tgt[:3, 3] = p_tgt

                # IK (SLSQP, roboarm-style: tilt/yaw decomposition)
                try:
                    q_hs, q_as, err, it = scipy_ik(
                        kin, T_tgt, q_head, q_arm,
                        z_weight=2.0, pos_tol=0.005,
                        tilt_tol_deg=5, yaw_tol_deg=10, max_iters=200)
                    dq_max = float(np.max(np.abs(q_as - q_arm)))
                    if np.isnan(err) or err > IK_FAIL_THR:
                        print(f"  ⚠ IK fail: err={err:.3f}  → not moved")
                        p_desired = p_prev
                    elif dq_max > JOINT_JUMP_THR:
                        print(f"  ⚠ joint jump {math.degrees(dq_max):.0f}° "
                              f"(near singularity) → not moved")
                        p_desired = p_prev
                    else:
                        robot.set_positions(np.concatenate([q_hs, q_as]))
                        q_head, q_arm = q_hs, q_as
                        # Read back actual achieved pose
                        T_ach = kin.ee_in_base(q_head, q_arm)
                        p_ach = T_ach[:3, 3]
                        rpy = R.from_matrix(T_ach[:3,:3]).as_euler('xyz', degrees=True)
                        # All joints in degrees
                        q_full = np.concatenate([q_hs, q_as])
                        j_deg = [f"{math.degrees(v):.1f}" for v in q_full]
                        print(f"  → ({p_ach[0]:.3f}, {p_ach[1]:.3f}, {p_ach[2]:.3f})  "
                              f"rpy=({rpy[0]:.0f},{rpy[1]:.0f},{rpy[2]:.0f})  "
                              f"sag=+{z_sag*1000:.0f}mm  err={err:.4f}\n"
                              f"     joints(°): head_yaw={j_deg[0]} head_pitch={j_deg[1]} | "
                              f"j1={j_deg[2]} j2={j_deg[3]} j3={j_deg[4]} j4={j_deg[5]} j5={j_deg[6]} j6={j_deg[7]}")
                except Exception as e:
                    print(f"  ✗ IK: {e}")
                    p_desired = p_prev

    except KeyboardInterrupt:
        print("\n[Exit]")
    finally:
        _restore(fd, old)
        hand.open_hand()
        hand.close_can()
        robot.close()


if __name__ == "__main__":
    main()
