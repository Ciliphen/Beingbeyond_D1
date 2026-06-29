#!/usr/bin/env python3
"""
Hand-eye calibration: pixel ↔ table coordinates (2D homography).

IMPORTANT: Fix the head at a known position before calibration.
           DO NOT move the head during or after calibration!
           Detection/grasping must use the SAME head position.

Flow:
  1. Click a reference point on the table in the camera view
  2. Use WASD/QE to move EE tip to that exact physical point
  3. Press SPACE to record a (pixel, world) pair
  4. Repeat 6+ times for different points across the table
  5. Press C to compute and save the homography matrix

Usage:
    conda activate bb_d1
    cd ~/Beingbeyond_D1
    python block_grasp/calibrate_handeye.py
"""
import math
import os
import select
import sys
import termios
import time
import tty

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from camera.d1_camera_primitive import D1CameraPrimitive
from beingbeyond_d1_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig
from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
from beingbeyond_d1_sdk.head_arm import HeadArmRobot

SAVE_PATH = os.path.join(os.path.dirname(__file__), "handeye_homography.npy")
STEP = 0.01
Z_STEP = 0.01
MAX_DXYZ = np.array([0.3, 0.3, 0.2])
IK_FAIL_THR = 0.05


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


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["click"] = (x, y)


def main():
    print("\033[91m⚠ 急停按钮请保持触手可及！\033[0m")
    print("\033[93m⚠ 校准期间头部不要移动！之后抓取时也保持头部在同一位置。\033[0m\n")

    urdf = get_default_urdf_path()
    kin = D1Kinematics(D1KinematicsConfig(urdf_path=urdf))

    # ── Init robot ────────────────────────────────────────────────────
    print("[Init] Robot ...")
    robot = HeadArmRobot(urdf_path=urdf, dev="/dev/ttyUSB0", baudrate=1_000_000)
    print("[Init] Camera ...")
    cam = D1CameraPrimitive(width=1280, height=720, fps=30)

    # ── Safe posture ──────────────────────────────────────────────────
    print("[Init] Safe posture ...")
    q_init = np.radians([0, 0, 0, -60, 60, 0, 0, 0])
    robot.set_positions(q_init)
    robot.wait_until_reached(q_init, active_joint_indices=range(8))
    time.sleep(0.3)

    q_cur = np.asarray(robot.get_positions(), dtype=float)
    q_head, q_arm = kin.split_q(q_cur)
    T0 = kin.ee_in_base(q_head, q_arm)
    p0 = T0[:3, 3].copy()
    R0 = T0[:3, :3].copy()
    p_des = p0.copy()
    print(f"       EE: ({p0[0]:.3f}, {p0[1]:.3f}, {p0[2]:.3f})")

    # ── Calibration state ─────────────────────────────────────────────
    pixel_pts = []   # (u, v) pixel
    world_pts = []   # (x, y) world
    click_state = {"click": None}
    last_click = None  # (u, v)

    WINDOW = "Calibration  |  Click → WASD move EE → SPACE record  |  C=compute  Q=quit"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.setMouseCallback(WINDOW, _on_mouse, click_state)

    print("\n" + "=" * 60)
    print("  1. Click point on image")
    print("  2. WASD/ZX → move EE tip to that point")
    print("  3. SPACE → record pair")
    print("  4. Repeat 6+ times across the table")
    print("  5. C → compute & save homography")
    print("=" * 60 + "\n")

    fd, old = _raw_mode()

    try:
        while True:
            # ── Camera frame ──────────────────────────────────────────
            rgb = cam.snapshot(filtered=False)
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # ── Mouse ─────────────────────────────────────────────────
            if click_state["click"] is not None:
                last_click = click_state["click"]
                click_state["click"] = None
                print(f"\n[Click] ({last_click[0]}, {last_click[1]}) — move EE here, then SPACE")

            if last_click is not None:
                cv2.drawMarker(vis, last_click, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

            # Draw recorded pairs
            for (u, v), (wx, wy) in zip(pixel_pts, world_pts):
                cv2.circle(vis, (int(u), int(v)), 6, (255, 100, 0), -1)

            # Current EE position
            q_cur = np.asarray(robot.get_positions(), dtype=float)
            q_head, q_arm = kin.split_q(q_cur)
            T_cur = kin.ee_in_base(q_head, q_arm)
            ex, ey, ez = T_cur[0, 3], T_cur[1, 3], T_cur[2, 3]

            cv2.putText(vis, f"EE: ({ex:.3f}, {ey:.3f}, {ez:.3f})  Pairs: {len(pixel_pts)}",
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow(WINDOW, vis)
            cv2.waitKey(5)

            # ── Keyboard ──────────────────────────────────────────────
            ch = _getch()
            if ch is None:
                continue

            moved = False

            if ch == '\x1b' or ch == 'q':
                break

            elif ch == ' ' and last_click is not None:
                pixel_pts.append(last_click)
                world_pts.append((ex, ey))
                n = len(pixel_pts)
                print(f"[#{n}] pixel=({last_click[0]},{last_click[1]}) → world=({ex:.3f},{ey:.3f})")
                last_click = None

            elif ch == ' ' and last_click is None:
                print("  ⚠ Click a point first!")

            elif ch in ('c', 'C'):
                if len(pixel_pts) < 4:
                    print(f"  Need ≥4 pairs, have {len(pixel_pts)}")
                else:
                    P = np.array(pixel_pts, dtype=float)
                    W = np.array(world_pts, dtype=float)
                    A = []
                    for (u, v), (wx, wy) in zip(P, W):
                        A.append([u, v, 1, 0, 0, 0, -wx*u, -wx*v, -wx])
                        A.append([0, 0, 0, u, v, 1, -wy*u, -wy*v, -wy])
                    A = np.array(A, dtype=float)
                    _, _, Vt = np.linalg.svd(A)
                    H = Vt[-1].reshape(3, 3)
                    H /= H[2, 2]

                    ones = np.ones((P.shape[0], 1))
                    Ph = np.hstack([P, ones])
                    Wp = (H @ Ph.T).T
                    Wp /= Wp[:, 2:3]
                    errs = np.linalg.norm(W - Wp[:, :2], axis=1) * 1000
                    print(f"\n  Homography (3×3):\n{H}")
                    print(f"  Errors: mean={errs.mean():.1f}mm  max={errs.max():.1f}mm")
                    if errs.mean() < 10:
                        np.save(SAVE_PATH, H)
                        print(f"  ✅ Saved → {SAVE_PATH}")
                    else:
                        print(f"  ⚠ Error too large. Add more pairs or redo.")

            # ── Teleop ────────────────────────────────────────────────
            elif ch == 'w':    p_des[0] += STEP; moved = True
            elif ch == 's':    p_des[0] -= STEP; moved = True
            elif ch == 'a':    p_des[1] += STEP; moved = True
            elif ch == 'd':    p_des[1] -= STEP; moved = True
            elif ch == 'z':    p_des[2] += Z_STEP; moved = True
            elif ch == 'x':    p_des[2] -= Z_STEP; moved = True
            elif ch == 'r':    p_des = p0.copy(); print("  ↺ reset")

            if moved:
                off = p_des - p0
                off = np.clip(off, -MAX_DXYZ, MAX_DXYZ)
                p_des = p0 + off
                T_tgt = np.eye(4)
                T_tgt[:3, :3] = R0
                T_tgt[:3, 3] = p_des
                try:
                    q_hs, q_as, err, it = kin.ik_T_ee_with_arm_only(T_tgt, q_head, q_arm)
                    if not np.isnan(err) and err <= IK_FAIL_THR:
                        robot.set_positions(np.concatenate([q_hs, q_as]))
                        q_head, q_arm = q_hs, q_as
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\n[Exit]")
    finally:
        _restore(fd, old)
        cam.close()
        robot.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
