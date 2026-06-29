#!/usr/bin/env python3
"""
Hand-eye calibration: pixel ↔ table coordinates (2D homography).

IMPORTANT: Keep the head still during calibration!
           The head angles are saved and must be restored for detection.

Flow:
  1. Adjust head with WASD to look at the table, then H to lock head
  2. Click a reference point on the table in the camera view (green cross)
  3. Move EE tip to that exact physical point (WASD/ZX + orientation keys)
  4. SPACE → record a (pixel, world) pair
  5. Repeat 6+ times across the table
  6. C → compute homography, save to calibration file

Saved file (handeye_calib.npz) contains:
  - H: 3×3 homography matrix
  - head_yaw, head_pitch: head angles at calibration time
  - pixel_pts, world_pts: recorded pairs (for debug)
  - mean_err_mm: calibration error

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
from beingbeyond_d1_sdk.dex_hand import DexHand

SAVE_PATH = os.path.join(os.path.dirname(__file__), "handeye_calib.npz")

STEP = 0.01
Z_STEP = 0.01
ORI_STEP = math.radians(5.0)
MAX_OFFSET = np.array([0.30, 0.30, 0.15])
IK_FAIL_THR = 0.10


def _rot_x(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
def _rot_y(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
def _rot_z(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
def _ortho(M): U, _, Vt = np.linalg.svd(M); return U @ Vt


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


# ── Load existing calibration ─────────────────────────────────────────────

def load_calib():
    """Return (H, head_yaw, head_pitch) or (None, None, None)."""
    if os.path.exists(SAVE_PATH):
        d = np.load(SAVE_PATH, allow_pickle=True)
        H = d["H"]
        head_yaw = float(d["head_yaw"])
        head_pitch = float(d["head_pitch"])
        mean_err = float(d["mean_err_mm"])
        n_pairs = len(d["pixel_pts"])
        print(f"[Load] Existing calibration: {n_pairs} pairs, error={mean_err:.1f}mm")
        print(f"       Head: yaw={head_yaw:.1f}°  pitch={head_pitch:.1f}°")
        return H, head_yaw, head_pitch
    return None, None, None


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\033[91m⚠ 急停按钮请保持触手可及！\033[0m")
    print("\033[93m⚠ 校准期间头部不要移动！\033[0m\n")

    # ── Check for existing calibration ─────────────────────────────────
    existing = os.path.exists(SAVE_PATH)
    if existing:
        H_old, hy_old, hp_old = load_calib()
        print("  Already calibrated. Overwrite? (y/N)")
        if input("  > ").strip().lower() != 'y':
            print("  Exiting.")
            return

    urdf = get_default_urdf_path()
    kin = D1Kinematics(D1KinematicsConfig(urdf_path=urdf))

    # ── Init hardware ──────────────────────────────────────────────────
    print("[Init] Robot ...")
    robot = HeadArmRobot(urdf_path=urdf, dev="/dev/ttyUSB0", baudrate=1_000_000)
    hand = DexHand(hand_type="right", can_iface="can0", baudrate=1_000_000)
    print("[Init] Camera ...")
    cam = D1CameraPrimitive(width=1280, height=720, fps=30)

    # ── Safe posture ───────────────────────────────────────────────────
    print("[Init] Safe posture ...")
    q_init = np.radians([0, 0, 0, -60, 60, 0, 0, 0])
    robot.set_positions(q_init)
    robot.wait_until_reached(q_init, active_joint_indices=range(8))
    time.sleep(0.3)
    hand_closed = True
    hand.set_joint_pos([0.64, 0.8, 0.54, 0.58, 0.0, 0.0])  # closed for precision
    print(f"       Hand: closed (fingertip for precise pointing)")

    # ── Fixed head position ──────────────────────────────────────────
    HEAD_YAW = math.radians(-10.0)
    HEAD_PITCH = math.radians(35.0)
    last_q = np.asarray(robot.get_positions(), dtype=float)
    last_q[0] = HEAD_YAW
    last_q[1] = HEAD_PITCH
    robot.set_positions(last_q)
    robot.wait_until_reached(last_q, active_joint_indices=[0, 1])
    time.sleep(0.3)
    q_cur = np.asarray(robot.get_positions(), dtype=float)
    q_head, q_arm = kin.split_q(q_cur)
    T0 = kin.ee_in_base(q_head, q_arm)
    p_des = T0[:3, 3].copy()
    R_des = T0[:3, :3].copy()
    p0 = p_des.copy()
    R0 = R_des.copy()
    print(f"       EE: ({p0[0]:.3f}, {p0[1]:.3f}, {p0[2]:.3f})")
    print(f"       Head: yaw={math.degrees(HEAD_YAW):.0f}°  pitch={math.degrees(HEAD_PITCH):.0f}°")

    # ── Calibration state ──────────────────────────────────────────────
    pixel_pts = []
    world_pts = []
    click_state = {"click": None}
    last_click = None

    WINDOW = "Hand-Eye Calibration"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.setMouseCallback(WINDOW, _on_mouse, click_state)

    print("\n" + "=" * 60)
    print("  1. Click a reference point on the table")
    print("  2. Move EE tip there: WASD=XY  ZX=Z  UO/IK/JL=RPY")
    print("  3. SPACE to record a pair")
    print("  4. Repeat 6+ times, then C to compute & save")
    print("  B=toggle hand  R=reset EE  Q=quit")
    print("=" * 60 + "\n")

    fd, old = _raw_mode()

    try:
        while True:
            # ── Camera ────────────────────────────────────────────────
            rgb = cam.snapshot(filtered=False)
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # ── Mouse ─────────────────────────────────────────────────
            if click_state["click"] is not None:
                last_click = click_state["click"]
                click_state["click"] = None
                print(f"\n[Click] ({last_click[0]}, {last_click[1]}) → move EE here, then SPACE")

            # Draw markers
            if last_click is not None:
                cv2.drawMarker(vis, last_click, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            for i, ((u, v), (wx, wy)) in enumerate(zip(pixel_pts, world_pts)):
                cv2.circle(vis, (int(u), int(v)), 6, (255, 100, 0), -1)
                cv2.putText(vis, f"#{i+1}", (int(u)+10, int(v)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

            # ── EE position (display only, don't overwrite IK state) ───
            q_disp = np.asarray(robot.get_positions(), dtype=float)
            qh_disp, qa_disp = kin.split_q(q_disp)
            T_disp = kin.ee_in_base(qh_disp, qa_disp)
            ex, ey, ez = T_disp[0, 3], T_disp[1, 3], T_disp[2, 3]

            # ── Overlay ────────────────────────────────────────────────
            status = "READY — click point, move EE, SPACE to record"
            cv2.putText(vis, status, (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, f"EE: ({ex:.3f}, {ey:.3f}, {ez:.3f})  Pairs: {len(pixel_pts)}",
                        (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, f"Head: yaw={math.degrees(qh_disp[0]):.0f} pitch={math.degrees(qh_disp[1]):.0f}",
                        (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            cv2.imshow(WINDOW, vis)
            cv2.waitKey(5)

            # ── Keyboard ──────────────────────────────────────────────
            ch = _getch()
            if ch is None:
                continue

            moved = False

            if ch == 'q' or ch == '\x1b':
                break

            # ── Lock head ─────────────────────────────────────────────
            elif ch == 'b':
                hand_closed = not hand_closed
                if hand_closed:
                    hand.set_joint_pos([0.64, 0.8, 0.54, 0.58, 0.0, 0.0])
                else:
                    hand.set_joint_pos([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
                print(f"  🖐 hand {'closed' if hand_closed else 'open'}")
            # ── Record pair ───────────────────────────────────────────
            elif ch == ' ':
                if last_click is None:
                    print("  ⚠ Click a point first!")
                else:
                    pixel_pts.append(last_click)
                    world_pts.append((ex, ey))
                    n = len(pixel_pts)
                    print(f"  ✅ Pair #{n}: pixel=({last_click[0]},{last_click[1]}) → world=({ex:.3f},{ey:.3f})")
                    last_click = None

            # ── Compute ───────────────────────────────────────────────
            elif ch in ('c', 'C'):
                if len(pixel_pts) < 4:
                    print(f"  ⚠ Need >=4 pairs, have {len(pixel_pts)}")
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

                    print(f"\n{'='*50}")
                    print(f"  Homography H (3x3):")
                    for row in H:
                        print(f"    {row}")
                    print(f"  Errors per pair (mm): {[f'{e:.1f}' for e in errs]}")
                    print(f"  Mean: {errs.mean():.1f}mm  Max: {errs.max():.1f}mm")

                    if errs.mean() < 10:
                        np.savez(
                            SAVE_PATH,
                            H=H,
                            head_yaw=HEAD_YAW,
                            head_pitch=HEAD_PITCH,
                            pixel_pts=np.array(pixel_pts),
                            world_pts=np.array(world_pts),
                            mean_err_mm=errs.mean(),
                        )
                        print(f"  ✅ Saved -> {SAVE_PATH}")
                    else:
                        print(f"  ⚠ Error too large ({errs.mean():.1f}mm). Add more pairs or redo.")
                    print(f"{'='*50}\n")

            # ── EE teleop ─────────────────────────────────────────────
            elif ch == 'w':    p_des[0] += STEP; moved = True
            elif ch == 's':    p_des[0] -= STEP; moved = True
            elif ch == 'a':    p_des[1] += STEP; moved = True
            elif ch == 'd':    p_des[1] -= STEP; moved = True
            elif ch == 'z':    p_des[2] += Z_STEP; moved = True
            elif ch == 'x':    p_des[2] -= Z_STEP; moved = True
            elif ch == 'u':    R_des = _ortho(_rot_x(+ORI_STEP) @ R_des); moved = True
            elif ch == 'o':    R_des = _ortho(_rot_x(-ORI_STEP) @ R_des); moved = True
            elif ch == 'i':    R_des = _ortho(_rot_y(-ORI_STEP) @ R_des); moved = True
            elif ch == 'k':    R_des = _ortho(_rot_y(+ORI_STEP) @ R_des); moved = True
            elif ch == 'j':    R_des = _ortho(_rot_z(+ORI_STEP) @ R_des); moved = True
            elif ch == 'l':    R_des = _ortho(_rot_z(-ORI_STEP) @ R_des); moved = True
            elif ch == 'r':
                p_des = p0.copy(); R_des = R0.copy()
                q_head, q_arm = kin.split_q(q_init)
                moved = True
                print("  ↺ EE reset")

            if moved:
                off = p_des - p0
                off = np.clip(off, -MAX_OFFSET, MAX_OFFSET)
                p_des = p0 + off
                T_tgt = np.eye(4)
                T_tgt[:3, :3] = R_des
                T_tgt[:3, 3] = p_des
                try:
                    q_hs, q_as, err, it = kin.ik_T_ee_with_arm_only(T_tgt, q_head, q_arm)
                    if not np.isnan(err) and err <= IK_FAIL_THR:
                        cmd = np.concatenate([q_hs, q_as])
                        cmd[0] = HEAD_YAW    # force head fixed
                        cmd[1] = HEAD_PITCH
                        robot.set_positions(cmd)
                        last_q = cmd
                        q_head, q_arm = kin.split_q(cmd)
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\n[Exit]")
    finally:
        _restore(fd, old)
        hand.open_hand()
        hand.close_can()
        cam.close()
        robot.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
