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
from vision import RealSenseCamera
from beingbeyond_d1_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig
from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
from beingbeyond_d1_sdk.dex_hand import DexHand
# Low-level Feetech bus primitives — the public HeadArmRobot API does not expose
# per-joint torque control, so drag-teach talks to ServoBus directly.
from beingbeyond_d1_sdk.core._head_arm_core import (
    ServoBus, JOINT_TO_ID, JOINT_ORDER, ARM_JOINTS,
    step_to_q, q_to_step, vel_rad_to_param, acc_rad_to_param,
    load_head_arm_limits,
)

SAVE_PATH = os.path.join(os.path.dirname(__file__), "handeye_calib.npz")

STEP = 0.01
Z_STEP = 0.01
ORI_STEP = math.radians(5.0)
MAX_OFFSET = np.array([0.50, 0.50, 0.15])
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


# ── Servo bus wrapper (HeadArmRobot-compatible + arm torque control) ───────

class _ArmBus:
    """Thin ServoBus wrapper exposing the HeadArmRobot methods this script
    uses, plus per-arm torque control for drag-teach.

    Head joints (ids 11/12) stay torque-enabled and hold their commanded
    angle at all times — only the 6 arm joints (ids 21~26) get released.
    Uses the same step<->rad conversions as the SDK core so joint angles
    match HeadArmRobot.get_positions().
    """

    IDS = [JOINT_TO_ID[j] for j in JOINT_ORDER]     # 8 servo ids, SDK order
    ARM_IDS = [JOINT_TO_ID[j] for j in ARM_JOINTS]  # 6 arm servo ids (21~26)

    def __init__(self, urdf, dev, baudrate,
                 vel=math.radians(60.0), acc=math.radians(60.0)):
        self.bus = ServoBus(dev, baudrate)
        self.bus.ensure_protection(self.IDS)
        self._lim = load_head_arm_limits(urdf)
        self._speeds = [vel_rad_to_param(vel)] * 8
        self._accs = [acc_rad_to_param(acc)] * 8
        self.arm_torque_on = True
        for i in self.IDS:
            self.bus.torque_enable(i, True)

    def get_positions(self):
        d = self.bus.sync_read_pos_speed(self.IDS)
        return [step_to_q(d[i][0]) for i in self.IDS]

    def set_positions(self, q_rad):
        steps = [q_to_step(q, self._lim[j]) for q, j in zip(q_rad, JOINT_ORDER)]
        self.bus.sync_write_pos_ex(self.IDS, steps, self._speeds, self._accs)

    def set_arm_torque(self, on):
        """Enable/disable torque on the 6 arm joints only (head untouched)."""
        if on:
            # Snap the goal register to the current dragged pose first, so the
            # arm holds where it is instead of jerking back to the stale goal.
            self.set_positions(self.get_positions())
        for i in self.ARM_IDS:
            self.bus.torque_enable(i, on)
        self.arm_torque_on = on

    def wait_until_reached(self, target, active_joint_indices=None,
                           pos_tol_deg=5.0, timeout=15.0):
        idx = list(active_joint_indices) if active_joint_indices is not None else range(8)
        tol = math.radians(pos_tol_deg)
        t0 = time.time()
        while time.time() - t0 < timeout:
            q = self.get_positions()
            if all(abs(q[i] - target[i]) <= tol for i in idx):
                return time.time() - t0
            time.sleep(0.02)
        return None

    def close(self):
        # Hold the current pose on exit (avoid a sudden sag if we quit while
        # torque was released), then release the port.
        try:
            if not self.arm_torque_on:
                self.set_arm_torque(True)
        except Exception:
            pass
        self.bus.close()


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
    robot = _ArmBus(urdf, dev="/dev/ttyUSB0", baudrate=1_000_000)
    hand = DexHand(hand_type="right", can_iface="can0", baudrate=1_000_000)
    print("[Init] Camera ...")
    cam = RealSenseCamera(width=1280, height=720, hz=30)

    # ── Default setup (same as click_goto) ────────────────────────────
    print("[Init] Safe posture ...")
    q_init = np.radians([0, 0, 0, -60, 60, 0, 0, 0])
    robot.set_positions(q_init)
    robot.wait_until_reached(q_init, active_joint_indices=range(8))
    time.sleep(0.3)
    hand.set_joint_pos([0.64, 0.8, 0.54, 0.58, 0.0, 0.0])

    # Fixed head
    HEAD_YAW = math.radians(-10.0)
    HEAD_PITCH = math.radians(35.0)
    last_q = np.asarray(robot.get_positions(), dtype=float)
    last_q[0] = HEAD_YAW
    last_q[1] = HEAD_PITCH
    robot.set_positions(last_q)
    robot.wait_until_reached(last_q, active_joint_indices=[0, 1])
    time.sleep(0.3)
    q_head, q_arm = kin.split_q(np.asarray(robot.get_positions(), dtype=float))

    # Fixed RPY (same as click_goto default)
    from scipy.spatial.transform import Rotation as _R
    R_des = _R.from_euler('xyz', [178, 61, -175], degrees=True).as_matrix()

    # Lift to safe height + rotate to target RPY
    T_cur = kin.ee_in_base(q_head, q_arm)
    p_tgt = T_cur[:3,3].copy(); p_tgt[2] = 0.30
    T_lift = np.eye(4); T_lift[:3,:3] = T_cur[:3,:3]; T_lift[:3,3] = p_tgt
    q_hs, q_as, err, _ = kin.ik_T_ee_with_arm_only(T_lift, q_head, q_arm)
    if err < 0.05:
        cmd = np.concatenate([q_hs, q_as]); cmd[0]=HEAD_YAW; cmd[1]=HEAD_PITCH
        robot.set_positions(cmd); robot.wait_until_reached(cmd, active_joint_indices=range(2,8))
        q_head, q_arm = kin.split_q(cmd)

    q0 = _R.from_matrix(kin.ee_in_base(q_head, q_arm)[:3,:3]).as_quat()
    q1 = _R.from_matrix(R_des).as_quat()
    if np.dot(q0, q1) < 0: q1 = -q1  # shortest path
    T_cur = kin.ee_in_base(q_head, q_arm); p_cur = T_cur[:3,3]
    omega = np.arccos(np.clip(np.dot(q0, q1), -1, 1))
    n = max(1, math.ceil(abs(omega)*2 / 0.05))
    for i in range(n):
        a = (i+1)/n; qi = q0 if abs(omega)<1e-10 else (np.sin((1-a)*omega)*q0 + np.sin(a*omega)*q1)/np.sin(omega)
        T_rt = np.eye(4); T_rt[:3,:3] = _R.from_quat(qi).as_matrix(); T_rt[:3,3] = p_cur
        q_hs, q_as, err, _ = kin.ik_T_ee_with_arm_only(T_rt, q_head, q_arm)
        if err < 0.05:
            cmd = np.concatenate([q_hs, q_as]); cmd[0]=HEAD_YAW; cmd[1]=HEAD_PITCH
            robot.set_positions(cmd); time.sleep(0.02)
            q_head, q_arm = kin.split_q(cmd)

    T0 = kin.ee_in_base(q_head, q_arm)
    p_des = T0[:3, 3].copy(); p0 = p_des.copy()
    R0 = T0[:3,:3].copy()
    rpy = _R.from_matrix(R0).as_euler('xyz', degrees=True)
    print(f"       EE: ({p0[0]:.3f},{p0[1]:.3f},{p0[2]:.3f})  RPY=({rpy[0]:.0f},{rpy[1]:.0f},{rpy[2]:.0f})")
    print(f"       Head: yaw={math.degrees(HEAD_YAW):.0f}°  pitch={math.degrees(HEAD_PITCH):.0f}°")

    # ── Calibration state ──────────────────────────────────────────────
    pixel_pts = []
    world_pts = []
    click_state = {"click": None}
    last_click = None
    last_print_t = 0.0   # throttle for the real-time drag-mode readout

    WINDOW = "Hand-Eye Calibration"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.setMouseCallback(WINDOW, _on_mouse, click_state)

    print("\n" + "=" * 60)
    print("  1. Click a reference point on the table")
    print("  2. Move EE tip there — two ways:")
    print("       • Keyboard/IK: WASD=XY  ZX=Z  UO/IK/JL=RPY")
    print("       • Drag-teach:  T to release arm torque, then hand-drag it")
    print("  3. SPACE to record a pair")
    print("  4. Repeat 6+ times, then C to compute & save")
    print("  T=toggle arm torque (drag mode)  B=toggle hand  R=reset EE  ESC=quit")
    print("\033[93m  ⚠ In drag mode the arm goes limp — support it before pressing T!\033[0m")
    print("=" * 60 + "\n")

    try:
        while True:
            # ── Camera ────────────────────────────────────────────────
            rgb, _ = cam.get_aligned_frames(filtered=False)
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # ── Mouse ─────────────────────────────────────────────────
            if click_state["click"] is not None:
                last_click = click_state["click"]
                click_state["click"] = None
                print(f"\n[Click] ({last_click[0]}, {last_click[1]}) → move EE here, then SPACE")

            # Draw markers
            if last_click is not None:
                cv2.drawMarker(vis, last_click, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            for i, ((u, v), (wx, wy, _wz)) in enumerate(zip(pixel_pts, world_pts)):
                cv2.circle(vis, (int(u), int(v)), 6, (255, 100, 0), -1)
                cv2.putText(vis, f"#{i+1}", (int(u)+10, int(v)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

            # ── EE position (display only, don't overwrite IK state) ───
            q_disp = np.asarray(robot.get_positions(), dtype=float)
            qh_disp, qa_disp = kin.split_q(q_disp)
            T_disp = kin.ee_in_base(qh_disp, qa_disp)
            ex, ey, ez = T_disp[0, 3], T_disp[1, 3], T_disp[2, 3]

            # ── Real-time readout while torque is released (drag mode) ─
            if not robot.arm_torque_on and time.time() - last_print_t > 0.1:
                rpy_d = _R.from_matrix(T_disp[:3, :3]).as_euler('xyz', degrees=True)
                print(f"\r[DRAG] EE=({ex:+.3f}, {ey:+.3f}, {ez:+.3f})  "
                      f"RPY=({rpy_d[0]:+.0f}, {rpy_d[1]:+.0f}, {rpy_d[2]:+.0f})   ",
                      end="", flush=True)
                last_print_t = time.time()

            # ── Overlay ────────────────────────────────────────────────
            if robot.arm_torque_on:
                status = "KEYBOARD/IK — click point, move EE, SPACE to record"
                status_color = (0, 255, 0)
            else:
                status = "DRAG MODE (torque OFF) — hand-drag EE, SPACE to record"
                status_color = (0, 165, 255)
            cv2.putText(vis, status, (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(vis, f"EE: ({ex:.3f}, {ey:.3f}, {ez:.3f})  Pairs: {len(pixel_pts)}",
                        (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, f"Head: yaw={math.degrees(qh_disp[0]):.0f} pitch={math.degrees(qh_disp[1]):.0f}",
                        (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            cv2.imshow(WINDOW, vis)
            # ── Keyboard (captured via OpenCV window — no focus switch needed) ──
            raw = cv2.waitKey(5)
            if raw == -1:
                continue
            key = raw & 0xFF
            ch = chr(key) if 32 <= key < 127 else None

            moved = False

            if key == 27:  # ESC
                break

            # ── Toggle arm torque (drag-teach) ────────────────────────
            elif ch == 't':
                robot.set_arm_torque(not robot.arm_torque_on)
                if robot.arm_torque_on:
                    # Re-sync IK state to the current (dragged) pose so the next
                    # keyboard nudge starts from here instead of the stale target.
                    q_now = np.asarray(robot.get_positions(), dtype=float)
                    q_head, q_arm = kin.split_q(q_now)
                    T_now = kin.ee_in_base(q_head, q_arm)
                    p_des = T_now[:3, 3].copy(); p0 = p_des.copy()  # re-center clamp
                    R_des = T_now[:3, :3].copy()
                    print("  🔒 arm torque ON — keyboard/IK mode (state re-synced)")
                else:
                    print("  🖐 arm torque OFF — DRAG the arm by hand (support it!)")

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
                    world_pts.append((ex, ey, ez))  # save z too
                    n = len(pixel_pts)
                    print(f"  ✅ Pair #{n}: pixel=({last_click[0]},{last_click[1]}) → world=({ex:.3f},{ey:.3f},{ez:.3f})")
                    last_click = None

            # ── Compute ───────────────────────────────────────────────
            elif ch in ('c', 'C'):
                if len(pixel_pts) < 4:
                    print(f"  ⚠ Need >=4 pairs, have {len(pixel_pts)}")
                else:
                    P = np.array(pixel_pts, dtype=float)
                    W = np.array(world_pts, dtype=float)  # (N, 3) with z
                    Wxy = W[:, :2]

                    # Homography: pixel → (x, y)
                    A = []
                    for (u, v), (wx, wy) in zip(P, Wxy):
                        A.append([u, v, 1, 0, 0, 0, -wx*u, -wx*v, -wx])
                        A.append([0, 0, 0, u, v, 1, -wy*u, -wy*v, -wy])
                    A = np.array(A, dtype=float)
                    _, _, Vt = np.linalg.svd(A)
                    H = Vt[-1].reshape(3, 3); H /= H[2, 2]

                    ones = np.ones((P.shape[0], 1))
                    Ph = np.hstack([P, ones])
                    Wp = (H @ Ph.T).T; Wp /= Wp[:, 2:3]
                    errs_xy = np.linalg.norm(Wxy - Wp[:, :2], axis=1) * 1000

                    # Fit table plane: z = a*x + b*y + c
                    Xz = np.column_stack([Wxy, np.ones(len(Wxy))])
                    plane, _, _, _ = np.linalg.lstsq(Xz, W[:, 2], rcond=None)
                    a, b, c = plane
                    z_pred = a * Wxy[:, 0] + b * Wxy[:, 1] + c
                    errs_z = np.abs(W[:, 2] - z_pred) * 1000

                    print(f"\n{'='*50}")
                    print(f"  Homography H (3x3):")
                    for row in H:
                        print(f"    {row}")
                    print(f"  Table plane: z = {a:.4f}*x + {b:.4f}*y + {c:.4f}")
                    print(f"  XY errors (mm): {[f'{e:.1f}' for e in errs_xy]}")
                    print(f"  Z  errors (mm): {[f'{e:.1f}' for e in errs_z]}")
                    print(f"  Mean XY: {errs_xy.mean():.1f}mm  Mean Z: {errs_z.mean():.1f}mm")

                    if errs_xy.mean() < 20:
                        np.savez(
                            SAVE_PATH,
                            H=H, plane=plane,
                            head_yaw=HEAD_YAW, head_pitch=HEAD_PITCH,
                            pixel_pts=np.array(pixel_pts),
                            world_pts=np.array(world_pts),
                            mean_err_mm=errs_xy.mean(),
                        )
                        print(f"  ✅ Saved -> {SAVE_PATH}")
                    else:
                        print(f"  ⚠ Error >20mm ({errs_xy.mean():.1f}mm). Add more pairs or redo.")
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

            if moved and robot.arm_torque_on:
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
        hand.open_hand()
        hand.close_can()
        cam.stop()
        robot.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
