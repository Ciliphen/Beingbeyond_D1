#!/usr/bin/env python3
"""
Click on image → arm moves fingertip to that position.

Uses hand-eye calibration (handeye_calib.npz) for pixel→world transform.
Head is set to calibration position automatically.

Usage:
    conda activate bb_d1
    cd ~/Beingbeyond_D1
    python block_grasp/test_click_goto.py
"""
import math, os, sys, time

import cv2, numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from clients.camera import D1CameraPrimitive
from beingbeyond_d1_sdk.head_arm import HeadArmRobot
from beingbeyond_d1_sdk.dex_hand import DexHand
from beingbeyond_d1_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig
from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
from block_grasp.ik_scipy import scipy_ik, scipy_ik_multi_restart

CALIB = os.path.join(os.path.dirname(__file__), "handeye_calib.npz")

def _map_hand(t):
    """Map t∈[0,1] to 6D joint positions."""
    A = [0.64, 0.8, 0.54, 0.58, 0.0, 0.0]
    B = [0.0,  0.8, 0.0,  0.0,  0.0, 0.0]
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return [b + t * (a - b) for a, b in zip(A, B)]
IK_FAIL_THR = 0.10
Z_SAFE = 0.25     # approach height
Z_TOUCH = 0.18    # height above table (arm can't reach below ~0.15)


def main():
    # ── Load calibration ──────────────────────────────────────────────
    data = np.load(CALIB, allow_pickle=True)
    H = data["H"]
    head_yaw = float(data["head_yaw"])
    head_pitch = float(data["head_pitch"])
    # Use calibration points for Z interpolation
    if "world_pts" in data:
        W = data["world_pts"]          # (N, 3) — x,y,z when touching table
        z_ref = float(np.median(W[:, 2]))
    else:
        W = None; z_ref = Z_SAFE
    print(f"Calib: head yaw={math.degrees(head_yaw):.0f}° pitch={math.degrees(head_pitch):.0f}°")
    print(f"       z_ref={z_ref:.3f} ({len(W)} calib pts)" if W is not None else f"       z_ref={z_ref:.3f}")

    urdf = get_default_urdf_path()
    kin = D1Kinematics(D1KinematicsConfig(urdf_path=urdf))

    # ── Init hardware ─────────────────────────────────────────────────
    print("[Init] Robot ...")
    robot = HeadArmRobot(urdf_path=urdf, dev="/dev/ttyUSB0", baudrate=1_000_000)
    hand = DexHand(hand_type="right", can_iface="can0", baudrate=1_000_000)
    print("[Init] Camera ...")
    cam = D1CameraPrimitive(width=1280, height=720, fps=30)

    # ── Safe posture first, then set head ─────────────────────────────
    print("[Init] Safe posture ...")
    q_init = np.radians([0, 0, 0, -60, 60, 0, 0, 0])
    robot.set_positions(q_init)
    robot.wait_until_reached(q_init, active_joint_indices=range(8))
    time.sleep(0.3)
    print("[Init] Setting head ...")
    q = np.asarray(robot.get_positions(), dtype=float)
    q[0] = head_yaw
    q[1] = head_pitch
    robot.set_positions(q)
    robot.wait_until_reached(q, active_joint_indices=[0, 1])
    time.sleep(0.3)
    HAND_LEVELS = [0.0, 0.3, 0.5, 0.65, 0.8, 1.0]
    HAND_NAMES  = ["open", "loose", "half", "firm", "tight", "max"]
    hand_level = 3  # "firm"
    hand.set_joint_pos(_map_hand(HAND_LEVELS[hand_level]))
    print("       Ready.")

    # ── IK state ──────────────────────────────────────────────────────
    q_cur = np.asarray(robot.get_positions(), dtype=float)
    q_head, q_arm = kin.split_q(q_cur)
    q_arm0 = q_arm.copy()  # save for multi-restart IK
    T0 = kin.ee_in_base(q_head, q_arm)
    p_des = T0[:3, 3].copy()
    R_des = R.from_euler('xyz', [178, 61, -175], degrees=True).as_matrix()

    # ── Startup: lift to safe Z, then rotate to target RPY ────────────
    print("[Init] Lift to safe height ...")
    T_cur = kin.ee_in_base(q_head, q_arm)
    p_tgt = T_cur[:3,3].copy(); p_tgt[2] = Z_SAFE + 0.05
    T_lift = np.eye(4); T_lift[:3,:3] = T_cur[:3,:3]; T_lift[:3,3] = p_tgt
    q_hs, q_as, err, _ = kin.ik_T_ee_with_arm_only(T_lift, q_head, q_arm)
    if err < 0.05:
        cmd = np.concatenate([q_hs, q_as]); cmd[0]=head_yaw; cmd[1]=head_pitch
        robot.set_positions(cmd); robot.wait_until_reached(cmd, active_joint_indices=range(2,8))
        q_head, q_arm = kin.split_q(cmd)

    print("[Init] Rotate to target RPY ...")
    T_cur = kin.ee_in_base(q_head, q_arm); p_cur = T_cur[:3,3]
    R_cur = T_cur[:3,:3]
    q0 = R.from_matrix(R_cur).as_quat()
    q1 = R.from_matrix(R_des).as_quat()
    # Ensure shortest path (quaternion double-cover)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    omega = np.arccos(np.clip(np.dot(q0, q1), -1, 1))
    angle = omega * 2
    n_rot = max(1, math.ceil(angle / 0.05))
    for i in range(n_rot):
        a = (i+1) / n_rot
        # SLERP
        if abs(omega) < 1e-10:
            qi = q0
        else:
            qi = (np.sin((1-a)*omega)*q0 + np.sin(a*omega)*q1) / np.sin(omega)
        Ri = R.from_quat(qi).as_matrix()
        T_rt = np.eye(4); T_rt[:3,:3] = Ri; T_rt[:3,3] = p_cur
        q_hs, q_as, err, _ = kin.ik_T_ee_with_arm_only(T_rt, q_head, q_arm)
        if err < 0.05:
            cmd = np.concatenate([q_hs, q_as]); cmd[0]=head_yaw; cmd[1]=head_pitch
            robot.set_positions(cmd); time.sleep(0.02)
            q_head, q_arm = kin.split_q(cmd)
        else:
            print(f"  ⚠ rot IK err={err:.3f} at step {i+1}/{n_rot}")
    rpy = R.from_matrix(R_des).as_euler('xyz', degrees=True)
    print(f"       RPY=({rpy[0]:.0f},{rpy[1]:.0f},{rpy[2]:.0f})")
    p0 = p_des.copy()  # ref for workspace clamping
    MAX_DXY = 0.50     # 50cm XY range (matching calib)
    z_offset = 0.0     # height above table, Z/X to adjust (0=just touch)

    # ── Mouse ─────────────────────────────────────────────────────────
    click_uv = None

    def _on_mouse(event, x, y, flags, param):
        nonlocal click_uv
        if event == cv2.EVENT_LBUTTONDOWN:
            click_uv = (x, y)

    WINDOW = "Click-to-Go  |  Left=move EE to point  |  ESC=quit"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.setMouseCallback(WINDOW, _on_mouse)

    z_offset = 0.10  # height above table (adjustable with Z/X)

    print(f"\n  Left-click → move to table + {z_offset*100:.0f}cm")
    print("  Z/X → raise/lower target height")
    print("  SPACE/B → hand tighter/looser")
    print("  ESC → quit\n")

    try:
        while True:
            rgb = cam.snapshot(filtered=False)
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # ── Handle click ──────────────────────────────────────────
            if click_uv is not None:
                u, v = click_uv
                click_uv = None

                # Pixel → world via homography
                p_pix = np.array([u, v, 1.0])
                w = H @ p_pix
                w /= w[2]
                wx, wy = float(w[0]), float(w[1])
                print(f"\n[Click] ({u},{v}) → world=({wx:.3f}, {wy:.3f})")

                # Move: interpolate position to target (RPY already set at startup)
                T_cur = kin.ee_in_base(q_head, q_arm)
                p_start = T_cur[:3, 3].copy()
                # Compute Z from table plane: z = a*x + b*y + c
                # Interpolate Z from 3 nearest calibration points
                if W is not None and len(W) >= 3:
                    dists = np.sqrt((W[:,0] - wx)**2 + (W[:,1] - wy)**2)
                    idx = np.argsort(dists)[:3]
                    if dists[idx[0]] < 1e-6:
                        z_table = W[idx[0], 2]
                    else:
                        wgt = 1.0 / (dists[idx] + 0.001)
                        wgt /= wgt.sum()
                        z_table = float(np.dot(wgt, W[idx, 2]))
                else:
                    z_table = z_ref
                z_target = z_table + z_offset
                # Clamp XY to workspace
                wx = np.clip(wx, p0[0]-MAX_DXY, p0[0]+MAX_DXY)
                wy = np.clip(wy, p0[1]-MAX_DXY, p0[1]+MAX_DXY)
                p_target = np.array([wx, wy, z_target])
                dist = np.linalg.norm(p_target - p_start)
                n_steps = max(1, int(dist / 0.005))

                # ── Interpolated movement with SLSQP IK ─────────────────
                R_cur = kin.ee_in_base(q_head, q_arm)[:3, :3]
                ik_ok = True
                for i in range(n_steps):
                    alpha = (i + 1) / n_steps
                    interp = p_start + alpha * (p_target - p_start)
                    T_tgt = np.eye(4)
                    T_tgt[:3, :3] = R_cur
                    T_tgt[:3, 3] = interp
                    try:
                        # Single-shot SLSQP IK (small step, fast)
                        q_hs, q_as, err, it = scipy_ik(
                            kin, T_tgt, q_head, q_arm,
                            z_weight=3.0, max_iters=200,
                            pos_tol=0.005, tilt_tol_deg=5, yaw_tol_deg=10)
                        if np.isnan(err) or err > 0.02:
                            if i == 0:
                                print(f"  ⚠ IK fail at step 0: err={err:.3f}")
                            ik_ok = False
                            break
                        cmd = np.concatenate([q_hs, q_as])
                        cmd[0] = head_yaw; cmd[1] = head_pitch
                        robot.set_positions(cmd)
                        time.sleep(0.02)
                        q_head, q_arm = kin.split_q(cmd)
                    except Exception as e:
                        print(f"  ✗ IK step {i}: {e}")
                        ik_ok = False
                        break

                if ik_ok:
                    # ── Final refinement: multi-restart SLSQP IK (roboarm-style) ──
                    T_final = np.eye(4)
                    T_final[:3, :3] = R_cur
                    T_final[:3, 3] = p_target
                    try:
                        q_hs, q_as, best_err, _ = scipy_ik_multi_restart(
                            kin, T_final, q_head, q_arm,
                            n_restarts=4, z_weight=3.0,
                            pos_tol=0.005, tilt_tol_deg=5, yaw_tol_deg=10)
                        cmd = np.concatenate([q_hs, q_as])
                        cmd[0] = head_yaw; cmd[1] = head_pitch
                        robot.set_positions(cmd)
                        time.sleep(0.05)
                        q_head, q_arm = kin.split_q(cmd)
                    except Exception as e:
                        print(f"  ✗ Refine IK: {e}")
                        best_err = 999
                    rpy = R.from_matrix(T_final[:3,:3]).as_euler('xyz', degrees=True)
                    print(f"  → ({p_target[0]:.3f},{p_target[1]:.3f},{p_target[2]:.3f})  err={best_err:.4f}")

            # ── Display ────────────────────────────────────────────────
            q_disp = np.asarray(robot.get_positions(), dtype=float)
            _, qa_disp = kin.split_q(q_disp)
            T_disp = kin.ee_in_base(kin.split_q(q_disp)[0], qa_disp)
            ex, ey, ez = T_disp[0, 3], T_disp[1, 3], T_disp[2, 3]
            cv2.putText(vis, f"EE: ({ex:.3f}, {ey:.3f}, {ez:.3f})",
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow(WINDOW, vis)
            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break
            if key == ord(' '):
                hand_level = min(hand_level + 1, len(HAND_LEVELS) - 1)
                pos = HAND_LEVELS[hand_level]
                hand.set_joint_pos(_map_hand(pos))
                print(f"  🖐 {HAND_NAMES[hand_level]} ({pos:.2f})")
            elif key in (ord('b'), ord('B')):
                hand_level = max(hand_level - 1, 0)
                pos = HAND_LEVELS[hand_level]
                hand.set_joint_pos(_map_hand(pos))
                print(f"  🖐 {HAND_NAMES[hand_level]} ({pos:.2f})")
            elif key in (ord('z'), ord('Z')):
                z_offset = min(z_offset + 0.01, 0.20)
                print(f"  📏 height above table: {z_offset*100:.0f}cm")
            elif key in (ord('x'), ord('X')):
                z_offset = max(z_offset - 0.01, -0.15)
                print(f"  📏 height above table: {z_offset*100:.0f}cm")

    except KeyboardInterrupt:
        print("\n[Exit]")
    finally:
        hand.open_hand()
        hand.close_can()
        cam.close()
        robot.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
