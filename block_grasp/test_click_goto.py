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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from camera.d1_camera_primitive import D1CameraPrimitive
from beingbeyond_d1_sdk.head_arm import HeadArmRobot
from beingbeyond_d1_sdk.dex_hand import DexHand
from beingbeyond_d1_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig
from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path

CALIB = os.path.join(os.path.dirname(__file__), "handeye_calib.npz")
IK_FAIL_THR = 0.10
Z_SAFE = 0.25     # approach height
Z_TOUCH = 0.08    # table height


def main():
    # ── Load calibration ──────────────────────────────────────────────
    data = np.load(CALIB)
    H = data["H"]
    head_yaw = float(data["head_yaw"])
    head_pitch = float(data["head_pitch"])
    print(f"Calib: head yaw={math.degrees(head_yaw):.0f}° pitch={math.degrees(head_pitch):.0f}°")

    urdf = get_default_urdf_path()
    kin = D1Kinematics(D1KinematicsConfig(urdf_path=urdf))

    # ── Init hardware ─────────────────────────────────────────────────
    print("[Init] Robot ...")
    robot = HeadArmRobot(urdf_path=urdf, dev="/dev/ttyUSB0", baudrate=1_000_000)
    hand = DexHand(hand_type="right", can_iface="can0", baudrate=1_000_000)
    print("[Init] Camera ...")
    cam = D1CameraPrimitive(width=1280, height=720, fps=30)

    # ── Safe posture + set head ───────────────────────────────────────
    print("[Init] Posture ...")
    q_init = np.radians([0, 0, 0, -60, 60, 0, 0, 0])
    robot.set_positions(q_init)
    robot.wait_until_reached(q_init, active_joint_indices=range(8))
    time.sleep(0.3)
    q = np.asarray(robot.get_positions(), dtype=float)
    q[0] = head_yaw
    q[1] = head_pitch
    robot.set_positions(q)
    robot.wait_until_reached(q, active_joint_indices=[0, 1])
    time.sleep(0.3)
    hand.set_joint_pos([0.64, 0.8, 0.54, 0.58, 0.0, 0.0])  # closed
    print("       Ready.")

    # ── IK state ──────────────────────────────────────────────────────
    q_cur = np.asarray(robot.get_positions(), dtype=float)
    q_head, q_arm = kin.split_q(q_cur)
    T0 = kin.ee_in_base(q_head, q_arm)
    p_des = T0[:3, 3].copy()
    R_des = T0[:3, :3].copy()

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

    print("\n  Left-click → move fingertip to that table position")
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

                # Move: approach from above → touch → lift
                for step_name, z_target in [("approach", Z_SAFE), ("touch", Z_TOUCH)]:
                    p_des[0] = wx
                    p_des[1] = wy
                    p_des[2] = z_target
                    T_tgt = np.eye(4)
                    T_tgt[:3, :3] = R_des
                    T_tgt[:3, 3] = p_des
                    try:
                        q_hs, q_as, err, it = kin.ik_T_ee_with_arm_only(T_tgt, q_head, q_arm)
                        if np.isnan(err) or err > IK_FAIL_THR:
                            print(f"  ⚠ IK fail ({step_name}): err={err:.3f}")
                            break
                        cmd = np.concatenate([q_hs, q_as])
                        cmd[0] = head_yaw
                        cmd[1] = head_pitch
                        robot.set_positions(cmd)
                        robot.wait_until_reached(cmd, active_joint_indices=range(2, 8))
                        q_head, q_arm = kin.split_q(cmd)
                        print(f"  → {step_name} ({wx:.3f},{wy:.3f},{z_target:.3f})")
                    except Exception as e:
                        print(f"  ✗ {step_name}: {e}")
                        break

            # ── Display ────────────────────────────────────────────────
            q_disp = np.asarray(robot.get_positions(), dtype=float)
            _, qa_disp = kin.split_q(q_disp)
            T_disp = kin.ee_in_base(kin.split_q(q_disp)[0], qa_disp)
            ex, ey, ez = T_disp[0, 3], T_disp[1, 3], T_disp[2, 3]
            cv2.putText(vis, f"EE: ({ex:.3f}, {ey:.3f}, {ez:.3f})",
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow(WINDOW, vis)
            if cv2.waitKey(5) & 0xFF == 27:
                break

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
