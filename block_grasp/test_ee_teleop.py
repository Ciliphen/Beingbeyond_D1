#!/usr/bin/env python3
"""
Simple EE teleop: palm faces down, WASD slides parallel to table.

W/S → forward/back   A/D → left/right
Q/E → up/down        ESC → quit

Usage:
    conda activate bb_d1
    cd ~/Beingbeyond_D1
    python block_grasp/test_ee_teleop.py
"""
import math
import os
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from beingbeyond_d1_sdk.head_arm import HeadArmRobot
from beingbeyond_d1_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig
from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path

STEP = 0.03   # 3cm per press
Z_STEP = 0.02

def main():
    urdf = get_default_urdf_path()
    print("[Init] Robot ...")
    robot = HeadArmRobot(urdf_path=urdf, dev="/dev/ttyUSB0", baudrate=1_000_000)
    kin = D1Kinematics(D1KinematicsConfig(urdf_path=urdf))

    # Start position: current EE pose
    q = np.asarray(robot.get_positions(), dtype=float)
    q_head, q_arm = kin.split_q(q)
    T = kin.ee_in_base(q_head, q_arm)
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    print(f"       EE start: ({x:.3f}, {y:.3f}, {z:.3f})")

    # Palm-down orientation: Z points downward
    R_down = np.array([[1, 0, 0],
                       [0, -1, 0],
                       [0, 0, -1]], dtype=float)
    q_down = R.from_matrix(R_down).as_quat()  # xyzw

    print("\n" + "=" * 50)
    print("  W/S forward/back   A/D left/right   Q/E up/down")
    print(f"  Step: {STEP*100:.0f}cm  Z_step: {Z_STEP*100:.0f}cm  ESC quit")
    print("=" * 50 + "\n")

    try:
        while True:
            # Show current EE position
            q = np.asarray(robot.get_positions(), dtype=float)
            q_head, q_arm = kin.split_q(q)
            T = kin.ee_in_base(q_head, q_arm)
            x, y, z = T[0, 3], T[1, 3], T[2, 3]

            key = cv2.waitKey(100) & 0xFF

            moved = False
            if key == 27:
                break
            elif key in (ord('w'), ord('W')):   y -= STEP; moved = True
            elif key in (ord('s'), ord('S')):   y += STEP; moved = True
            elif key in (ord('a'), ord('A')):   x -= STEP; moved = True
            elif key in (ord('d'), ord('D')):   x += STEP; moved = True
            elif key in (ord('q'), ord('Q')):   z += Z_STEP; moved = True
            elif key in (ord('e'), ord('E')):   z -= Z_STEP; moved = True

            if moved:
                tgt = np.array([x, y, z, q_down[0], q_down[1], q_down[2], q_down[3]], dtype=float)
                try:
                    q_hs, q_as, cost, it = kin.ik_ee_quatpose_with_arm_only(tgt, q_head, q_arm)
                    robot.set_positions(np.concatenate([q_hs, q_as]))
                    print(f"  EE → ({x:.3f}, {y:.3f}, {z:.3f})  cost={cost:.3f} it={it}")
                except Exception as e:
                    print(f"  IK failed: {e}")

    except KeyboardInterrupt:
        print("\n[Exit]")
    finally:
        robot.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
