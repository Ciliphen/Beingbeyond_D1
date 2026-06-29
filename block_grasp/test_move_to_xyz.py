#!/usr/bin/env python3
"""Type x y z → arm moves fingertip there. Enter blank to quit."""
import math, time, numpy as np
from beingbeyond_d1_sdk.head_arm import HeadArmRobot
from beingbeyond_d1_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig
from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
from beingbeyond_d1_sdk.dex_hand import DexHand

urdf = get_default_urdf_path()
kin = D1Kinematics(D1KinematicsConfig(urdf_path=urdf))
robot = HeadArmRobot(urdf_path=urdf, dev="/dev/ttyUSB0", baudrate=1_000_000)
hand = DexHand(hand_type="right", can_iface="can0", baudrate=1_000_000)

# Just set head, keep arm where it is
q_cur = np.asarray(robot.get_positions(), dtype=float)
head_yaw = math.radians(-10)
head_pitch = math.radians(35)
q_cur[0] = head_yaw
q_cur[1] = head_pitch
robot.set_positions(q_cur)
robot.wait_until_reached(q_cur, active_joint_indices=[0, 1])
time.sleep(0.3)
hand.set_joint_pos([0.64, 0.8, 0.54, 0.58, 0.0, 0.0])
q_head, q_arm = kin.split_q(q_cur)
T = kin.ee_in_base(q_head, q_arm)
print(f"EE start: ({T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f})")
print("Enter x y z (space-separated, blank to quit):")

try:
    while True:
        line = input("> ").strip()
        if not line:
            break
        parts = line.split()
        if len(parts) != 3:
            print("  Need 3 numbers: x y z")
            continue
        try:
            x, y, z = map(float, parts)
        except ValueError:
            print("  Invalid numbers")
            continue

        T_tgt = np.eye(4)
        T_tgt[:3, :3] = T[:3, :3]
        T_tgt[:3, 3] = [x, y, z]
        try:
            q_hs, q_as, err, it = kin.ik_T_ee_with_arm_only(T_tgt, q_head, q_arm)
            if np.isnan(err) or err > 0.20:
                print(f"  IK fail: err={err:.3f} it={it}")
            else:
                cmd = np.concatenate([q_hs, q_as])
                robot.set_positions(cmd)
                robot.wait_until_reached(cmd, active_joint_indices=range(2, 8))
                q_head, q_arm = kin.split_q(cmd)
                print(f"  → ({x:.3f},{y:.3f},{z:.3f})  err={err:.3f}")
        except Exception as e:
            print(f"  ✗ {e}")
finally:
    hand.open_hand()
    hand.close_can()
    robot.close()
