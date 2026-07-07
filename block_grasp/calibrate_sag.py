#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
Gravity-sag calibration for the D1 arm (companion to run_block_grasp.py).

Moves the EE — at the *grasp posture* — to several radial distances, descends
to the nominal grasp height with **sag compensation disabled**, and pauses so
you can measure the real EE-to-table height with a ruler.  It then fits the
cubic model  dz = factor · dist³  by linear regression and prints the
GRAVITY_SAG_FACTOR to paste into config.py.

Because the fit uses the *slope* of height-vs-dist³, the exact reference point
you measure to (fingertip, flange, …) doesn't matter — the unknown offset is
absorbed into the intercept.  Just measure the same point to the table at every
stop.

Usage:
    conda activate bb_d1
    cd ~/Beingbeyond_D1
    python block_grasp/calibrate_sag.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from beingbeyond_d1_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig
from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
from beingbeyond_d1_sdk.head_arm import HeadArmRobot

from block_grasp.ik_scipy import scipy_ik
from block_grasp.config import (
    CALIB_PATH,
    HEAD_YAW_DEG, HEAD_PITCH_DEG,
    EE_ROLL_DEG, EE_PITCH_DEG, EE_YAW_DEG,
    GRASP_Z_OFFSET, APPROACH_HEIGHT_OFFSET, Z_SAFE,
    INTERP_STEP_SIZE, IK_POS_TOL, IK_TILT_TOL_DEG, IK_YAW_TOL_DEG,
    IK_MAX_ITERS, IK_Z_WEIGHT, IK_FAIL_THRESHOLD, JOINT_JUMP_THR_DEG,
    GRAVITY_SAG_FACTOR,
)

# Calibration points: radial distances (m) straight ahead (y = 0).
# Sag depends on radial distance, so y = 0 is enough to fit the factor.
# Densified around 0.25–0.30 to locate the IK configuration-switch step.
# Edit to match your reachable workspace.
CALIB_X = [0.20, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.35, 0.40]
CALIB_Y = 0.0


def _move_to(robot, kin, q_state, p_target, R_target, label):
    """Cartesian straight-line move (NO sag) with near-singularity guard.

    Returns the new (q_head, q_arm) on success, or None if unreachable.
    """
    q_head, q_arm = q_state
    T_cur = kin.ee_in_base(q_head, q_arm)
    p_start = T_cur[:3, 3].copy()
    n = max(1, int(np.linalg.norm(p_target - p_start) / INTERP_STEP_SIZE))
    max_dq = 0.0
    for i in range(n):
        a = (i + 1) / n
        interp = p_start + a * (p_target - p_start)
        T = np.eye(4)
        T[:3, :3] = R_target
        T[:3, 3] = interp
        q_hs, q_as, err, _ = scipy_ik(
            kin, T, q_head, q_arm,
            z_weight=IK_Z_WEIGHT, pos_tol=IK_POS_TOL,
            tilt_tol_deg=IK_TILT_TOL_DEG, yaw_tol_deg=IK_YAW_TOL_DEG,
            max_iters=IK_MAX_ITERS)
        if np.isnan(err) or err > IK_FAIL_THRESHOLD:
            print(f"  ⚠ IK fail ({label}) step {i}/{n}: err={err:.3f}")
            return None
        dq = float(np.max(np.abs(q_as[:5] - q_arm[:5])))
        max_dq = max(max_dq, dq)
        if dq > math.radians(JOINT_JUMP_THR_DEG):
            print(f"  ⚠ joint jump {math.degrees(dq):.0f}° ({label}) — near singularity")
            return None
        cmd = np.concatenate([q_hs, q_as])
        cmd[0] = math.radians(HEAD_YAW_DEG)
        cmd[1] = math.radians(HEAD_PITCH_DEG)
        robot.set_positions(cmd)
        time.sleep(0.02)
        q_head, q_arm = kin.split_q(cmd)
    # Largest single-step joint change along this move — a big value here (even
    # if below the reject threshold) means the solver reshuffled the arm mid-path.
    print(f"  [{label}] 过程最大单步关节变化 {math.degrees(max_dq):.1f}°")
    return (q_head, q_arm)


def _startup(robot, kin, R_target):
    """Safe posture → head → lift → SLERP-rotate to grasp posture.

    Mirrors BlockGraspController.__init__ so the calibrated droop matches the
    real grasp posture.  Returns (q_head, q_arm).
    """
    print("[Init] Moving to safe posture ...")
    q_init = np.radians([0, 0, 0, -60, 60, 0, 0, 0])
    robot.set_positions(q_init)
    robot.wait_until_reached(q_init, active_joint_indices=range(8))
    time.sleep(0.3)

    q = np.asarray(robot.get_positions(), dtype=float)
    q[0] = math.radians(HEAD_YAW_DEG)
    q[1] = math.radians(HEAD_PITCH_DEG)
    robot.set_positions(q)
    robot.wait_until_reached(q, active_joint_indices=[0, 1])
    time.sleep(0.3)
    q_head, q_arm = kin.split_q(np.asarray(robot.get_positions(), dtype=float))

    # Lift to safe Z (keep current orientation)
    print("[Init] Lifting to safe height ...")
    T_cur = kin.ee_in_base(q_head, q_arm)
    p_lift = T_cur[:3, 3].copy()
    p_lift[2] = Z_SAFE + 0.05
    T_lift = np.eye(4)
    T_lift[:3, :3] = T_cur[:3, :3]
    T_lift[:3, 3] = p_lift
    q_hs, q_as, err, _ = kin.ik_T_ee_with_arm_only(T_lift, q_head, q_arm)
    if err < 0.05:
        cmd = np.concatenate([q_hs, q_as])
        cmd[0] = math.radians(HEAD_YAW_DEG)
        cmd[1] = math.radians(HEAD_PITCH_DEG)
        robot.set_positions(cmd)
        robot.wait_until_reached(cmd, active_joint_indices=range(2, 8))
        q_head, q_arm = kin.split_q(cmd)
    else:
        print(f"  ⚠ Lift IK error: {err:.3f}")

    # SLERP-rotate to grasp RPY (in place)
    print("[Init] Rotating to grasp posture ...")
    T_cur = kin.ee_in_base(q_head, q_arm)
    p_cur = T_cur[:3, 3]
    q0 = R.from_matrix(T_cur[:3, :3]).as_quat()
    q1 = R.from_matrix(R_target).as_quat()
    if np.dot(q0, q1) < 0:
        q1 = -q1
    omega = float(np.arccos(np.clip(np.dot(q0, q1), -1.0, 1.0)))
    n_rot = max(1, int(math.ceil(omega * 2 / 0.05)))
    for i in range(n_rot):
        a = (i + 1) / n_rot
        if abs(omega) < 1e-10:
            qi = q0
        else:
            qi = (np.sin((1 - a) * omega) * q0 +
                  np.sin(a * omega) * q1) / np.sin(omega)
        Ri = R.from_quat(qi).as_matrix()
        T_rt = np.eye(4)
        T_rt[:3, :3] = Ri
        T_rt[:3, 3] = p_cur
        q_hs, q_as, err, _ = kin.ik_T_ee_with_arm_only(T_rt, q_head, q_arm)
        if err < 0.05:
            cmd = np.concatenate([q_hs, q_as])
            cmd[0] = math.radians(HEAD_YAW_DEG)
            cmd[1] = math.radians(HEAD_PITCH_DEG)
            robot.set_positions(cmd)
            time.sleep(0.02)
            q_head, q_arm = kin.split_q(cmd)
    time.sleep(0.3)
    return q_head, q_arm


def _fit(samples):
    """Fit h = K - factor·dist³ ; return (factor, K, r2)."""
    d = np.array([s[0] for s in samples])
    h = np.array([s[1] for s in samples])
    A = np.vstack([d ** 3, np.ones_like(d)]).T
    (slope, K), *_ = np.linalg.lstsq(A, h, rcond=None)
    h_pred = A @ [slope, K]
    ss_res = float(np.sum((h - h_pred) ** 2))
    ss_tot = float(np.sum((h - np.mean(h)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return -slope, K, r2


def main():
    ap = argparse.ArgumentParser(description="D1 重力下垂标定")
    ap.add_argument("--reverse", action="store_true",
                    help="从远到近遍历，与正向对比迟滞（判断机械背隙）")
    args = ap.parse_args()
    xs = list(reversed(CALIB_X)) if args.reverse else list(CALIB_X)

    print("\033[91m⚠ 急停按钮请保持触手可及！机械臂将下降到接近桌面的抓取高度。\033[0m")
    print(f"[遍历方向] {'远→近 (reverse)' if args.reverse else '近→远 (forward)'}\n")

    urdf = get_default_urdf_path()
    kin = D1Kinematics(D1KinematicsConfig(urdf_path=urdf))
    robot = HeadArmRobot(urdf_path=urdf, dev="/dev/ttyUSB0", baudrate=1_000_000)

    # Table height from hand-eye calibration
    calib = np.load(CALIB_PATH, allow_pickle=True)
    if "world_pts" in calib:
        z_table = float(np.median(calib["world_pts"][:, 2]))
    else:
        z_table = 0.08
    z_grasp = z_table + GRASP_Z_OFFSET
    z_appr = z_grasp + APPROACH_HEIGHT_OFFSET

    R_target = R.from_euler(
        "xyz", [EE_ROLL_DEG, EE_PITCH_DEG, EE_YAW_DEG], degrees=True
    ).as_matrix()

    q_state = _startup(robot, kin, R_target)

    print("\n" + "=" * 60)
    print(f"  桌面 z_table={z_table:.3f} m   名义抓取高度 z={z_grasp:.3f} m")
    print(f"  (即末端 EE 原点应在桌面上方 {GRASP_Z_OFFSET*1000:.0f} mm)")
    print(f"  ★ 下垂补偿已关闭，量的是原始下垂 ★")
    print(f"  当前 config 的 GRAVITY_SAG_FACTOR = {GRAVITY_SAG_FACTOR}")
    print("  每点停住后，用尺子量【同一个末端参考点】到桌面的垂直距离(mm)")
    print("=" * 60)

    samples = []  # (dist_m, h_meas_m)
    try:
        for x in xs:
            dist = math.hypot(x, CALIB_Y)
            print(f"\n=== 点 x={x:.2f} y={CALIB_Y:.2f}  径向 dist={dist:.3f} m ===")
            s = _move_to(robot, kin, q_state,
                         np.array([x, CALIB_Y, z_appr]), R_target, "approach")
            if s is None:
                print("  跳过该点（approach 不可达）")
                continue
            q_state = s
            s = _move_to(robot, kin, q_state,
                         np.array([x, CALIB_Y, z_grasp]), R_target, "descend")
            if s is None:
                print("  跳过该点（descend 不可达）")
                _move_to(robot, kin, q_state,
                         np.array([x, CALIB_Y, z_appr]), R_target, "lift")
                continue
            q_state = s
            T = kin.ee_in_base(*q_state)
            j = [f"{math.degrees(v):.1f}" for v in q_state[1]]
            print(f"  已到位，名义 EE z(FK)={T[2,3]:.3f} m（目标 {z_grasp:.3f}）")
            print(f"  臂关节(°): j1={j[0]} j2={j[1]} j3={j[2]} "
                  f"j4={j[3]} j5={j[4]} j6={j[5]}")
            raw = input("  量末端到桌面距离(mm)，直接回车=跳过: ").strip()
            if raw:
                try:
                    h_mm = float(raw)
                    samples.append((dist, h_mm / 1000.0))
                    print(f"  ✓ 记录 dist={dist:.3f}m  h={h_mm:.1f}mm")
                except ValueError:
                    print("  无效输入，跳过")
            # Lift back to approach height before travelling to the next point
            q_state = _move_to(robot, kin, q_state,
                               np.array([x, CALIB_Y, z_appr]), R_target, "lift") or q_state
    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        robot.close()

    # ── Fit ──
    print("\n" + "=" * 60)
    if len(samples) < 2:
        print("样本不足（<2 点），无法拟合。")
        return
    factor, K, r2 = _fit(samples)
    print("拟合模型  h = K − factor·dist³   (h=末端到桌面, dist=径向距离)")
    print(f"{'dist(m)':>8} {'h(mm)':>8} {'下垂(mm)':>10}")
    for di, hi in samples:
        print(f"{di:8.3f} {hi*1000:8.1f} {(K-hi)*1000:10.1f}")
    print(f"\n  建议 GRAVITY_SAG_FACTOR = {factor:.3f}   (R²={r2:.3f})")
    if factor < 0:
        print("  ⚠ factor 为负：远点反而更高？检查测量或桌面是否水平。")
    print("  → 填入 block_grasp/config.py 的 GRAVITY_SAG_FACTOR")
    print("=" * 60)


if __name__ == "__main__":
    main()
