#!/usr/bin/env python3
"""
Compare 3 IK solvers on Z-plane consistency:
  1. SDK IK (QP optimizer)
  2. Jacobian IK (damped least-squares)
  3. SLSQP IK (scipy.optimize, roboarm-style)

Usage:
    conda activate bb_d1
    cd ~/Beingbeyond_D1
    python block_grasp/test_ik_compare.py
"""
import math, time, sys, os
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from beingbeyond_d1_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig, IKOptions
from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
from block_grasp.ik_jacobian import jacobian_ik
from block_grasp.ik_scipy import scipy_ik


def slerp_rotate_sdk(kin, q_head, q_arm, R_target, p_fixed):
    """Use SDK IK for SLERP rotation (known to work)."""
    T_cur = kin.ee_in_base(q_head, q_arm)
    R_cur = T_cur[:3, :3]
    q0 = R.from_matrix(R_cur).as_quat()
    q1 = R.from_matrix(R_target).as_quat()
    if np.dot(q0, q1) < 0:
        q1 = -q1
    omega = np.arccos(np.clip(np.dot(q0, q1), -1, 1))
    angle = omega * 2
    n_steps = max(1, math.ceil(angle / 0.05))
    for i in range(n_steps):
        a = (i + 1) / n_steps
        if abs(omega) < 1e-10:
            qi = q0
        else:
            qi = (np.sin((1-a)*omega)*q0 + np.sin(a*omega)*q1) / np.sin(omega)
        Ri = R.from_quat(qi).as_matrix()
        T_step = np.eye(4); T_step[:3,:3] = Ri; T_step[:3,3] = p_fixed
        q_head, q_arm, err, _ = kin.ik_T_ee_with_arm_only(T_step, q_head, q_arm)
    return q_head, q_arm


def main():
    urdf = get_default_urdf_path()
    kin = D1Kinematics(D1KinematicsConfig(urdf_path=urdf))
    head_yaw = math.radians(-10)
    head_pitch = math.radians(35)
    q_head = np.array([head_yaw, head_pitch])
    q_arm0 = np.radians([0, -60, 60, 0, 0, 0])
    R_des = R.from_euler('xyz', [178, 61, -175], degrees=True).as_matrix()

    # ── Startup: lift + SLERP rotate (SDK IK) ──────────────────────
    T0 = kin.ee_in_base(q_head, q_arm0)
    x0, y0 = T0[0,3], T0[1,3]

    T_lift = np.eye(4); T_lift[:3,:3] = T0[:3,:3]
    T_lift[:3,3] = [x0, y0, 0.35]
    q_head, q_arm, err, _ = kin.ik_T_ee_with_arm_only(T_lift, q_head, q_arm0)
    q_head, q_arm = slerp_rotate_sdk(kin, q_head, q_arm, R_des, [x0, y0, 0.35])
    T_rot = kin.ee_in_base(q_head, q_arm)
    rpy = R.from_matrix(T_rot[:3,:3]).as_euler('xyz', degrees=True)
    print(f"Startup: rpy=({rpy[0]:.0f},{rpy[1]:.0f},{rpy[2]:.0f})  "
          f"pos=({T_rot[0,3]:.3f},{T_rot[1,3]:.3f},{T_rot[2,3]:.3f})")

    # ── Z-plane grid ───────────────────────────────────────────────
    z_target = 0.18
    x0, y0 = T_rot[0,3], T_rot[1,3]
    grid = [(x0+dx, y0+dy)
            for dx in np.linspace(-0.10, 0.10, 5)
            for dy in np.linspace(-0.10, 0.10, 5)]

    print(f"\n{'='*80}")
    print(f"  Z-plane grid at z={z_target:.2f} | 5×5 = {len(grid)} points")
    print(f"  Jacobian IK (z_weight=3.0)  vs  SLSQP IK (z_weight=3.0)")
    print(f"{'='*80}")
    print(f"{'x':>8s} {'y':>8s} | {'Jac Z':>8s} {'J err':>8s} {'J it':>5s} | "
          f"{'SLSQP Z':>8s} {'S err':>8s} {'S it':>5s} | {'ΔZ mm':>8s}")
    print("-" * 80)

    jac_z_vals = []
    slsqp_z_vals = []
    jac_times = []
    slsqp_times = []

    for wx, wy in grid:
        T_target = np.eye(4); T_target[:3,:3] = R_des; T_target[:3,3] = [wx, wy, z_target]

        # Jacobian IK
        t0 = time.time()
        q_hj, q_aj, err_j, it_j = jacobian_ik(kin, T_target, q_head, q_arm,
                                                z_weight=3.0, max_iters=300)
        jac_times.append(time.time() - t0)
        T_j = kin.ee_in_base(q_hj, q_aj); zj = T_j[2,3]

        # SLSQP IK
        t0 = time.time()
        q_hs, q_as, err_s, it_s = scipy_ik(kin, T_target, q_head, q_arm,
                                             pos_tol=0.005, tilt_tol_deg=5, yaw_tol_deg=10,
                                             z_weight=3.0, max_iters=200)
        slsqp_times.append(time.time() - t0)
        T_s = kin.ee_in_base(q_hs, q_as); zs = T_s[2,3]

        jac_z_vals.append(zj)
        slsqp_z_vals.append(zs)
        dz = (zj - zs) * 1000

        print(f"{wx:8.3f} {wy:8.3f} | {zj:8.4f} {err_j:8.5f} {it_j:>5d} | "
              f"{zs:8.4f} {err_s:8.5f} {it_s:>5d} | {dz:+8.1f}")

    print("-" * 80)
    jac_z = np.array(jac_z_vals)
    slsqp_z = np.array(slsqp_z_vals)

    print(f"  Jacobian IK: Z ∈ [{jac_z.min():.4f}, {jac_z.max():.4f}]  "
          f"Δ={jac_z.max()-jac_z.min():.4f} ({1000*(jac_z.max()-jac_z.min()):.1f}mm)  "
          f"σ={jac_z.std():.5f}  avg time={np.mean(jac_times)*1000:.1f}ms")
    print(f"  SLSQP IK:    Z ∈ [{slsqp_z.min():.4f}, {slsqp_z.max():.4f}]  "
          f"Δ={slsqp_z.max()-slsqp_z.min():.4f} ({1000*(slsqp_z.max()-slsqp_z.min()):.1f}mm)  "
          f"σ={slsqp_z.std():.5f}  avg time={np.mean(slsqp_times)*1000:.1f}ms")


if __name__ == "__main__":
    main()
